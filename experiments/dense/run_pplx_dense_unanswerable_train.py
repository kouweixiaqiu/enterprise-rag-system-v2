#!/usr/bin/env python3
"""Retrieve the 150 unanswerable TechQA TRAIN queries with cached PPLX fixed-600/0 vectors.

These records have no relevant-document labels, so this script deliberately does not
report Recall or MRR.  It saves the complete document Top-100 plus score, timing, and
resource diagnostics that are valid for an unlabelled query set.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.dense.run_dense_fixed_600_0 import document_layout, jsonl_rows  # noqa: E402
from experiments.dense.run_pplx_embed_full_train import (  # noqa: E402
    CACHED_REVISION,
    DIMENSION,
    EXPECTED_CHUNKS,
    EXPECTED_DOCUMENTS,
    FIXED_METADATA,
    MODEL_NAME,
    normalize_float32,
    sha256_file,
)


RAW_TRAIN = (
    PROJECT_ROOT
    / "data/raw/techqa/TechQA/training_and_dev/training_Q_A.json"
)
CACHE_DIR = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/pplx-embed-v1-0.6b_fixed_600_0"
)
VECTORS = CACHE_DIR / "embeddings.npy"
MANIFEST = CACHE_DIR / "manifest.json"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/dense/results/pplx_embed_v1_0.6b_fixed_600_0_train_unanswerable"
)
EXPECTED_QUERIES = 150
DEFAULT_QUERY_BATCH_SIZE = 16
DEFAULT_TOP_K = 100
SCORE_DEPTHS = (1, 5, 10, 20, 50, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.query_batch_size <= 0:
        parser.error("--query-batch-size must be positive")
    if args.top_k < max(SCORE_DEPTHS):
        parser.error(f"--top-k must be at least {max(SCORE_DEPTHS)}")
    return args


class RunLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.stream.write(line + "\n")

    def close(self) -> None:
        self.stream.close()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    os.replace(temporary, path)


def build_query(record: dict[str, Any], index: int) -> str:
    title = record.get("QUESTION_TITLE")
    body = record.get("QUESTION_TEXT")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"unanswerable TRAIN record {index} has no QUESTION_TITLE")
    if body is None:
        body = ""
    if not isinstance(body, str):
        raise ValueError(f"unanswerable TRAIN record {index} has invalid QUESTION_TEXT")
    title, body = title.strip(), body.strip()
    return f"{title}\n\n{body}" if body else title


def load_queries() -> list[dict[str, str]]:
    raw = json.loads(RAW_TRAIN.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{RAW_TRAIN} does not contain a list")
    queries: list[dict[str, str]] = []
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"TRAIN record {index} is not an object")
        if record.get("ANSWERABLE") != "N":
            continue
        question_id = record.get("QUESTION_ID")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"unanswerable TRAIN record {index} has no QUESTION_ID")
        if record.get("DOCUMENT") != "-" or record.get("ANSWER") != "-":
            raise ValueError(
                f"{question_id} is marked unanswerable but has a reference document/answer"
            )
        queries.append({"question_id": question_id, "query": build_query(record, index)})
    ids = [row["question_id"] for row in queries]
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} unanswerable TRAIN queries, got {len(queries)}")
    if len(set(ids)) != len(ids):
        raise ValueError("unanswerable TRAIN question IDs are not unique")
    return queries


def validate_cache() -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "model_name": MODEL_NAME,
        "model_revision": CACHED_REVISION,
        "offline": True,
        "document_count": EXPECTED_DOCUMENTS,
        "chunk_count": EXPECTED_CHUNKS,
        "embedding_dimension": DIMENSION,
        "query_instruction": None,
        "normalized_for_cosine": True,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"PPLX cache manifest mismatch for {key}: {manifest.get(key)!r} != {value!r}"
            )
    metadata = list(jsonl_rows(FIXED_METADATA))
    vectors = np.load(VECTORS, mmap_mode="r")
    if len(metadata) != EXPECTED_CHUNKS:
        raise ValueError(f"expected {EXPECTED_CHUNKS} metadata rows, got {len(metadata)}")
    if vectors.shape != (EXPECTED_CHUNKS, DIMENSION) or vectors.dtype != np.float32:
        raise ValueError(f"invalid PPLX vector cache: shape={vectors.shape}, dtype={vectors.dtype}")
    return manifest, vectors, metadata


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def run(args: argparse.Namespace, logger: RunLogger) -> int:
    wall_started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_name = torch.cuda.get_device_name(0)
    manifest, document_vectors, metadata = validate_cache()
    queries = load_queries()
    document_ids, starts, ends, _ = document_layout(metadata)
    if len(document_ids) != EXPECTED_DOCUMENTS:
        raise ValueError(f"expected {EXPECTED_DOCUMENTS} documents, got {len(document_ids)}")

    logger.log(
        f"starting: split=TRAIN ANSWERABLE=N queries={len(queries)} "
        f"chunk_config=fixed_600_0 documents={len(document_ids)} chunks={len(metadata)}"
    )
    logger.log(
        f"model={MODEL_NAME}@{CACHED_REVISION} offline=true query_instruction=None "
        f"query_batch_size={args.query_batch_size} top_k={args.top_k} device={device_name}"
    )
    model = SentenceTransformer(
        MODEL_NAME,
        revision=CACHED_REVISION,
        trust_remote_code=True,
        local_files_only=True,
        device="cuda",
    )
    model.eval()
    model.default_prompt_name = None
    if model.get_embedding_dimension() != DIMENSION:
        raise ValueError(f"unexpected PPLX embedding dimension: {model.get_embedding_dimension()}")
    model_dtype = str(next(model.parameters()).dtype)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    encoding_started = time.perf_counter()
    with torch.inference_mode():
        query_vectors = model.encode(
            [row["query"] for row in queries],
            batch_size=args.query_batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    torch.cuda.synchronize()
    encoding_seconds = time.perf_counter() - encoding_started
    query_vectors = normalize_float32(query_vectors)
    encoding_peak_allocated = torch.cuda.max_memory_allocated()
    encoding_peak_reserved = torch.cuda.max_memory_reserved()
    logger.log(
        f"encoded {len(queries)}/{len(queries)} queries in {encoding_seconds:.3f}s "
        f"({len(queries) / encoding_seconds:.3f} queries/s), model_dtype={model_dtype}"
    )
    del model
    torch.cuda.empty_cache()

    document_tensor = torch.tensor(document_vectors, dtype=torch.float32, device="cuda")
    query_tensor = torch.tensor(query_vectors, dtype=torch.float32, device="cuda")
    rows: list[dict[str, Any]] = []
    per_query: list[dict[str, Any]] = []
    scores_at_depth: dict[int, list[float]] = {depth: [] for depth in SCORE_DEPTHS}
    top1_top2_margins: list[float] = []
    top1_top5_margins: list[float] = []
    retrieval_started = time.perf_counter()
    for query_index, query in enumerate(queries):
        chunk_scores = torch.mv(document_tensor, query_tensor[query_index]).cpu().numpy()
        document_scores = np.maximum.reduceat(chunk_scores, starts)
        order = np.lexsort((document_ids, -document_scores))[: args.top_k]
        documents: list[dict[str, Any]] = []
        for rank, document_index_value in enumerate(order, 1):
            document_index = int(document_index_value)
            start, end = int(starts[document_index]), int(ends[document_index])
            representative = start + int(np.argmax(chunk_scores[start:end]))
            documents.append(
                {
                    "rank": rank,
                    "document_id": str(document_ids[document_index]),
                    "score": float(document_scores[document_index]),
                    "representative_chunk_id": metadata[representative]["chunk_id"],
                }
            )
        if len(documents) != args.top_k or len({row["document_id"] for row in documents}) != args.top_k:
            raise ValueError(f"invalid Top-{args.top_k} for {query['question_id']}")
        score_map = {depth: documents[depth - 1]["score"] for depth in SCORE_DEPTHS}
        for depth, score in score_map.items():
            scores_at_depth[depth].append(score)
        margin12 = documents[0]["score"] - documents[1]["score"]
        margin15 = documents[0]["score"] - documents[4]["score"]
        top1_top2_margins.append(margin12)
        top1_top5_margins.append(margin15)
        rows.append({"question_id": query["question_id"], "top_documents": documents})
        per_query.append(
            {
                "question_id": query["question_id"],
                "scores_at_rank": {str(depth): score_map[depth] for depth in SCORE_DEPTHS},
                "top1_minus_top2": margin12,
                "top1_minus_top5": margin15,
            }
        )
        completed = query_index + 1
        if completed % 25 == 0 or completed == len(queries):
            logger.log(f"retrieved {completed}/{len(queries)} queries")
    torch.cuda.synchronize()
    retrieval_seconds = time.perf_counter() - retrieval_started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    wall_seconds = time.perf_counter() - wall_started

    metrics = {
        "status": "complete",
        "evaluation_split": "TRAIN ANSWERABLE=N",
        "query_count": len(queries),
        "ranking_depth": args.top_k,
        "all_queries_have_distinct_top_k_documents": True,
        "label_note": (
            "These questions have DOCUMENT='-' and ANSWER='-'. Recall and MRR are undefined "
            "and are intentionally not reported. DOC_IDS is not treated as relevance ground truth."
        ),
        "configuration": {
            "chunk_config": "fixed_600_0",
            "retriever": "PPLX Dense",
            "model_name": MODEL_NAME,
            "model_revision": CACHED_REVISION,
            "model_parameter_dtype": model_dtype,
            "query_instruction": None,
            "query_batch_size": args.query_batch_size,
            "similarity": "normalized vector dot product",
            "document_aggregation": "maximum chunk score",
            "tie_break": "score descending, document_id ascending",
        },
        "score_distributions": {
            f"rank_{depth}": distribution(scores_at_depth[depth]) for depth in SCORE_DEPTHS
        },
        "score_margin_distributions": {
            "top1_minus_top2": distribution(top1_top2_margins),
            "top1_minus_top5": distribution(top1_top5_margins),
        },
        "timing": {
            "query_encoding_seconds": encoding_seconds,
            "query_encoding_queries_per_second": len(queries) / encoding_seconds,
            "retrieval_seconds": retrieval_seconds,
            "retrieval_queries_per_second": len(queries) / retrieval_seconds,
            "experiment_wall_seconds": wall_seconds,
        },
        "resources": {
            "device": device_name,
            "query_encoding_peak_gpu_allocated_gib": encoding_peak_allocated / (1024**3),
            "query_encoding_peak_gpu_reserved_gib": encoding_peak_reserved / (1024**3),
            "whole_run_peak_gpu_allocated_gib": peak_allocated / (1024**3),
            "whole_run_peak_gpu_reserved_gib": peak_reserved / (1024**3),
            "process_peak_ram_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
        },
        "cache": {
            "directory": str(CACHE_DIR.relative_to(PROJECT_ROOT)),
            "reused": True,
            "manifest": manifest,
        },
        "inputs": {
            "raw_train": str(RAW_TRAIN.relative_to(PROJECT_ROOT)),
            "raw_train_sha256": sha256_file(RAW_TRAIN),
            "chunk_metadata": str(FIXED_METADATA.relative_to(PROJECT_ROOT)),
            "chunk_metadata_sha256": sha256_file(FIXED_METADATA),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "numpy": np.__version__,
        },
    }
    rankings_path = args.output_dir / "dense_rankings_top100.jsonl"
    diagnostics_path = args.output_dir / "per_query_metrics.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    atomic_jsonl(rankings_path, rows)
    atomic_jsonl(diagnostics_path, per_query)
    atomic_json(metrics_path, metrics)
    logger.log(
        f"complete: wall={wall_seconds:.3f}s rankings={rankings_path} "
        f"per_query_metrics={diagnostics_path} metrics={metrics_path}"
    )
    return 0


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    logger = RunLogger(args.output_dir / "run.log")
    try:
        return run(args, logger)
    except BaseException:
        logger.log("FAILED\n" + traceback.format_exc())
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
