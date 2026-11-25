#!/usr/bin/env python3
"""Final offline PPLX Dense evaluation on 147 answerable, deduplicated DEV queries."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import time
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


DEV_QUERIES = PROJECT_ROOT / "data/processed/techqa/dev_queries.jsonl"
CACHE_DIR = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/pplx-embed-v1-0.6b_fixed_600_0"
)
VECTORS = CACHE_DIR / "embeddings.npy"
MANIFEST = CACHE_DIR / "manifest.json"
RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments/dense/results/pplx_embed_v1_0.6b_fixed_600_0_dev"
)
EXPECTED_QUERIES = 147
QUERY_BATCH_SIZE = 16
TOP_K = 50
RECALL_DEPTHS = (1, 5, 10, 20, 50)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    os.replace(temporary, path)


def validate_cache() -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_manifest = {
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
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"PPLX cache manifest mismatch for {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    metadata = list(jsonl_rows(FIXED_METADATA))
    vectors = np.load(VECTORS, mmap_mode="r")
    if len(metadata) != EXPECTED_CHUNKS:
        raise ValueError(f"expected {EXPECTED_CHUNKS} metadata rows, got {len(metadata)}")
    if vectors.shape != (EXPECTED_CHUNKS, DIMENSION) or vectors.dtype != np.float32:
        raise ValueError(f"invalid PPLX vector cache: {vectors.shape} {vectors.dtype}")
    for start in range(0, EXPECTED_CHUNKS, 4096):
        block = np.asarray(vectors[start : start + 4096])
        if not np.isfinite(block).all():
            raise ValueError(f"non-finite cached vectors at row {start}")
        if not np.allclose(np.linalg.norm(block, axis=1), 1.0, atol=2e-5):
            raise ValueError(f"non-normalized cached vectors at row {start}")
    return manifest, vectors, metadata


def load_queries() -> list[dict[str, Any]]:
    queries = list(jsonl_rows(DEV_QUERIES))
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} DEV queries, got {len(queries)}")
    question_ids = [row.get("question_id") for row in queries]
    if len(set(question_ids)) != EXPECTED_QUERIES:
        raise ValueError("DEV question IDs are not unique")
    for row in queries:
        if not isinstance(row.get("query"), str) or not row["query"]:
            raise ValueError(f"invalid DEV query: {row.get('question_id')}")
        if not isinstance(row.get("reference_document_id"), str):
            raise ValueError(f"invalid DEV reference: {row.get('question_id')}")
    return queries


def metrics(ranks: Sequence[int | None]) -> dict[str, Any]:
    output: dict[str, Any] = {"query_count": len(ranks)}
    for depth in RECALL_DEPTHS:
        hits = sum(rank is not None and rank <= depth for rank in ranks)
        output[f"hits_at_{depth}"] = hits
        output[f"recall_at_{depth}"] = hits / len(ranks)
        output[f"recall_at_{depth}_percent"] = 100.0 * hits / len(ranks)
    output["mrr_at_10"] = sum(
        1.0 / rank if rank is not None and rank <= 10 else 0.0 for rank in ranks
    ) / len(ranks)
    return output


def main() -> int:
    wall_started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = "cuda"
    device_name = torch.cuda.get_device_name(0)
    manifest, document_vectors, metadata = validate_cache()
    queries = load_queries()
    document_ids, starts, ends, _ = document_layout(metadata)
    if len(document_ids) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"expected {EXPECTED_DOCUMENTS} documents, got {len(document_ids)}"
        )
    reference_ids = {row["reference_document_id"] for row in queries}
    if not reference_ids <= set(map(str, document_ids)):
        raise ValueError("a DEV reference document is missing from the fixed corpus")
    print(
        f"offline DEV evaluation: model={MODEL_NAME}@{CACHED_REVISION} "
        f"documents={len(document_ids)} chunks={len(metadata)} queries={len(queries)}",
        flush=True,
    )
    print(
        f"loading cached model on {device_name}; query_instruction=None "
        f"query_batch_size={QUERY_BATCH_SIZE}",
        flush=True,
    )
    model = SentenceTransformer(
        MODEL_NAME,
        revision=CACHED_REVISION,
        trust_remote_code=True,
        local_files_only=True,
        device=device,
    )
    model.eval()
    if model.get_embedding_dimension() != DIMENSION:
        raise ValueError("unexpected PPLX embedding dimension")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    encoding_started = time.perf_counter()
    with torch.inference_mode():
        query_vectors = model.encode(
            [row["query"] for row in queries],
            batch_size=QUERY_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    torch.cuda.synchronize()
    query_encoding_seconds = time.perf_counter() - encoding_started
    query_vectors = normalize_float32(query_vectors)
    query_encoding_peak_allocated = torch.cuda.max_memory_allocated()
    query_encoding_peak_reserved = torch.cuda.max_memory_reserved()
    del model
    torch.cuda.empty_cache()

    document_tensor = torch.tensor(document_vectors, dtype=torch.float32, device=device)
    query_tensor = torch.tensor(query_vectors, dtype=torch.float32, device=device)
    dense_rows: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    retrieval_started = time.perf_counter()
    print("retrieving all 147 DEV queries over every cached document chunk", flush=True)
    for query_index, query in enumerate(queries):
        # Rank independently of the reference label. The label is consulted only below.
        chunk_scores = torch.mv(document_tensor, query_tensor[query_index]).cpu().numpy()
        document_scores = np.maximum.reduceat(chunk_scores, starts)
        # Match TRAIN: descending score, then ascending document ID for exact ties.
        order = np.lexsort((document_ids, -document_scores))[:TOP_K]
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
        if len(documents) != TOP_K or len(
            {item["document_id"] for item in documents}
        ) != TOP_K:
            raise ValueError(f"invalid Top-50 for {query['question_id']}")
        reference_id = query["reference_document_id"]
        reference_rank = next(
            (item["rank"] for item in documents if item["document_id"] == reference_id),
            None,
        )
        ranks.append(reference_rank)
        dense_rows.append(
            {
                "question_id": query["question_id"],
                "reference_document_id": reference_id,
                "reference_document_rank": reference_rank,
                "top_documents": documents,
            }
        )
        if (query_index + 1) % 25 == 0 or query_index + 1 == EXPECTED_QUERIES:
            print(f"retrieved {query_index + 1}/{EXPECTED_QUERIES}", flush=True)
    torch.cuda.synchronize()
    retrieval_seconds = time.perf_counter() - retrieval_started

    if len(dense_rows) != EXPECTED_QUERIES or any(
        len(row["top_documents"]) != TOP_K
        or len({item["document_id"] for item in row["top_documents"]}) != TOP_K
        for row in dense_rows
    ):
        raise ValueError("final DEV Top-50 validation failed")
    result_metrics = metrics(ranks)
    summary = {
        "status": "complete",
        "evaluation_split": "DEV answerable, exact TRAIN duplicates excluded",
        "query_count": EXPECTED_QUERIES,
        "ranking_depth": TOP_K,
        "all_queries_have_50_distinct_documents": True,
        "configuration": {
            "chunk_config": "fixed_600_0",
            "retriever": "dense",
            "model_name": MODEL_NAME,
            "model_revision": CACHED_REVISION,
            "query_instruction": None,
            "document_aggregation": "maximum chunk score",
            "similarity": "normalized vector dot product",
            "tie_break": "score descending, document_id ascending",
        },
        "cache": {
            "directory": str(CACHE_DIR.relative_to(PROJECT_ROOT)),
            "reused": True,
            "document_count": EXPECTED_DOCUMENTS,
            "chunk_count": EXPECTED_CHUNKS,
            "manifest": manifest,
        },
        "inputs": {
            "dev_queries": str(DEV_QUERIES.relative_to(PROJECT_ROOT)),
            "dev_queries_sha256": sha256_file(DEV_QUERIES),
            "chunk_metadata": str(FIXED_METADATA.relative_to(PROJECT_ROOT)),
        },
        "metrics": result_metrics,
        "timing": {
            "query_encoding_seconds": query_encoding_seconds,
            "retrieval_seconds": retrieval_seconds,
            "experiment_wall_seconds": time.perf_counter() - wall_started,
        },
        "query_encoding_peak_gpu_memory_allocated_gib": (
            query_encoding_peak_allocated / (1024**3)
        ),
        "query_encoding_peak_gpu_memory_reserved_gib": (
            query_encoding_peak_reserved / (1024**3)
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "numpy": np.__version__,
            "device": device_name,
        },
    }
    rankings_path = RESULTS_DIR / "rankings_top50.jsonl"
    summary_path = RESULTS_DIR / "summary.json"
    atomic_jsonl(rankings_path, dense_rows)
    atomic_json(summary_path, summary)
    print("PPLX_DEV_RESULT=" + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"rankings={rankings_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
