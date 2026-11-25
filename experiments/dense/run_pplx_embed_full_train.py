#!/usr/bin/env python3
"""Run the offline PPLX 0.6B fixed-600/0 full-corpus TRAIN comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import (  # noqa: E402
    fixed_chunks,
    jsonl_rows,
)
from experiments.dense.run_dense_fixed_600_0 import (  # noqa: E402
    document_layout,
)


MODEL_NAME = "perplexity-ai/pplx-embed-v1-0.6b"
CACHED_REVISION = "2c4d510dd4a732063c31a0f70193e35067b51fd8"
CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
FIXED_METADATA = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/bge-small-en-v1.5_fixed_600_0/chunk_metadata.jsonl"
)
BM25_RANKINGS = (
    PROJECT_ROOT / "experiments/hybrid/results/fixed_600_0/bm25_top100.jsonl"
)
BGE_DENSE_METRICS = (
    PROJECT_ROOT
    / "experiments/matrix/results/train/fixed_600_0/dense/metrics.json"
)
BGE_HYBRID_METRICS = (
    PROJECT_ROOT
    / "experiments/matrix/results/train/fixed_600_0/rrf/metrics.json"
)
CACHE_DIR = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/pplx-embed-v1-0.6b_fixed_600_0"
)
RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments/dense/results/pplx_embed_v1_0.6b_fixed_600_0_train"
)

EXPECTED_DOCUMENTS = 28_482
EXPECTED_CHUNKS = 182_835
EXPECTED_QUERIES = 450
DIMENSION = 1024
DEFAULT_BATCH_SIZE = 32
TOP_K = 100
RRF_DEPTH = 50
RRF_K = 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Document (and full-run query) encoding batch size.",
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        help="Encode only this many streaming batches, validate them, and stop.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.validation_batches is not None and args.validation_batches <= 0:
        parser.error("--validation-batches must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
    os.replace(temporary, path)


def normalize_float32(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(output, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(norms).all():
        raise ValueError("PPLX returned zero-norm or non-finite embeddings")
    output /= norms
    return output


def iter_fixed_600_0() -> Iterable[tuple[dict[str, Any], str]]:
    """Yield fixed-600/0 metadata and model input in canonical corpus order."""
    for document in jsonl_rows(CORPUS):
        document_id = document.get("document_id")
        title = document.get("title")
        text = document.get("text")
        if (
            not isinstance(document_id, str)
            or not document_id
            or not isinstance(title, str)
            or not isinstance(text, str)
            or not text
        ):
            raise ValueError(f"invalid corpus record: {document}")
        for chunk_index, body in enumerate(fixed_chunks(text, 600, 0)):
            yield (
                {
                    "chunk_id": f"{document_id}::{chunk_index:06d}",
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "body_length": len(body),
                },
                f"{title}\n\n{body}",
            )


def validate_staged_vectors(path: Path, expected_rows: int) -> None:
    vectors = np.load(path, mmap_mode="r")
    if vectors.shape != (expected_rows, DIMENSION) or vectors.dtype != np.float32:
        raise ValueError(f"invalid staged vector array: {vectors.shape} {vectors.dtype}")
    for start in range(0, expected_rows, 4096):
        block = np.asarray(vectors[start : start + 4096])
        if not np.isfinite(block).all():
            raise ValueError(f"non-finite staged vectors at row {start}")
        norms = np.linalg.norm(block, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-5):
            raise ValueError(f"non-normalized staged vectors at row {start}")


def stream_encode_documents(
    model: SentenceTransformer,
    staging_path: Path,
    target_rows: int,
    require_complete_corpus: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Encode checked rows batch-by-batch directly into a staging NPY memmap."""
    generated = iter(iter_fixed_600_0())
    expected_metadata = iter(jsonl_rows(FIXED_METADATA))
    output = np.lib.format.open_memmap(
        staging_path,
        mode="w+",
        dtype=np.float32,
        shape=(target_rows, DIMENSION),
    )
    rows_written = 0
    encoding_seconds = 0.0
    native_dtype: str | None = None
    document_count = 0
    previous_document_id: str | None = None
    wall_started = time.perf_counter()

    while rows_written < target_rows:
        batch_metadata: list[dict[str, Any]] = []
        batch_inputs: list[str] = []
        current_batch_size = min(batch_size, target_rows - rows_written)
        for offset in range(current_batch_size):
            try:
                actual_metadata, model_input = next(generated)
            except StopIteration as exc:
                raise ValueError(
                    f"corpus ended at row {rows_written + offset}, expected {target_rows}"
                ) from exc
            try:
                saved_metadata = next(expected_metadata)
            except StopIteration as exc:
                raise ValueError(
                    f"saved metadata ended at row {rows_written + offset}"
                ) from exc
            row_number = rows_written + offset
            if actual_metadata != saved_metadata:
                raise ValueError(
                    f"metadata mismatch at row {row_number}: "
                    f"generated={actual_metadata}, saved={saved_metadata}"
                )
            if actual_metadata["document_id"] != previous_document_id:
                document_count += 1
                previous_document_id = actual_metadata["document_id"]
            batch_metadata.append(actual_metadata)
            batch_inputs.append(model_input)

        torch.cuda.synchronize()
        batch_started = time.perf_counter()
        with torch.inference_mode():
            raw_vectors = model.encode(
                batch_inputs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        torch.cuda.synchronize()
        encoding_seconds += time.perf_counter() - batch_started
        if raw_vectors.shape != (current_batch_size, DIMENSION):
            raise ValueError(
                f"unexpected embedding shape at row {rows_written}: {raw_vectors.shape}"
            )
        current_dtype = str(raw_vectors.dtype)
        if native_dtype is None:
            native_dtype = current_dtype
        elif native_dtype != current_dtype:
            raise ValueError(
                f"embedding dtype changed at row {rows_written}: "
                f"{native_dtype} -> {current_dtype}"
            )
        vectors = normalize_float32(raw_vectors)
        end = rows_written + current_batch_size
        output[rows_written:end] = vectors
        # Verify that this exact checked row range was written successfully.
        written = np.asarray(output[rows_written:end])
        if written.shape[0] != len(batch_metadata) or not np.array_equal(
            written, vectors
        ):
            raise ValueError(f"vector-row write verification failed at row {rows_written}")
        rows_written = end
        output.flush()
        print(
            f"streamed batch {rows_written // batch_size}/"
            f"{(target_rows + batch_size - 1) // batch_size}: "
            f"rows={rows_written}/{target_rows}",
            flush=True,
        )

    if require_complete_corpus:
        sentinel = object()
        if next(generated, sentinel) is not sentinel:
            raise ValueError("corpus contains more chunks than EXPECTED_CHUNKS")
        if next(expected_metadata, sentinel) is not sentinel:
            raise ValueError("saved metadata contains more rows than EXPECTED_CHUNKS")
        if document_count != EXPECTED_DOCUMENTS:
            raise ValueError(
                f"expected {EXPECTED_DOCUMENTS} documents, got {document_count}"
            )

    output.flush()
    del output
    validate_staged_vectors(staging_path, target_rows)
    wall_seconds = time.perf_counter() - wall_started
    return {
        "rows": rows_written,
        "documents_started": document_count,
        "batch_size": batch_size,
        "batches": (rows_written + batch_size - 1) // batch_size,
        "native_output_dtype": native_dtype,
        "encoding_seconds": encoding_seconds,
        "wall_seconds": wall_seconds,
        "chunks_per_second": rows_written / encoding_seconds,
    }


def metrics_from_ranks(ranks: list[int | None]) -> dict[str, Any]:
    count = len(ranks)
    hits_at_5 = sum(rank is not None and rank <= 5 for rank in ranks)
    hits_at_50 = sum(rank is not None and rank <= 50 for rank in ranks)
    return {
        "query_count": count,
        "hits_at_5": hits_at_5,
        "recall_at_5": hits_at_5 / count,
        "hits_at_50": hits_at_50,
        "recall_at_50": hits_at_50 / count,
    }


def fuse_rrf(
    bm25_rows: list[dict[str, Any]], dense_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    ranks: list[int | None] = []
    for bm25, dense in zip(bm25_rows, dense_rows, strict=True):
        if (
            bm25["question_id"] != dense["question_id"]
            or bm25["reference_document_id"] != dense["reference_document_id"]
        ):
            raise ValueError("BM25 and PPLX Dense rows are not aligned")
        candidates: dict[str, dict[str, Any]] = {}
        for route, documents in (
            ("bm25", bm25["top_documents"][:RRF_DEPTH]),
            ("dense", dense["top_documents"][:RRF_DEPTH]),
        ):
            for item in documents:
                candidate = candidates.setdefault(
                    item["document_id"],
                    {
                        "document_id": item["document_id"],
                        "score": 0.0,
                        "bm25_rank": None,
                        "dense_rank": None,
                    },
                )
                candidate[f"{route}_rank"] = item["rank"]
                candidate["score"] += 1.0 / (RRF_K + item["rank"])
        documents = sorted(
            candidates.values(), key=lambda item: (-item["score"], item["document_id"])
        )[:50]
        for rank, item in enumerate(documents, 1):
            item["rank"] = rank
        reference_id = dense["reference_document_id"]
        reference_rank = next(
            (item["rank"] for item in documents if item["document_id"] == reference_id),
            None,
        )
        ranks.append(reference_rank)
        rows.append(
            {
                "question_id": dense["question_id"],
                "reference_document_id": reference_id,
                "reference_document_rank": reference_rank,
                "top_documents": documents,
            }
        )
    return rows, metrics_from_ranks(ranks)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = "cuda"
    device_name = torch.cuda.get_device_name(0)
    print(f"encoding batch_size={args.batch_size}", flush=True)
    print(f"loading cached {MODEL_NAME}@{CACHED_REVISION} on {device_name}", flush=True)
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

    warmup_inputs = [
        model_input
        for _, model_input in itertools.islice(iter_fixed_600_0(), 4)
    ]
    with torch.inference_mode():
        model.encode(warmup_inputs, batch_size=4, show_progress_bar=False)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if args.validation_batches is not None:
        target_rows = args.validation_batches * args.batch_size
        if target_rows > EXPECTED_CHUNKS:
            raise ValueError("validation target exceeds full chunk count")
        with tempfile.TemporaryDirectory(prefix="pplx-stream-validation-", dir="/tmp") as temporary:
            staging_path = Path(temporary) / "embeddings.npy.tmp"
            statistics = stream_encode_documents(
                model,
                staging_path,
                target_rows,
                require_complete_corpus=False,
                batch_size=args.batch_size,
            )
        torch.cuda.synchronize()
        result = {
            "status": "passed",
            "mode": "stream_validation",
            **statistics,
            "device": device_name,
            "peak_gpu_memory_allocated_gib": (
                torch.cuda.max_memory_allocated() / (1024**3)
            ),
            "peak_gpu_memory_reserved_gib": (
                torch.cuda.max_memory_reserved() / (1024**3)
            ),
            "published_full_cache": False,
        }
        print("STREAM_VALIDATION_RESULT=" + json.dumps(result), flush=True)
        return 0

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vectors_path = CACHE_DIR / "embeddings.npy"
    temporary_vectors = CACHE_DIR / ".embeddings.npy.tmp"
    statistics = stream_encode_documents(
        model,
        temporary_vectors,
        EXPECTED_CHUNKS,
        require_complete_corpus=True,
        batch_size=args.batch_size,
    )
    torch.cuda.synchronize()
    encoding_peak_allocated = torch.cuda.max_memory_allocated()
    encoding_peak_reserved = torch.cuda.max_memory_reserved()
    # Only a fully written, row-checked, finite, normalized staging array is published.
    os.replace(temporary_vectors, vectors_path)
    cache_manifest = {
        "status": "complete",
        "model_name": MODEL_NAME,
        "model_revision": CACHED_REVISION,
        "offline": True,
        "corpus_sha256": sha256_file(CORPUS),
        "metadata_sha256": sha256_file(FIXED_METADATA),
        "chunk_config": {
            "name": "fixed_600_0",
            "strategy": "fixed",
            "size": 600,
            "overlap": 0,
            "step": 600,
        },
        "input_format": "document title + two newlines + fixed-600/0 chunk body",
        "query_instruction": None,
        "document_count": EXPECTED_DOCUMENTS,
        "chunk_count": statistics["rows"],
        "embedding_dimension": DIMENSION,
        "native_output_dtype": statistics["native_output_dtype"],
        "saved_dtype": "float32",
        "normalized_for_cosine": True,
        "batch_size": args.batch_size,
        "document_encoding_seconds": statistics["encoding_seconds"],
        "document_encoding_wall_seconds": statistics["wall_seconds"],
        "chunks_per_second": statistics["chunks_per_second"],
        "device": device_name,
    }
    atomic_json(CACHE_DIR / "manifest.json", cache_manifest)

    metadata = list(jsonl_rows(FIXED_METADATA))
    if len(metadata) != EXPECTED_CHUNKS:
        raise ValueError("published cache metadata row count changed")
    document_vectors = np.load(vectors_path, mmap_mode="r")
    queries = list(jsonl_rows(QUERIES))
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} TRAIN queries")

    query_texts = [row["query"] for row in queries]
    with torch.inference_mode():
        query_vectors = model.encode(
            query_texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    query_vectors = normalize_float32(query_vectors)
    torch.cuda.synchronize()
    encoding_peak_allocated = torch.cuda.max_memory_allocated()
    encoding_peak_reserved = torch.cuda.max_memory_reserved()
    del model, query_texts
    torch.cuda.empty_cache()

    document_ids, starts, ends, _ = document_layout(metadata)
    reference_ids = {row["reference_document_id"] for row in queries}
    if not reference_ids <= set(map(str, document_ids)):
        raise ValueError("a TRAIN reference document is missing from the corpus")
    document_tensor = torch.tensor(document_vectors, dtype=torch.float32, device=device)
    query_tensor = torch.tensor(query_vectors, dtype=torch.float32, device=device)
    dense_rows = []
    dense_ranks: list[int | None] = []
    print("retrieving 450 TRAIN queries", flush=True)
    for query_index, query in enumerate(queries):
        chunk_scores = torch.mv(document_tensor, query_tensor[query_index]).cpu().numpy()
        document_scores = np.maximum.reduceat(chunk_scores, starts)
        order = np.lexsort((document_ids, -document_scores))[:TOP_K]
        documents = []
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
        reference_id = query["reference_document_id"]
        reference_rank = next(
            (item["rank"] for item in documents if item["document_id"] == reference_id),
            None,
        )
        dense_ranks.append(reference_rank)
        dense_rows.append(
            {
                "question_id": query["question_id"],
                "reference_document_id": reference_id,
                "reference_document_rank": reference_rank,
                "top_documents": documents,
            }
        )
        if (query_index + 1) % 50 == 0:
            print(f"retrieved {query_index + 1}/{EXPECTED_QUERIES}", flush=True)
    dense_metrics = metrics_from_ranks(dense_ranks)

    bm25_rows = list(jsonl_rows(BM25_RANKINGS))
    if len(bm25_rows) != EXPECTED_QUERIES:
        raise ValueError("existing BM25 rankings do not contain 450 rows")
    hybrid_rows, hybrid_metrics = fuse_rrf(bm25_rows, dense_rows)
    bge_dense = json.loads(BGE_DENSE_METRICS.read_text(encoding="utf-8"))
    bge_hybrid = json.loads(BGE_HYBRID_METRICS.read_text(encoding="utf-8"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dense_path = RESULTS_DIR / "dense_rankings_top100.jsonl"
    hybrid_path = RESULTS_DIR / "hybrid_rankings_top50.jsonl"
    atomic_jsonl(dense_path, dense_rows)
    atomic_jsonl(hybrid_path, hybrid_rows)
    comparison = {
        "status": "complete",
        "evaluation_split": "TRAIN only",
        "chunk_config": "fixed_600_0",
        "pplx_dense": dense_metrics,
        "pplx_hybrid": {
            **hybrid_metrics,
            "rrf_k": RRF_K,
            "input_depth_per_route": RRF_DEPTH,
            "bm25_weight": 1,
            "dense_weight": 1,
        },
        "bge_small_dense": {
            "recall_at_5": bge_dense["recall_at_5"],
            "recall_at_50": bge_dense["recall_at_50"],
        },
        "bge_small_hybrid": {
            "recall_at_5": bge_hybrid["recall_at_5"],
            "recall_at_50": bge_hybrid["recall_at_50"],
        },
        "delta_pplx_minus_bge_small": {
            "dense_recall_at_5": dense_metrics["recall_at_5"] - bge_dense["recall_at_5"],
            "dense_recall_at_50": dense_metrics["recall_at_50"] - bge_dense["recall_at_50"],
            "hybrid_recall_at_5": hybrid_metrics["recall_at_5"] - bge_hybrid["recall_at_5"],
            "hybrid_recall_at_50": hybrid_metrics["recall_at_50"] - bge_hybrid["recall_at_50"],
        },
        "encoding": {
            **cache_manifest,
            "peak_gpu_memory_allocated_gib": encoding_peak_allocated / (1024**3),
            "peak_gpu_memory_reserved_gib": encoding_peak_reserved / (1024**3),
        },
        "artifacts": {
            "cache": str(CACHE_DIR.relative_to(PROJECT_ROOT)),
            "dense_rankings": str(dense_path.relative_to(PROJECT_ROOT)),
            "hybrid_rankings": str(hybrid_path.relative_to(PROJECT_ROOT)),
            "bm25_rankings_reused": str(BM25_RANKINGS.relative_to(PROJECT_ROOT)),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "numpy": np.__version__,
        },
    }
    atomic_json(RESULTS_DIR / "metrics.json", comparison)
    report = (
        "# PPLX 0.6B fixed 600/0 TRAIN 对照\n\n"
        "结果见 `metrics.json`。本运行仅使用 TRAIN；未读取或评测 DEV。\n\n"
        "- Dense: "
        f"R@5={dense_metrics['recall_at_5']:.4f}, R@50={dense_metrics['recall_at_50']:.4f}\n"
        "- Hybrid: "
        f"R@5={hybrid_metrics['recall_at_5']:.4f}, R@50={hybrid_metrics['recall_at_50']:.4f}\n"
    )
    atomic_text(RESULTS_DIR / "report.md", report)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
