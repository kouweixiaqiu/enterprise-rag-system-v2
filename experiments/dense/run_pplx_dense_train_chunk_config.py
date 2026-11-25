#!/usr/bin/env python3
"""Offline PPLX Dense TRAIN retrieval for a fixed character chunk configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import fixed_chunks, jsonl_rows  # noqa: E402
from experiments.dense.run_dense_fixed_600_0 import document_layout  # noqa: E402


MODEL_NAME = "perplexity-ai/pplx-embed-v1-0.6b"
MODEL_REVISION = "2c4d510dd4a732063c31a0f70193e35067b51fd8"
MODEL_SLUG = "pplx-embed-v1-0.6b"
CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
BGE_CACHE_ROOT = PROJECT_ROOT / "data/processed/techqa/dense_cache"
PPLX_CACHE_ROOT = PROJECT_ROOT / "data/processed/techqa/dense_cache"
RESULTS_ROOT = PROJECT_ROOT / "experiments/dense/results"
QUERY_CACHE = PPLX_CACHE_ROOT / f"{MODEL_SLUG}_train_queries_no_instruction"
EXPECTED_DOCUMENTS = 28_482
EXPECTED_QUERIES = 450
DIMENSION = 1024
TOP_K = 100
DEFAULT_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--chunk-overlap", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--force-reencode",
        action="store_true",
        help="Replace an otherwise compatible complete document-vector cache.",
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must satisfy 0 <= overlap < size")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def chunk_name(size: int, overlap: int) -> str:
    return f"fixed_{size}_{overlap}"


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
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


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


def iter_chunks(size: int, overlap: int) -> Iterator[tuple[dict[str, Any], str]]:
    for document in jsonl_rows(CORPUS):
        document_id = document.get("document_id")
        title = document.get("title")
        text = document.get("text")
        if not all(isinstance(value, str) and value for value in (document_id, title, text)):
            raise ValueError(f"invalid corpus row: {document}")
        for index, body in enumerate(fixed_chunks(text, size, overlap)):
            yield (
                {
                    "chunk_id": f"{document_id}::{index:06d}",
                    "document_id": document_id,
                    "chunk_index": index,
                    "body_length": len(body),
                },
                f"{title}\n\n{body}",
            )


def validate_and_publish_metadata(
    size: int, overlap: int, cache_dir: Path
) -> list[dict[str, Any]]:
    name = chunk_name(size, overlap)
    bge_path = BGE_CACHE_ROOT / f"bge-small-en-v1.5_{name}" / "chunk_metadata.jsonl"
    if not bge_path.is_file():
        raise FileNotFoundError(
            f"missing canonical BGE chunk metadata for {name}: {bge_path}"
        )
    expected = list(jsonl_rows(bge_path))
    generated = iter_chunks(size, overlap)
    document_ids: set[str] = set()
    for row_number, saved in enumerate(expected):
        try:
            actual, _ = next(generated)
        except StopIteration as exc:
            raise ValueError(f"corpus ended before metadata row {row_number}") from exc
        if actual != saved:
            raise ValueError(
                f"{name} boundary mismatch at row {row_number}: "
                f"generated={actual}, saved={saved}"
            )
        document_ids.add(actual["document_id"])
    if next(generated, None) is not None:
        raise ValueError("corpus generated more chunks than canonical metadata")
    if len(document_ids) != EXPECTED_DOCUMENTS:
        raise ValueError(f"expected {EXPECTED_DOCUMENTS} documents, got {len(document_ids)}")
    atomic_jsonl(cache_dir / "chunk_metadata.jsonl", expected)
    return expected


def load_model() -> SentenceTransformer:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
        device="cuda",
    )
    model.eval()
    if model.get_embedding_dimension() != DIMENSION:
        raise ValueError("unexpected PPLX embedding dimension")
    return model


def validate_vectors(path: Path, rows: int) -> np.ndarray:
    vectors = np.load(path, mmap_mode="r")
    if vectors.shape != (rows, DIMENSION) or vectors.dtype != np.float32:
        raise ValueError(f"invalid vector cache: {vectors.shape} {vectors.dtype}")
    for start in range(0, rows, 4096):
        block = np.asarray(vectors[start : start + 4096])
        if not np.isfinite(block).all() or not np.allclose(
            np.linalg.norm(block, axis=1), 1.0, atol=2e-5
        ):
            raise ValueError(f"invalid normalized vectors at row {start}")
    return vectors


def encode_documents(
    model: SentenceTransformer,
    metadata: list[dict[str, Any]],
    size: int,
    overlap: int,
    batch_size: int,
    output_path: Path,
) -> dict[str, Any]:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(metadata), DIMENSION)
    )
    source = iter_chunks(size, overlap)
    started = time.perf_counter()
    encoding_seconds = 0.0
    rows_written = 0
    while rows_written < len(metadata):
        batch_rows: list[dict[str, Any]] = []
        inputs: list[str] = []
        for _ in range(min(batch_size, len(metadata) - rows_written)):
            row, model_input = next(source)
            if row != metadata[rows_written + len(batch_rows)]:
                raise ValueError(f"metadata changed at row {rows_written + len(batch_rows)}")
            batch_rows.append(row)
            inputs.append(model_input)
        torch.cuda.synchronize()
        batch_started = time.perf_counter()
        with torch.inference_mode():
            vectors = model.encode(
                inputs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        torch.cuda.synchronize()
        encoding_seconds += time.perf_counter() - batch_started
        vectors = normalize_float32(vectors)
        end = rows_written + len(batch_rows)
        output[rows_written:end] = vectors
        rows_written = end
        output.flush()
        print(f"encoded chunks={rows_written}/{len(metadata)}", flush=True)
    if next(source, None) is not None:
        raise ValueError("chunk iterator contains unexpected trailing rows")
    output.flush()
    del output
    validate_vectors(temporary, len(metadata))
    os.replace(temporary, output_path)
    wall_seconds = time.perf_counter() - started
    return {
        "document_encoding_seconds": encoding_seconds,
        "document_encoding_wall_seconds": wall_seconds,
        "chunks_per_second": len(metadata) / encoding_seconds,
    }


def cache_is_compatible(
    manifest_path: Path,
    vectors_path: Path,
    metadata_path: Path,
    name: str,
    rows: int,
) -> bool:
    if not all(path.is_file() for path in (manifest_path, vectors_path, metadata_path)):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "chunk_config": name,
        "chunk_count": rows,
        "embedding_dimension": DIMENSION,
        "normalized": True,
        "query_instruction": None,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def load_or_encode_queries(
    model: SentenceTransformer | None,
    queries: list[dict[str, Any]],
    batch_size: int,
) -> tuple[np.ndarray, SentenceTransformer | None]:
    vectors_path = QUERY_CACHE / "embeddings.npy"
    manifest_path = QUERY_CACHE / "manifest.json"
    query_sha = sha256_file(QUERIES)
    compatible = False
    if vectors_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatible = all(
            (
                manifest.get("status") == "complete",
                manifest.get("model_name") == MODEL_NAME,
                manifest.get("model_revision") == MODEL_REVISION,
                manifest.get("queries_sha256") == query_sha,
                manifest.get("query_count") == len(queries),
                manifest.get("query_instruction") is None,
            )
        )
    if compatible:
        print(f"reusing query vectors: {vectors_path}", flush=True)
        return validate_vectors(vectors_path, len(queries)), model
    if model is None:
        model = load_model()
    texts = [row["query"] for row in queries]
    with torch.inference_mode():
        vectors = normalize_float32(
            model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        )
    QUERY_CACHE.mkdir(parents=True, exist_ok=True)
    temporary = vectors_path.with_name(f".{vectors_path.name}.tmp")
    with temporary.open("wb") as output:
        np.save(output, vectors)
    os.replace(temporary, vectors_path)
    atomic_json(
        manifest_path,
        {
            "status": "complete",
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "queries_sha256": query_sha,
            "query_count": len(queries),
            "embedding_dimension": DIMENSION,
            "normalized": True,
            "query_instruction": None,
            "batch_size": batch_size,
        },
    )
    return vectors, model


def metrics_from_ranks(ranks: list[int | None]) -> dict[str, Any]:
    count = len(ranks)
    output: dict[str, Any] = {"query_count": count}
    for cutoff in (1, 5, 10, 20, 50):
        hits = sum(rank is not None and rank <= cutoff for rank in ranks)
        output[f"hits_at_{cutoff}"] = hits
        output[f"recall_at_{cutoff}"] = hits / count
    output["mrr_at_10"] = sum(
        1.0 / rank for rank in ranks if rank is not None and rank <= 10
    ) / count
    return output


def main() -> int:
    args = parse_args()
    name = chunk_name(args.chunk_size, args.chunk_overlap)
    cache_dir = PPLX_CACHE_ROOT / f"{MODEL_SLUG}_{name}"
    results_dir = RESULTS_ROOT / f"{MODEL_SLUG}_{name}_train"
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = validate_and_publish_metadata(
        args.chunk_size, args.chunk_overlap, cache_dir
    )
    metadata_path = cache_dir / "chunk_metadata.jsonl"
    vectors_path = cache_dir / "embeddings.npy"
    manifest_path = cache_dir / "manifest.json"
    compatible = cache_is_compatible(
        manifest_path, vectors_path, metadata_path, name, len(metadata)
    )
    model: SentenceTransformer | None = None
    encoding: dict[str, Any] = {"reused": compatible and not args.force_reencode}
    if compatible and not args.force_reencode:
        print(f"reusing document vectors: {vectors_path}", flush=True)
        document_vectors = validate_vectors(vectors_path, len(metadata))
    else:
        model = load_model()
        print(
            f"encoding {name}: chunks={len(metadata)} batch_size={args.batch_size}",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats()
        encoding.update(
            encode_documents(
                model,
                metadata,
                args.chunk_size,
                args.chunk_overlap,
                args.batch_size,
                vectors_path,
            )
        )
        encoding.update(
            {
                "peak_gpu_memory_allocated_gib": torch.cuda.max_memory_allocated()
                / (1024**3),
                "peak_gpu_memory_reserved_gib": torch.cuda.max_memory_reserved()
                / (1024**3),
            }
        )
        atomic_json(
            manifest_path,
            {
                "status": "complete",
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "offline": True,
                "corpus_sha256": sha256_file(CORPUS),
                "metadata_sha256": sha256_file(metadata_path),
                "chunk_config": name,
                "chunk_parameters": {
                    "strategy": "fixed",
                    "size": args.chunk_size,
                    "overlap": args.chunk_overlap,
                    "step": args.chunk_size - args.chunk_overlap,
                },
                "input_format": "document title + two newlines + chunk body",
                "query_instruction": None,
                "document_count": EXPECTED_DOCUMENTS,
                "chunk_count": len(metadata),
                "embedding_dimension": DIMENSION,
                "saved_dtype": "float32",
                "normalized": True,
                "batch_size": args.batch_size,
                **encoding,
            },
        )
        document_vectors = validate_vectors(vectors_path, len(metadata))

    queries = list(jsonl_rows(QUERIES))
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} TRAIN queries")
    query_vectors, model = load_or_encode_queries(model, queries, args.batch_size)
    if model is not None:
        del model
        torch.cuda.empty_cache()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for retrieval")
    document_ids, starts, ends, _ = document_layout(metadata)
    document_tensor = torch.tensor(document_vectors, dtype=torch.float32, device="cuda")
    query_tensor = torch.tensor(query_vectors, dtype=torch.float32, device="cuda")
    rankings: list[dict[str, Any]] = []
    ranks: list[int | None] = []
    retrieval_started = time.perf_counter()
    for query_index, query in enumerate(queries):
        chunk_scores = torch.mv(document_tensor, query_tensor[query_index]).cpu().numpy()
        document_scores = np.maximum.reduceat(chunk_scores, starts)
        order = np.lexsort((document_ids, -document_scores))[:TOP_K]
        top_documents: list[dict[str, Any]] = []
        for rank, raw_document_index in enumerate(order, 1):
            document_index = int(raw_document_index)
            start, end = int(starts[document_index]), int(ends[document_index])
            representative = start + int(np.argmax(chunk_scores[start:end]))
            top_documents.append(
                {
                    "rank": rank,
                    "document_id": str(document_ids[document_index]),
                    "score": float(document_scores[document_index]),
                    "representative_chunk_id": metadata[representative]["chunk_id"],
                }
            )
        reference_id = query["reference_document_id"]
        reference_rank = next(
            (row["rank"] for row in top_documents if row["document_id"] == reference_id),
            None,
        )
        ranks.append(reference_rank)
        rankings.append(
            {
                "question_id": query["question_id"],
                "reference_document_id": reference_id,
                "reference_document_rank": reference_rank,
                "top_documents": top_documents,
            }
        )
        if (query_index + 1) % 25 == 0:
            print(f"retrieved queries={query_index + 1}/{len(queries)}", flush=True)

    results_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = results_dir / "dense_rankings_top100.jsonl"
    atomic_jsonl(rankings_path, rankings)
    summary = {
        "status": "complete",
        "evaluation_split": "TRAIN only",
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "query_instruction": None,
        "chunk_config": name,
        "chunk_parameters": {
            "size": args.chunk_size,
            "overlap": args.chunk_overlap,
            "step": args.chunk_size - args.chunk_overlap,
        },
        "document_count": EXPECTED_DOCUMENTS,
        "chunk_count": len(metadata),
        "top_k": TOP_K,
        "tie_break": "score descending, then document_id ascending",
        "document_score": "maximum normalized query/chunk dot product",
        "metrics": metrics_from_ranks(ranks),
        "retrieval_seconds": time.perf_counter() - retrieval_started,
        "encoding": encoding,
        "artifacts": {
            "rankings": str(rankings_path.relative_to(PROJECT_ROOT)),
            "document_vectors": str(vectors_path.relative_to(PROJECT_ROOT)),
            "query_vectors": str((QUERY_CACHE / "embeddings.npy").relative_to(PROJECT_ROOT)),
            "chunk_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
        },
    }
    atomic_json(results_dir / "metrics.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
