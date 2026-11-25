#!/usr/bin/env python3
"""Run the TechQA TRAIN dense baseline for the fixed 600/0 chunk configuration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import fixed_chunks, jsonl_rows


DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
DEFAULT_QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/bge-small-en-v1.5_fixed_600_0"
)
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments/dense/results/fixed_600_0"
DEFAULT_BM25_METRICS = (
    PROJECT_ROOT
    / "experiments/chunk/bm25_char_results/fixed_600_0/metrics.json"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REQUESTED_REVISION = "main"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 0
TOP_K_DOCUMENTS = 10
INPUT_FORMAT = "document title + two newlines + fixed-600/0 chunk body"
CACHE_SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--bm25-metrics", type=Path, default=DEFAULT_BM25_METRICS)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--revision", default=MODEL_REQUESTED_REVISION)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-chunks", type=int, default=512)
    parser.add_argument("--smoke-queries", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    return requested


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def model_revision(model: SentenceTransformer, model_name: str) -> str:
    first_module = model._first_module()
    auto_model = getattr(first_module, "auto_model", None)
    config = getattr(auto_model, "config", None)
    commit_hash = getattr(config, "_commit_hash", None)
    if isinstance(commit_hash, str) and commit_hash:
        return commit_hash
    try:
        from huggingface_hub import model_info

        return model_info(model_name).sha
    except Exception:
        return "unknown"


def load_model(model_name: str, revision: str, device: str) -> SentenceTransformer:
    model = SentenceTransformer(model_name, revision=revision, device=device)
    # Keep this baseline instruction-free even if a future model config defines a default prompt.
    model.default_prompt_name = None
    model.eval()
    return model


def count_truncated(
    model: SentenceTransformer, texts: list[str], batch_size: int
) -> int:
    tokenizer = model.tokenizer
    maximum = model.max_seq_length
    truncated = 0
    for start in range(0, len(texts), batch_size):
        lengths = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=True,
            truncation=False,
            return_length=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["length"]
        truncated += sum(int(length) > maximum for length in lengths)
    return truncated


def build_chunks(
    corpus_path: Path, limit: int | None = None
) -> tuple[list[dict[str, Any]], list[str], int]:
    metadata: list[dict[str, Any]] = []
    inputs: list[str] = []
    document_count = 0
    for document in jsonl_rows(corpus_path):
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
            raise ValueError(f"Invalid corpus record: {document}")
        document_count += 1
        for chunk_index, body in enumerate(fixed_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)):
            chunk_id = f"{document_id}::{chunk_index:06d}"
            metadata.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "body_length": len(body),
                }
            )
            inputs.append(f"{title}\n\n{body}")
            if limit is not None and len(inputs) >= limit:
                return metadata, inputs, document_count
    return metadata, inputs, document_count


def encode(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    device: str,
    output_path: Path | None = None,
) -> tuple[np.ndarray, float]:
    dimension = model.get_embedding_dimension()
    if not isinstance(dimension, int):
        raise RuntimeError("Model did not report a fixed embedding dimension")
    if output_path is None:
        output = np.empty((len(texts), dimension), dtype=np.float32)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=(len(texts), dimension)
        )

    synchronize(device)
    started = time.perf_counter()
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        output[start:end] = model.encode(
            texts[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        if output_path is not None and (end == len(texts) or end % (batch_size * 100) == 0):
            print(f"encoded {end}/{len(texts)} document chunks", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - started
    if isinstance(output, np.memmap):
        output.flush()
    return output, elapsed


def validate_embeddings(embeddings: np.ndarray) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError(f"Invalid embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError(
            f"Embeddings are not normalized: min={norms.min()}, max={norms.max()}"
        )


def smoke_test(
    args: argparse.Namespace, model: SentenceTransformer, device: str, revision: str
) -> None:
    metadata, document_inputs, _ = build_chunks(args.corpus, args.smoke_chunks)
    queries = list(jsonl_rows(args.queries))[: args.smoke_queries]
    query_texts = [row["query"] for row in queries]
    document_truncated = count_truncated(model, document_inputs, args.batch_size)
    query_truncated = count_truncated(model, query_texts, args.query_batch_size)
    document_vectors, document_seconds = encode(
        model, document_inputs, args.batch_size, device
    )
    query_vectors, query_seconds = encode(
        model, query_texts, args.query_batch_size, device
    )
    validate_embeddings(document_vectors)
    validate_embeddings(query_vectors)

    document_ids, starts, _ends, _ = document_layout(metadata)
    if len(document_ids) < 2:
        raise ValueError(
            "Smoke test must cover at least two documents; increase --smoke-chunks"
        )
    samples = []
    for query, vector in zip(queries, query_vectors, strict=True):
        chunk_scores = document_vectors @ vector
        scores = np.maximum.reduceat(chunk_scores, starts)
        order = np.lexsort((document_ids, -scores))
        if any(scores[order[index]] < scores[order[index + 1]] for index in range(len(order) - 1)):
            raise RuntimeError("Smoke-test similarity ordering is not descending")
        top_document_index = int(order[0])
        top_chunk_start = int(starts[top_document_index])
        top_chunk_end = int(starts[top_document_index + 1]) if top_document_index + 1 < len(starts) else len(metadata)
        top_chunk_index = top_chunk_start + int(
            np.argmax(chunk_scores[top_chunk_start:top_chunk_end])
        )
        samples.append(
            {
                "question_id": query["question_id"],
                "top_document_id": str(document_ids[top_document_index]),
                "top_chunk_id": metadata[top_chunk_index]["chunk_id"],
                "score": float(scores[top_document_index]),
            }
        )
    print(
        json.dumps(
            {
                "status": "ok",
                "model": args.model,
                "model_revision": revision,
                "device": device,
                "chunks": len(metadata),
                "queries": len(queries),
                "embedding_dimension": int(document_vectors.shape[1]),
                "document_truncated": document_truncated,
                "query_truncated": query_truncated,
                "document_encoding_seconds": document_seconds,
                "query_encoding_seconds": query_seconds,
                "ranking_samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def cache_is_valid(
    manifest: dict[str, Any],
    corpus_sha256: str,
    model_name: str,
    revision: str,
    embeddings_path: Path,
    metadata_path: Path,
) -> bool:
    expected = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "corpus_sha256": corpus_sha256,
        "model_name": model_name,
        "model_revision": revision,
        "chunk_config": {
            "name": "fixed_600_0",
            "strategy": "fixed",
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "step": CHUNK_SIZE,
        },
        "input_format": INPUT_FORMAT,
        "normalized": True,
        "dtype": "float32",
    }
    return (
        embeddings_path.is_file()
        and metadata_path.is_file()
        and all(manifest.get(key) == value for key, value in expected.items())
    )


def prepare_cache(
    args: argparse.Namespace,
    model: SentenceTransformer,
    device: str,
    revision: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], bool]:
    cache_dir = args.cache_dir
    embeddings_path = cache_dir / "embeddings.npy"
    metadata_path = cache_dir / "chunk_metadata.jsonl"
    manifest_path = cache_dir / "manifest.json"
    corpus_sha256 = sha256_file(args.corpus)
    manifest = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)

    if not args.rebuild_cache and cache_is_valid(
        manifest,
        corpus_sha256,
        args.model,
        revision,
        embeddings_path,
        metadata_path,
    ):
        print(f"reusing document embedding cache: {cache_dir}", flush=True)
        metadata = list(jsonl_rows(metadata_path))
        embeddings = np.load(embeddings_path, mmap_mode="r")
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Cached embeddings and chunk metadata row counts differ")
        validate_embeddings(embeddings)
        return embeddings, metadata, manifest, True

    print("building fixed 600/0 chunks", flush=True)
    metadata, document_inputs, document_count = build_chunks(args.corpus)
    if document_count != 28_482 or len(metadata) != 182_835:
        raise ValueError(
            f"Expected 28,482 documents / 182,835 chunks, found "
            f"{document_count:,} / {len(metadata):,}"
        )
    document_truncated = count_truncated(model, document_inputs, args.batch_size)
    print(
        f"encoding {len(document_inputs):,} chunks on {device}; "
        f"pre-truncation count={document_truncated:,}",
        flush=True,
    )
    embeddings, document_seconds = encode(
        model,
        document_inputs,
        args.batch_size,
        device,
        output_path=embeddings_path,
    )
    validate_embeddings(embeddings)
    write_jsonl(metadata_path, metadata)
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "corpus": str(args.corpus.resolve().relative_to(PROJECT_ROOT)),
        "corpus_sha256": corpus_sha256,
        "model_name": args.model,
        "model_requested_revision": args.revision,
        "model_revision": revision,
        "chunk_config": {
            "name": "fixed_600_0",
            "strategy": "fixed",
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "step": CHUNK_SIZE,
        },
        "input_format": INPUT_FORMAT,
        "document_count": document_count,
        "chunk_count": len(metadata),
        "embedding_dimension": int(embeddings.shape[1]),
        "normalized": True,
        "dtype": "float32",
        "max_sequence_length_tokens": model.max_seq_length,
        "document_truncated_count": document_truncated,
        "document_encoding_seconds": document_seconds,
        "encoding_device": device,
        "batch_size": args.batch_size,
        "artifacts": {
            "embeddings": str(embeddings_path.resolve().relative_to(PROJECT_ROOT)),
            "chunk_metadata": str(metadata_path.resolve().relative_to(PROJECT_ROOT)),
        },
    }
    write_json(manifest_path, manifest)
    return embeddings, metadata, manifest, False


def document_layout(
    metadata: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    document_ids: list[str] = []
    starts: list[int] = []
    seen_document_ids: set[str] = set()
    chunk_document_indices = np.empty(len(metadata), dtype=np.int32)
    previous = None
    document_index = -1
    for chunk_index, row in enumerate(metadata):
        document_id = row["document_id"]
        if document_id != previous:
            if document_id in seen_document_ids:
                raise ValueError("Chunk metadata is not contiguous by document")
            document_ids.append(document_id)
            seen_document_ids.add(document_id)
            starts.append(chunk_index)
            document_index += 1
            previous = document_id
        chunk_document_indices[chunk_index] = document_index
    starts_array = np.asarray(starts, dtype=np.int64)
    ends_array = np.r_[starts_array[1:], len(metadata)].astype(np.int64)
    return (
        np.asarray(document_ids, dtype=object),
        starts_array,
        ends_array,
        chunk_document_indices,
    )


def evaluate(
    args: argparse.Namespace,
    model: SentenceTransformer,
    device: str,
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries = list(jsonl_rows(args.queries))
    if len(queries) != 450:
        raise ValueError(f"Expected 450 TRAIN queries, found {len(queries)}")
    query_texts = [row.get("query") for row in queries]
    if not all(isinstance(text, str) and text for text in query_texts):
        raise ValueError("A TRAIN query has missing or empty query text")
    query_truncated = count_truncated(model, query_texts, args.query_batch_size)
    query_vectors, query_encoding_seconds = encode(
        model, query_texts, args.query_batch_size, device
    )
    validate_embeddings(query_vectors)

    document_ids, starts, ends, _ = document_layout(metadata)
    document_id_to_index = {
        document_id: index for index, document_id in enumerate(document_ids)
    }
    if len(document_ids) != 28_482:
        raise ValueError(f"Expected 28,482 documents, found {len(document_ids):,}")

    document_tensor = torch.as_tensor(
        np.asarray(embeddings), dtype=torch.float32, device=device
    )
    query_tensor = torch.as_tensor(query_vectors, dtype=torch.float32, device=device)
    results = []
    retrieval_durations = []
    invalid_ids: set[str] = set()

    for query_index, query_record in enumerate(queries):
        reference_id = query_record.get("reference_document_id")
        if reference_id not in document_id_to_index:
            raise ValueError(f"Reference document missing from corpus: {reference_id}")
        synchronize(device)
        started = time.perf_counter()
        chunk_scores = torch.mv(document_tensor, query_tensor[query_index]).cpu().numpy()
        document_scores = np.maximum.reduceat(chunk_scores, starts)
        order = np.lexsort((document_ids, -document_scores))
        synchronize(device)
        retrieval_durations.append(time.perf_counter() - started)

        reference_document_index = document_id_to_index[reference_id]
        reference_exact_rank = int(np.flatnonzero(order == reference_document_index)[0]) + 1
        top_documents = []
        for rank, document_index in enumerate(order[:TOP_K_DOCUMENTS], start=1):
            start = int(starts[document_index])
            end = int(ends[document_index])
            representative_offset = int(np.argmax(chunk_scores[start:end]))
            representative_index = start + representative_offset
            document_id = str(document_ids[document_index])
            if document_id not in document_id_to_index:
                invalid_ids.add(document_id)
            top_documents.append(
                {
                    "rank": rank,
                    "document_id": document_id,
                    "score": float(document_scores[document_index]),
                    "representative_chunk_id": metadata[representative_index]["chunk_id"],
                }
            )
        results.append(
            {
                "question_id": query_record["question_id"],
                "reference_document_id": reference_id,
                "reference_document_rank": (
                    reference_exact_rank if reference_exact_rank <= TOP_K_DOCUMENTS else None
                ),
                "reference_document_exact_rank": reference_exact_rank,
                "reference_rank_scope": "top_10; null means not in top 10",
                "top_documents": top_documents,
            }
        )
        if (query_index + 1) % 50 == 0:
            print(f"retrieved {query_index + 1}/450 queries", flush=True)

    denominator = len(results)
    recall_at_1 = sum(row["reference_document_exact_rank"] == 1 for row in results) / denominator
    recall_at_5 = sum(row["reference_document_exact_rank"] <= 5 for row in results) / denominator
    mrr_at_10 = sum(
        1.0 / row["reference_document_exact_rank"]
        if row["reference_document_exact_rank"] <= 10
        else 0.0
        for row in results
    ) / denominator
    search_seconds = sum(retrieval_durations)
    metrics = {
        "query_count": denominator,
        "queries_with_10_documents": sum(
            len(row["top_documents"]) == TOP_K_DOCUMENTS for row in results
        ),
        "invalid_returned_document_ids": len(invalid_ids),
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "mrr_at_10": mrr_at_10,
        "query_truncated_count": query_truncated,
        "query_encoding_seconds": query_encoding_seconds,
        "exact_search_seconds": search_seconds,
        "average_search_seconds": search_seconds / denominator,
        "average_query_seconds": (query_encoding_seconds + search_seconds) / denominator,
        "exact_reference_ranks": {
            row["question_id"]: row["reference_document_exact_rank"]
            for row in results
            if row["question_id"] in {"TRAIN_Q003", "TRAIN_Q024"}
        },
    }
    return results, metrics


def comparison_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    retrieval = metrics["retrieval"]
    return {
        "retriever": name,
        "chunk_config": "fixed_600_0",
        "query_count": retrieval["query_count"],
        "recall_at_1": retrieval["recall_at_1"],
        "recall_at_5": retrieval["recall_at_5"],
        "mrr_at_10": retrieval["mrr_at_10"],
        "average_query_seconds": retrieval["average_query_seconds"],
    }


def main() -> int:
    args = parse_args()
    args.corpus = args.corpus.resolve()
    args.queries = args.queries.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.results_dir = args.results_dir.resolve()
    args.bm25_metrics = args.bm25_metrics.resolve()
    if args.batch_size <= 0 or args.query_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")

    device = choose_device(args.device)
    print(f"loading {args.model}@{args.revision} on {device}", flush=True)
    model = load_model(args.model, args.revision, device)
    revision = model_revision(model, args.model)
    print(
        f"resolved model revision={revision}; max_seq_length={model.max_seq_length}",
        flush=True,
    )
    if args.smoke_test:
        smoke_test(args, model, device, revision)
        return 0

    embeddings, metadata, cache_manifest, cache_reused = prepare_cache(
        args, model, device, revision
    )
    rankings, retrieval_metrics = evaluate(
        args, model, device, embeddings, metadata
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = args.results_dir / "query_rankings.jsonl"
    write_jsonl(rankings_path, rankings)

    dense_metrics = {
        "evaluation_split": "TRAIN only",
        "config": {
            "name": "fixed_600_0",
            "strategy": "fixed",
            "size": CHUNK_SIZE,
            "overlap": CHUNK_OVERLAP,
            "step": CHUNK_SIZE,
        },
        "dense": {
            "model_name": args.model,
            "model_requested_revision": args.revision,
            "model_revision": revision,
            "query_instruction": None,
            "input_format": INPUT_FORMAT,
            "normalized_embeddings": True,
            "similarity": "exact dot product",
            "document_aggregation": "maximum chunk score per document before top-10",
            "document_tie_break": "document_id ascending",
            "reranking": False,
        },
        "encoding": {
            "device": device,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if device == "cuda" else None
            ),
            "max_sequence_length_tokens": model.max_seq_length,
            "document_count": cache_manifest["document_count"],
            "chunk_count": cache_manifest["chunk_count"],
            "embedding_dimension": cache_manifest["embedding_dimension"],
            "document_truncated_count": cache_manifest["document_truncated_count"],
            "query_truncated_count": retrieval_metrics["query_truncated_count"],
            "document_encoding_seconds": cache_manifest["document_encoding_seconds"],
            "document_batch_size": cache_manifest["batch_size"],
            "query_batch_size": args.query_batch_size,
            "cache_reused_for_this_run": cache_reused,
        },
        "retrieval": retrieval_metrics,
        "artifacts": {
            "rankings": str(rankings_path.relative_to(PROJECT_ROOT)),
            "cache_manifest": str((args.cache_dir / "manifest.json").relative_to(PROJECT_ROOT)),
            "embeddings": str((args.cache_dir / "embeddings.npy").relative_to(PROJECT_ROOT)),
            "chunk_metadata": str((args.cache_dir / "chunk_metadata.jsonl").relative_to(PROJECT_ROOT)),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sentence_transformers": importlib.metadata.version("sentence-transformers"),
            "transformers": importlib.metadata.version("transformers"),
        },
    }
    metrics_path = args.results_dir / "metrics.json"
    write_json(metrics_path, dense_metrics)

    with args.bm25_metrics.open(encoding="utf-8") as stream:
        bm25_metrics = json.load(stream)
    comparison = {
        "evaluation_split": "TRAIN only",
        "query_denominator": 450,
        "chunk_config": "fixed_600_0",
        "results": [
            comparison_row("BM25", bm25_metrics),
            comparison_row("Dense BAAI/bge-small-en-v1.5", dense_metrics),
        ],
    }
    write_json(args.results_dir / "comparison_with_bm25.json", comparison)
    print(
        f"Dense R@1={retrieval_metrics['recall_at_1']:.6f} "
        f"R@5={retrieval_metrics['recall_at_5']:.6f} "
        f"MRR@10={retrieval_metrics['mrr_at_10']:.6f}",
        flush=True,
    )
    print(
        f"exact ranks: {retrieval_metrics['exact_reference_ranks']}", flush=True
    )
    print(f"wrote {metrics_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
