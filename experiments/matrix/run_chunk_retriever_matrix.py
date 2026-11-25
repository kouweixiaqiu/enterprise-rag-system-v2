#!/usr/bin/env python3
"""Run the resumable TechQA TRAIN 3 chunk x 3 retriever matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import (  # noqa: E402
    CONFIGS,
    chunks_for,
    create_schema,
    jsonl_rows,
    retrieve_top_documents,
    tokenize,
)
from experiments.dense.run_dense_fixed_600_0 import (  # noqa: E402
    count_truncated,
    document_layout,
    validate_embeddings,
)


CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
INDEX_DIR = PROJECT_ROOT / "data/processed/techqa/bm25_chunk_indexes"
CACHE_ROOT = PROJECT_ROOT / "data/processed/techqa/dense_cache"
RESULTS_ROOT = PROJECT_ROOT / "experiments/matrix/results/train"
DEFAULT_LOG = PROJECT_ROOT / "experiments/matrix/logs/chunk_retriever_matrix.log"
FIXED_RANKINGS = PROJECT_ROOT / "experiments/hybrid/results/fixed_600_0"

CHUNK_NAMES = ("fixed_600_0", "paragraph_merge_600_0", "fixed_1200_300")
RETRIEVERS = ("bm25", "dense", "rrf")
MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
TOP_K = 50
RRF_K = 60
EXPECTED_QUERIES = 450
EXPECTED_DOCUMENTS = 28_482
CACHE_SCHEMA_VERSION = 2
RUN_SCHEMA_VERSION = 1
LOGGER = logging.getLogger("chunk_retriever_matrix")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--smoke-queries", type=int, default=8)
    parser.add_argument("--smoke-documents", type=int, default=64)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
    os.replace(temporary, path)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return requested


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def load_model(device: str) -> SentenceTransformer:
    LOGGER.info("loading %s@%s on %s", MODEL_NAME, MODEL_REVISION, device)
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=device,
        local_files_only=True,
    )
    model.default_prompt_name = None
    model.eval()
    resolved = getattr(model._first_module().auto_model.config, "_commit_hash", None)
    if resolved is not None and resolved != MODEL_REVISION:
        raise RuntimeError(f"model revision mismatch: expected {MODEL_REVISION}, got {resolved}")
    LOGGER.info("model loaded; dimension=%s max_tokens=%s", model.get_embedding_dimension(), model.max_seq_length)
    return model


def encode(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    device: str,
    output_path: Path | None = None,
    label: str = "items",
) -> tuple[np.ndarray, float]:
    dimension = model.get_embedding_dimension()
    if not isinstance(dimension, int):
        raise RuntimeError("model has no fixed embedding dimension")
    if output_path is None:
        output: np.ndarray = np.empty((len(texts), dimension), dtype=np.float32)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=(len(texts), dimension)
        )
    synchronize(device)
    started = time.perf_counter()
    report_every = max(batch_size, batch_size * 50)
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        output[start:end] = model.encode(
            texts[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        if end == len(texts) or end % report_every == 0:
            LOGGER.info("encoded %s/%s %s", end, len(texts), label)
    synchronize(device)
    elapsed = time.perf_counter() - started
    if isinstance(output, np.memmap):
        output.flush()
    return output, elapsed


def chunk_config_dict(name: str) -> dict[str, Any]:
    config = CONFIGS[name]
    return {
        "name": config.name,
        "strategy": config.strategy,
        "size": config.size,
        "overlap": config.overlap,
        "step": config.size - config.overlap if config.size else None,
    }


def build_chunks(
    corpus_path: Path, config_name: str
) -> tuple[list[dict[str, Any]], list[str], int]:
    metadata: list[dict[str, Any]] = []
    inputs: list[str] = []
    document_count = 0
    for document in jsonl_rows(corpus_path):
        document_id = document.get("document_id")
        title = document.get("title")
        text = document.get("text")
        if not all(isinstance(value, str) for value in (document_id, title, text)) or not document_id or not text:
            raise ValueError(f"invalid corpus row: {document}")
        document_count += 1
        for chunk_index, body in enumerate(chunks_for(text, CONFIGS[config_name])):
            metadata.append(
                {
                    "chunk_id": f"{document_id}::{chunk_index:06d}",
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "body_length": len(body),
                }
            )
            inputs.append(f"{title}\n\n{body}")
    return metadata, inputs, document_count


def cache_dir(cache_root: Path, config_name: str) -> Path:
    return cache_root / f"bge-small-en-v1.5_{config_name}"


def load_or_build_cache(
    args: argparse.Namespace,
    model: SentenceTransformer,
    device: str,
    config_name: str,
    corpus_hash: str,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any], bool]:
    directory = cache_dir(args.cache_root, config_name)
    vectors_path = directory / "embeddings.npy"
    metadata_path = directory / "chunk_metadata.jsonl"
    manifest_path = directory / "manifest.json"
    expected = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "corpus_sha256": corpus_hash,
        "chunk_config": chunk_config_dict(config_name),
        "normalized": True,
        "dtype": "float32",
    }
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    valid = (
        not args.force
        and vectors_path.is_file()
        and metadata_path.is_file()
        and all(manifest.get(key) == value for key, value in expected.items())
    )
    if valid:
        metadata = list(jsonl_rows(metadata_path))
        vectors = np.load(vectors_path, mmap_mode="r")
        if vectors.shape != (len(metadata), 384):
            raise ValueError(f"invalid cached vector shape for {config_name}: {vectors.shape}")
        LOGGER.info("reusing dense cache for %s: %s chunks", config_name, len(metadata))
        return vectors, metadata, manifest, True

    LOGGER.info("building %s chunks for dense cache", config_name)
    metadata, document_inputs, document_count = build_chunks(args.corpus, config_name)
    if document_count != EXPECTED_DOCUMENTS:
        raise ValueError(f"expected {EXPECTED_DOCUMENTS} documents, got {document_count}")
    truncated = count_truncated(model, document_inputs, args.batch_size)
    LOGGER.info(
        "encoding dense cache for %s: chunks=%s truncated=%s",
        config_name,
        len(metadata),
        truncated,
    )
    directory.mkdir(parents=True, exist_ok=True)
    temporary_vectors = directory / ".embeddings.npy.tmp"
    vectors, elapsed = encode(
        model,
        document_inputs,
        args.batch_size,
        device,
        temporary_vectors,
        f"{config_name} document chunks",
    )
    validate_embeddings(vectors)
    if isinstance(vectors, np.memmap):
        vectors.flush()
    del vectors
    os.replace(temporary_vectors, vectors_path)
    atomic_jsonl(metadata_path, metadata)
    manifest = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        **expected,
        "corpus": str(args.corpus.relative_to(PROJECT_ROOT)),
        "document_count": document_count,
        "chunk_count": len(metadata),
        "embedding_dimension": 384,
        "input_format": "document title + two newlines + chunk body",
        "query_instruction": None,
        "max_sequence_length_tokens": model.max_seq_length,
        "document_truncated_count": truncated,
        "document_encoding_seconds": elapsed,
        "encoding_device": device,
        "batch_size": args.batch_size,
        "artifacts": {
            "embeddings": str(vectors_path.relative_to(PROJECT_ROOT)),
            "chunk_metadata": str(metadata_path.relative_to(PROJECT_ROOT)),
        },
    }
    atomic_json(manifest_path, manifest)
    LOGGER.info("dense cache complete for %s in %.1fs", config_name, elapsed)
    return np.load(vectors_path, mmap_mode="r"), metadata, manifest, False


def metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [row["reference_document_rank"] for row in rows]
    denominator = len(ranks)
    metrics: dict[str, Any] = {"query_count": denominator}
    for cutoff in (1, 5, 10, 20, 50):
        metrics[f"recall_at_{cutoff}"] = sum(
            rank is not None and rank <= cutoff for rank in ranks
        ) / denominator
    metrics["mrr_at_10"] = sum(
        1.0 / rank if rank is not None and rank <= 10 else 0.0 for rank in ranks
    ) / denominator
    return metrics


def normalize_saved_rows(source: Path) -> list[dict[str, Any]]:
    rows = list(jsonl_rows(source))
    if len(rows) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} rows in {source}, got {len(rows)}")
    normalized = []
    for row in rows:
        documents = row["top_documents"][:TOP_K]
        if len(documents) != TOP_K or len({item["document_id"] for item in documents}) != TOP_K:
            raise ValueError(f"invalid Top-{TOP_K} in {source} for {row['question_id']}")
        reference_id = row["reference_document_id"]
        rank = next((item["rank"] for item in documents if item["document_id"] == reference_id), None)
        normalized.append(
            {
                "question_id": row["question_id"],
                "reference_document_id": reference_id,
                "reference_document_rank": rank,
                "reference_rank_scope": "top_50; null means not in top 50",
                "top_documents": documents,
            }
        )
    return normalized


def run_bm25(index_path: Path, queries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    connection = sqlite3.connect(index_path)
    connection.execute("PRAGMA query_only=ON")
    rows = []
    started = time.perf_counter()
    try:
        for index, query in enumerate(queries, 1):
            documents = retrieve_top_documents(connection, query["query"], TOP_K)
            if len(documents) != TOP_K:
                raise ValueError(f"BM25 returned {len(documents)} documents for {query['question_id']}")
            reference_id = query["reference_document_id"]
            rank = next((item["rank"] for item in documents if item["document_id"] == reference_id), None)
            rows.append(
                {
                    "question_id": query["question_id"],
                    "reference_document_id": reference_id,
                    "reference_document_rank": rank,
                    "reference_rank_scope": "top_50; null means not in top 50",
                    "top_documents": documents,
                }
            )
            if index % 50 == 0:
                LOGGER.info("BM25 retrieved %s/%s queries", index, len(queries))
    finally:
        connection.close()
    return rows, time.perf_counter() - started


def run_dense(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    query_vectors: np.ndarray,
    queries: list[dict[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], float]:
    document_ids, starts, ends, _ = document_layout(metadata)
    if len(document_ids) != EXPECTED_DOCUMENTS:
        raise ValueError(f"expected {EXPECTED_DOCUMENTS} documents, got {len(document_ids)}")
    tensor = torch.as_tensor(np.asarray(embeddings), dtype=torch.float32, device=device)
    rows = []
    started = time.perf_counter()
    for query_index, query in enumerate(queries):
        scores = torch.mv(tensor, torch.as_tensor(query_vectors[query_index], device=device)).cpu().numpy()
        document_scores = np.maximum.reduceat(scores, starts)
        order = np.lexsort((document_ids, -document_scores))[:TOP_K]
        documents = []
        for rank, document_index_value in enumerate(order, 1):
            document_index = int(document_index_value)
            start = int(starts[document_index])
            end = int(ends[document_index])
            representative = start + int(np.argmax(scores[start:end]))
            documents.append(
                {
                    "rank": rank,
                    "document_id": str(document_ids[document_index]),
                    "score": float(document_scores[document_index]),
                    "representative_chunk_id": metadata[representative]["chunk_id"],
                }
            )
        reference_id = query["reference_document_id"]
        reference_rank = next((item["rank"] for item in documents if item["document_id"] == reference_id), None)
        rows.append(
            {
                "question_id": query["question_id"],
                "reference_document_id": reference_id,
                "reference_document_rank": reference_rank,
                "reference_rank_scope": "top_50; null means not in top 50",
                "top_documents": documents,
            }
        )
        if (query_index + 1) % 50 == 0:
            LOGGER.info("Dense retrieved %s/%s queries", query_index + 1, len(queries))
    synchronize(device)
    return rows, time.perf_counter() - started


def run_rrf(
    bm25_rows: list[dict[str, Any]], dense_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float]:
    started = time.perf_counter()
    output = []
    for bm25, dense in zip(bm25_rows, dense_rows, strict=True):
        if bm25["question_id"] != dense["question_id"]:
            raise ValueError("BM25 and Dense rows are not aligned")
        candidates: dict[str, dict[str, Any]] = {}
        for route, documents in (("bm25", bm25["top_documents"]), ("dense", dense["top_documents"])):
            for item in documents[:TOP_K]:
                candidate = candidates.setdefault(
                    item["document_id"],
                    {"document_id": item["document_id"], "score": 0.0, "bm25_rank": None, "dense_rank": None},
                )
                candidate[f"{route}_rank"] = item["rank"]
                candidate["score"] += 1.0 / (RRF_K + item["rank"])
        documents = sorted(candidates.values(), key=lambda item: (-item["score"], item["document_id"]))[:TOP_K]
        for rank, item in enumerate(documents, 1):
            item["rank"] = rank
        reference_id = bm25["reference_document_id"]
        reference_rank = next((item["rank"] for item in documents if item["document_id"] == reference_id), None)
        output.append(
            {
                "question_id": bm25["question_id"],
                "reference_document_id": reference_id,
                "reference_document_rank": reference_rank,
                "reference_rank_scope": "top_50 fused ranking; null means not in top 50",
                "top_documents": documents,
            }
        )
    return output, time.perf_counter() - started


def result_signature(corpus_hash: str, query_hash: str, chunk_name: str, retriever: str) -> str:
    payload = {
        "schema": RUN_SCHEMA_VERSION,
        "corpus_sha256": corpus_hash,
        "queries_sha256": query_hash,
        "chunk": chunk_config_dict(chunk_name),
        "retriever": retriever,
        "top_k": TOP_K,
        "model": f"{MODEL_NAME}@{MODEL_REVISION}" if retriever in {"dense", "rrf", "preflight"} else None,
        "query_instruction": None,
        "rrf": {"k": RRF_K, "depth": TOP_K, "weights": [1, 1]} if retriever == "rrf" else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def completed_group(
    directory: Path, signature: str, expected_question_ids: list[str], force: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    if force:
        return None
    rankings_path = directory / "rankings_top50.jsonl"
    metrics_path = directory / "metrics.json"
    if not rankings_path.is_file() or not metrics_path.is_file():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows = list(jsonl_rows(rankings_path))
        valid = (
            metrics.get("status") == "complete"
            and metrics.get("run_signature") == signature
            and [row.get("question_id") for row in rows] == expected_question_ids
            and all(len(row.get("top_documents", [])) == TOP_K for row in rows)
        )
        return (rows, metrics) if valid else None
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_group(
    directory: Path,
    rows: list[dict[str, Any]],
    chunk_name: str,
    retriever: str,
    signature: str,
    elapsed: float,
    details: dict[str, Any],
) -> dict[str, Any]:
    rankings_path = directory / "rankings_top50.jsonl"
    metrics_path = directory / "metrics.json"
    atomic_jsonl(rankings_path, rows)
    metrics = {
        "status": "complete",
        "run_signature": signature,
        "evaluation_split": "TRAIN",
        "chunk_config": chunk_config_dict(chunk_name),
        "retriever": retriever,
        "top_k_saved": TOP_K,
        **metrics_from_rows(rows),
        "elapsed_seconds": elapsed,
        "details": details,
        "rankings": str(rankings_path.relative_to(PROJECT_ROOT)),
        "rankings_sha256": sha256_file(rankings_path),
    }
    atomic_json(metrics_path, metrics)
    LOGGER.info(
        "saved %s x %s: R@1=%.4f R@5=%.4f R@50=%.4f MRR@10=%.4f",
        chunk_name,
        retriever,
        metrics["recall_at_1"],
        metrics["recall_at_5"],
        metrics["recall_at_50"],
        metrics["mrr_at_10"],
    )
    return metrics


def smoke_test_config(
    args: argparse.Namespace,
    model: SentenceTransformer,
    device: str,
    config_name: str,
    queries: list[dict[str, Any]],
    corpus_hash: str,
    query_hash: str,
) -> None:
    output_path = args.results_dir / "preflight" / f"{config_name}.json"
    signature = result_signature(corpus_hash, query_hash, config_name, "preflight")
    if output_path.is_file() and not args.force:
        current = json.loads(output_path.read_text(encoding="utf-8"))
        if current.get("status") == "ok" and current.get("run_signature") == signature:
            LOGGER.info("reusing successful preflight for %s", config_name)
            return
    selected_queries = queries[: args.smoke_queries]
    reference_ids = {row["reference_document_id"] for row in selected_queries}
    selected_documents = []
    for document in jsonl_rows(args.corpus):
        if len(selected_documents) < args.smoke_documents or document["document_id"] in reference_ids:
            selected_documents.append(document)
    present = {row["document_id"] for row in selected_documents}
    if not reference_ids <= present:
        missing = reference_ids - present
        for document in jsonl_rows(args.corpus):
            if document["document_id"] in missing:
                selected_documents.append(document)
                missing.remove(document["document_id"])
            if not missing:
                break
    metadata = []
    inputs = []
    chunk_rows = []
    rowid = 0
    for document in selected_documents:
        for chunk_index, body in enumerate(chunks_for(document["text"], CONFIGS[config_name])):
            rowid += 1
            chunk_id = f"{document['document_id']}::{chunk_index:06d}"
            metadata.append({"chunk_id": chunk_id, "document_id": document["document_id"], "chunk_index": chunk_index, "body_length": len(body)})
            inputs.append(f"{document['title']}\n\n{body}")
            chunk_rows.append((rowid, chunk_id, document["document_id"], chunk_index, len(body), " ".join(tokenize(inputs[-1]))))
    doc_vectors, _ = encode(model, inputs, args.batch_size, device, label=f"{config_name} preflight chunks")
    query_vectors, _ = encode(model, [row["query"] for row in selected_queries], args.query_batch_size, device, label=f"{config_name} preflight queries")
    validate_embeddings(doc_vectors)
    validate_embeddings(query_vectors)
    document_ids, starts, ends, _ = document_layout(metadata)
    dense_rows = []
    for query_index, query in enumerate(selected_queries):
        scores = doc_vectors @ query_vectors[query_index]
        document_scores = np.maximum.reduceat(scores, starts)
        order = np.lexsort((document_ids, -document_scores))[: min(TOP_K, len(document_ids))]
        documents = []
        for rank, document_index_value in enumerate(order, 1):
            document_index = int(document_index_value)
            start, end = int(starts[document_index]), int(ends[document_index])
            representative = start + int(np.argmax(scores[start:end]))
            documents.append({"rank": rank, "document_id": str(document_ids[document_index]), "score": float(document_scores[document_index]), "representative_chunk_id": metadata[representative]["chunk_id"]})
        reference = query["reference_document_id"]
        dense_rows.append({"question_id": query["question_id"], "reference_document_id": reference, "reference_document_rank": next((item["rank"] for item in documents if item["document_id"] == reference), None), "top_documents": documents})
    connection = sqlite3.connect(":memory:")
    create_schema(connection)
    connection.executemany("INSERT INTO chunk_meta(rowid, chunk_id, document_id, chunk_index, body_length) VALUES (?, ?, ?, ?, ?)", [row[:5] for row in chunk_rows])
    connection.executemany("INSERT INTO chunks_fts(rowid, tokens) VALUES (?, ?)", [(row[0], row[5]) for row in chunk_rows])
    bm25_rows = []
    for query in selected_queries:
        documents = retrieve_top_documents(connection, query["query"], min(TOP_K, len(document_ids)))
        reference = query["reference_document_id"]
        bm25_rows.append({"question_id": query["question_id"], "reference_document_id": reference, "reference_document_rank": next((item["rank"] for item in documents if item["document_id"] == reference), None), "top_documents": documents})
    connection.close()
    rrf_rows, _ = run_rrf(bm25_rows, dense_rows)
    report = {
        "status": "ok",
        "run_signature": signature,
        "chunk_config": chunk_config_dict(config_name),
        "queries": len(selected_queries),
        "documents": len(document_ids),
        "chunks": len(metadata),
        "metrics": {
            "bm25": metrics_from_rows(bm25_rows),
            "dense": metrics_from_rows(dense_rows),
            "rrf": metrics_from_rows(rrf_rows),
        },
        "note": "Preflight metrics validate plumbing on a small reference-containing corpus and are not comparable to full-corpus TRAIN metrics.",
    }
    atomic_json(output_path, report)
    LOGGER.info("preflight passed for %s: documents=%s chunks=%s", config_name, len(document_ids), len(metadata))


def write_summary(results_dir: Path, metrics_rows: list[dict[str, Any]]) -> None:
    ordered = sorted(metrics_rows, key=lambda row: (CHUNK_NAMES.index(row["chunk_config"]["name"]), RETRIEVERS.index(row["retriever"])))
    fields = ["chunk_config", "retriever", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20", "recall_at_50", "mrr_at_10"]
    csv_path = results_dir / "summary.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in ordered:
            writer.writerow({**{field: row[field] for field in fields[2:]}, "chunk_config": row["chunk_config"]["name"], "retriever": row["retriever"]})
    os.replace(temporary, csv_path)
    markdown = [
        "# TechQA TRAIN Chunk × Retriever matrix",
        "",
        "| Chunk | Retriever | R@1 | R@5 | R@10 | R@20 | R@50 | MRR@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        markdown.append(
            f"| {row['chunk_config']['name']} | {row['retriever']} | {row['recall_at_1']:.4f} | {row['recall_at_5']:.4f} | {row['recall_at_10']:.4f} | {row['recall_at_20']:.4f} | {row['recall_at_50']:.4f} | {row['mrr_at_10']:.4f} |"
        )
    markdown_path = results_dir / "summary.md"
    temporary_markdown = markdown_path.with_name(f".{markdown_path.name}.tmp")
    temporary_markdown.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    os.replace(temporary_markdown, markdown_path)
    atomic_json(results_dir / "summary.json", {"status": "complete", "evaluation_split": "TRAIN", "query_count": EXPECTED_QUERIES, "chunk_configs": list(CHUNK_NAMES), "retrievers": list(RETRIEVERS), "results": ordered})
    LOGGER.info("nine-group summary written to %s", markdown_path)


def main() -> int:
    args = parse_args()
    for name in ("corpus", "queries", "index_dir", "cache_root", "results_dir", "log_file"):
        setattr(args, name, getattr(args, name).resolve())
    configure_logging(args.log_file)
    if min(args.batch_size, args.query_batch_size, args.smoke_queries, args.smoke_documents) <= 0:
        raise ValueError("batch and smoke sizes must be positive")
    LOGGER.info("starting resumable TRAIN 3x3 matrix")
    corpus_hash, query_hash = sha256_file(args.corpus), sha256_file(args.queries)
    queries = list(jsonl_rows(args.queries))
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} TRAIN queries, got {len(queries)}")
    question_ids = [row["question_id"] for row in queries]
    device = choose_device(args.device)
    model = load_model(device)
    for config_name in CHUNK_NAMES[1:]:
        smoke_test_config(args, model, device, config_name, queries, corpus_hash, query_hash)
    if args.smoke_only:
        LOGGER.info("smoke-only run complete")
        return 0

    query_texts = [row["query"] for row in queries]
    query_truncated = count_truncated(model, query_texts, args.query_batch_size)
    query_vectors, query_encoding_seconds = encode(model, query_texts, args.query_batch_size, device, label="TRAIN queries")
    validate_embeddings(query_vectors)
    LOGGER.info("encoded all TRAIN queries in %.2fs; truncated=%s", query_encoding_seconds, query_truncated)

    all_metrics: list[dict[str, Any]] = []
    group_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for config_name in CHUNK_NAMES:
        for retriever in RETRIEVERS:
            directory = args.results_dir / config_name / retriever
            signature = result_signature(corpus_hash, query_hash, config_name, retriever)
            complete = completed_group(directory, signature, question_ids, args.force)
            if complete is not None:
                rows, metrics = complete
                group_rows[(config_name, retriever)] = rows
                all_metrics.append(metrics)
                LOGGER.info("reusing completed group %s x %s", config_name, retriever)
                continue
            LOGGER.info("running full group %s x %s", config_name, retriever)
            if retriever == "bm25":
                if config_name == "fixed_600_0":
                    source = FIXED_RANKINGS / "bm25_top100.jsonl"
                    started = time.perf_counter()
                    rows = normalize_saved_rows(source)
                    elapsed = time.perf_counter() - started
                    details = {"reused_rankings": str(source.relative_to(PROJECT_ROOT)), "source_sha256": sha256_file(source)}
                else:
                    rows, elapsed = run_bm25(args.index_dir / f"{config_name}.sqlite", queries)
                    details = {"reused_sqlite_index": str((args.index_dir / f"{config_name}.sqlite").relative_to(PROJECT_ROOT)), "bm25_k1": 1.2, "bm25_b": 0.75}
            elif retriever == "dense":
                if config_name == "fixed_600_0":
                    source = FIXED_RANKINGS / "dense_top100.jsonl"
                    started = time.perf_counter()
                    rows = normalize_saved_rows(source)
                    elapsed = time.perf_counter() - started
                    details = {"reused_rankings": str(source.relative_to(PROJECT_ROOT)), "source_sha256": sha256_file(source), "query_instruction": None}
                else:
                    embeddings, metadata, manifest, cache_reused = load_or_build_cache(args, model, device, config_name, corpus_hash)
                    rows, elapsed = run_dense(embeddings, metadata, query_vectors, queries, device)
                    details = {"dense_cache": str(cache_dir(args.cache_root, config_name).relative_to(PROJECT_ROOT)), "cache_reused": cache_reused, "model": MODEL_NAME, "model_revision": MODEL_REVISION, "query_instruction": None, "query_truncated_count": query_truncated, "query_encoding_seconds_shared": query_encoding_seconds, "document_truncated_count": manifest["document_truncated_count"], "document_encoding_seconds": manifest["document_encoding_seconds"]}
                    del embeddings, metadata
                    if device == "cuda":
                        torch.cuda.empty_cache()
            else:
                bm25_rows = group_rows[(config_name, "bm25")]
                dense_rows = group_rows[(config_name, "dense")]
                rows, elapsed = run_rrf(bm25_rows, dense_rows)
                details = {"rrf_k": RRF_K, "input_depth_per_route": TOP_K, "bm25_weight": 1, "dense_weight": 1, "rank_origin": 1, "sort": "score descending, document_id ascending"}
            metrics = save_group(directory, rows, config_name, retriever, signature, elapsed, details)
            group_rows[(config_name, retriever)] = rows
            all_metrics.append(metrics)
    write_summary(args.results_dir, all_metrics)
    LOGGER.info("TRAIN 3x3 matrix complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
