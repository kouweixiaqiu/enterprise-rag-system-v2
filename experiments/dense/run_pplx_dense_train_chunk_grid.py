#!/usr/bin/env python3
"""Run the six TechQA TRAIN chunk configurations with PPLX Dense, serially."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import (  # noqa: E402
    ChunkConfig,
    chunks_for,
    jsonl_rows,
)
from experiments.chunk.run_bm25_parameter_grid import CONFIGS  # noqa: E402
from experiments.dense.run_dense_fixed_600_0 import document_layout  # noqa: E402
from experiments.dense.run_pplx_dense_train_chunk_config import (  # noqa: E402
    DIMENSION,
    EXPECTED_DOCUMENTS,
    EXPECTED_QUERIES,
    MODEL_NAME,
    MODEL_REVISION,
    MODEL_SLUG,
    QUERY_CACHE,
    atomic_json,
    atomic_jsonl,
    load_model,
    load_or_encode_queries,
    normalize_float32,
    sha256_file,
    validate_vectors,
)


CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data/processed/techqa/bm25_parameter_grid_indexes"
DEFAULT_CACHE_ROOT = PROJECT_ROOT / "data/processed/techqa/dense_cache"
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "experiments/dense/results/pplx_embed_v1_0.6b_six_chunk_train"
)
DEFAULT_CONFIGS = (
    "fixed_600_0",
    "fixed_600_150",
    "fixed_1200_0",
    "fixed_1200_300",
    "paragraph_merge_600_0",
    "paragraph_merge_1200_0",
)
DEFAULT_BATCH_SIZE = 16
TOP_K = 100
CACHE_SCHEMA_VERSION = 1

# These are known project artifacts, not a directory scan or format guess.
KNOWN_COMPLETE_RANKINGS = {
    "fixed_600_0": PROJECT_ROOT
    / "experiments/dense/results/pplx_embed_v1_0.6b_fixed_600_0_train"
    / "dense_rankings_top100.jsonl",
    "fixed_1200_300": PROJECT_ROOT
    / "experiments/dense/results/pplx-embed-v1-0.6b_fixed_1200_300_train"
    / "dense_rankings_top100.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--bm25-index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--configs", nargs="+", choices=DEFAULT_CONFIGS, default=list(DEFAULT_CONFIGS)
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force-reencode", action="store_true")
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def metadata_file_matches(path: Path, expected: Sequence[dict[str, Any]]) -> bool:
    if not path.is_file():
        return False
    iterator = jsonl_rows(path)
    for wanted in expected:
        if next(iterator, None) != wanted:
            return False
    return next(iterator, None) is None


def load_bm25_metadata(
    index_path: Path, corpus: Path, config: ChunkConfig
) -> list[dict[str, Any]]:
    if not index_path.is_file():
        raise FileNotFoundError(f"missing BM25 chunk index: {index_path}")
    connection = sqlite3.connect(f"file:{index_path}?mode=ro", uri=True)
    try:
        metadata_row = connection.execute(
            "SELECT value FROM run_metadata WHERE key='metadata'"
        ).fetchone()
        if metadata_row is None:
            raise ValueError("BM25 index has no completion metadata")
        index_metadata = json.loads(metadata_row[0])
        if index_metadata.get("signature", {}).get("config") != asdict(config):
            raise ValueError(
                "BM25 index chunk configuration differs: "
                f"{index_metadata.get('signature', {}).get('config')}"
            )
        saved_corpus = index_metadata.get("signature", {}).get("corpus", {})
        corpus_stat = corpus.stat()
        if (
            Path(saved_corpus.get("path", "")).resolve() != corpus
            or saved_corpus.get("size") != corpus_stat.st_size
            or saved_corpus.get("mtime_ns") != corpus_stat.st_mtime_ns
        ):
            raise ValueError(
                f"BM25 index was not built from the current corpus: {saved_corpus}"
            )
        rows = [
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "chunk_index": int(chunk_index),
                "body_length": int(body_length),
            }
            for chunk_id, document_id, chunk_index, body_length in connection.execute(
                "SELECT chunk_id, document_id, chunk_index, body_length "
                "FROM chunk_meta ORDER BY rowid"
            )
        ]
    finally:
        connection.close()
    if not rows:
        raise ValueError("BM25 index contains no chunks")
    document_ids, _, _, _ = document_layout(rows)
    if len(document_ids) != EXPECTED_DOCUMENTS:
        raise ValueError(
            f"BM25 metadata has {len(document_ids)} documents, expected {EXPECTED_DOCUMENTS}"
        )
    return rows


def iter_model_inputs(
    corpus: Path, config: ChunkConfig
) -> Iterator[tuple[dict[str, Any], str]]:
    for document in jsonl_rows(corpus):
        document_id = document.get("document_id")
        title = document.get("title")
        text = document.get("text")
        if not all(
            isinstance(value, str) and value for value in (document_id, title, text)
        ):
            raise ValueError(f"invalid corpus row: {document_id!r}")
        for chunk_index, body in enumerate(chunks_for(text, config)):
            yield (
                {
                    "chunk_id": f"{document_id}::{chunk_index:06d}",
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "body_length": len(body),
                },
                f"{title}\n\n{body}",
            )


def validate_chunk_bodies(
    corpus: Path, config: ChunkConfig, metadata: Sequence[dict[str, Any]]
) -> None:
    generated = iter_model_inputs(corpus, config)
    for row_number, expected in enumerate(metadata):
        actual = next(generated, None)
        if actual is None:
            raise ValueError(f"corpus ended before BM25 chunk row {row_number}")
        if actual[0] != expected:
            raise ValueError(
                f"BM25/corpus chunk mismatch at row {row_number}: "
                f"expected={expected}, generated={actual[0]}"
            )
    if next(generated, None) is not None:
        raise ValueError("corpus generated chunks beyond the BM25 metadata")


def cache_is_compatible(
    manifest_path: Path,
    vectors_path: Path,
    metadata_path: Path,
    corpus_sha256: str,
    config: ChunkConfig,
    chunk_count: int,
) -> bool:
    if not all(path.is_file() for path in (manifest_path, vectors_path, metadata_path)):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    saved_config = manifest.get("chunk_config")
    saved_name = saved_config.get("name") if isinstance(saved_config, dict) else saved_config
    normalized = manifest.get("normalized") is True or manifest.get(
        "normalized_for_cosine"
    ) is True
    compatible = all(
        (
            manifest.get("status") == "complete",
            manifest.get("model_name") == MODEL_NAME,
            manifest.get("model_revision") == MODEL_REVISION,
            manifest.get("corpus_sha256") == corpus_sha256,
            manifest.get("metadata_sha256") == sha256_file(metadata_path),
            saved_name == config.name,
            manifest.get("chunk_count") == chunk_count,
            manifest.get("embedding_dimension") == DIMENSION,
            manifest.get("query_instruction") is None,
            normalized,
        )
    )
    if not compatible:
        return False
    try:
        validate_vectors(vectors_path, chunk_count)
    except (OSError, ValueError):
        return False
    return True


def encode_documents(
    model: SentenceTransformer,
    corpus: Path,
    config: ChunkConfig,
    metadata: Sequence[dict[str, Any]],
    batch_size: int,
    output_path: Path,
) -> dict[str, Any]:
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(metadata), DIMENSION)
    )
    source = iter_model_inputs(corpus, config)
    started = time.perf_counter()
    encoding_seconds = 0.0
    rows_written = 0
    next_report = 5_000
    try:
        while rows_written < len(metadata):
            inputs: list[str] = []
            for offset in range(min(batch_size, len(metadata) - rows_written)):
                row = next(source, None)
                if row is None:
                    raise ValueError(f"corpus ended at chunk {rows_written + offset}")
                actual, model_input = row
                if actual != metadata[rows_written + offset]:
                    raise ValueError(
                        f"BM25/corpus chunk mismatch at row {rows_written + offset}"
                    )
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
            end = rows_written + len(inputs)
            output[rows_written:end] = normalize_float32(vectors)
            rows_written = end
            if rows_written >= next_report or rows_written == len(metadata):
                print(
                    f"[{config.name}] encoded {rows_written}/{len(metadata)} chunks",
                    flush=True,
                )
                next_report += 5_000
        if next(source, None) is not None:
            raise ValueError("corpus generated chunks beyond the BM25 metadata")
        output.flush()
    except BaseException:
        del output
        temporary.unlink(missing_ok=True)
        raise
    del output
    validate_vectors(temporary, len(metadata))
    os.replace(temporary, output_path)
    return {
        "reused": False,
        "document_encoding_seconds": encoding_seconds,
        "document_encoding_wall_seconds": time.perf_counter() - started,
        "chunks_per_second": len(metadata) / encoding_seconds,
        "peak_gpu_memory_allocated_gib": torch.cuda.max_memory_allocated()
        / (1024**3),
        "peak_gpu_memory_reserved_gib": torch.cuda.max_memory_reserved()
        / (1024**3),
    }


def validate_rankings(
    path: Path,
    queries: Sequence[dict[str, Any]],
    valid_chunks: set[str],
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        rows = list(jsonl_rows(path))
    except (OSError, ValueError):
        return None
    if len(rows) != len(queries):
        return None
    for row, query in zip(rows, queries, strict=True):
        if row.get("question_id") != query["question_id"]:
            return None
        if row.get("reference_document_id") != query["reference_document_id"]:
            return None
        candidates = row.get("top_documents")
        if not isinstance(candidates, list) or len(candidates) != TOP_K:
            return None
        seen: set[str] = set()
        for rank, candidate in enumerate(candidates, 1):
            document_id = candidate.get("document_id")
            chunk_id = candidate.get("representative_chunk_id")
            score = candidate.get("score")
            if (
                candidate.get("rank") != rank
                or not isinstance(document_id, str)
                or document_id in seen
                or not isinstance(chunk_id, str)
                or chunk_id not in valid_chunks
                or not chunk_id.startswith(f"{document_id}::")
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                return None
            seen.add(document_id)
    return rows


def result_config_is_compatible(rankings_path: Path, config: ChunkConfig) -> bool:
    metrics_path = rankings_path.with_name("metrics.json")
    if not metrics_path.is_file():
        return False
    try:
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    saved_config = saved.get("chunk_config")
    saved_name = saved_config.get("name") if isinstance(saved_config, dict) else saved_config
    embedded = saved.get("encoding")
    settings = (
        embedded
        if isinstance(embedded, dict) and "model_name" in embedded
        else saved
    )
    return all(
        (
            saved.get("status") == "complete",
            saved.get("evaluation_split") == "TRAIN only",
            settings.get("model_name") == MODEL_NAME,
            settings.get("model_revision") == MODEL_REVISION,
            settings.get("query_instruction") is None,
            saved_name == config.name,
        )
    )


def metrics_from_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ranks: list[int | None] = []
    for row in rows:
        reference_id = row["reference_document_id"]
        ranks.append(
            next(
                (
                    candidate["rank"]
                    for candidate in row["top_documents"]
                    if candidate["document_id"] == reference_id
                ),
                None,
            )
        )
    output: dict[str, Any] = {"query_count": len(rows)}
    for cutoff in (1, 5, 10, 20, 50, 100):
        hits = sum(rank is not None and rank <= cutoff for rank in ranks)
        output[f"hits_at_{cutoff}"] = hits
        output[f"recall_at_{cutoff}"] = hits / len(rows)
    output["mrr_at_10"] = sum(
        1.0 / rank if rank is not None and rank <= 10 else 0.0 for rank in ranks
    ) / len(rows)
    return output


def retrieve(
    config: ChunkConfig,
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    metadata: Sequence[dict[str, Any]],
    queries: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for dense retrieval")
    document_ids, starts, ends, _ = document_layout(list(metadata))
    document_tensor = torch.tensor(document_vectors, dtype=torch.float32, device="cuda")
    query_tensor = torch.tensor(query_vectors, dtype=torch.float32, device="cuda")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for query_index, query in enumerate(queries):
            chunk_scores = torch.mv(document_tensor, query_tensor[query_index]).cpu().numpy()
            document_scores = np.maximum.reduceat(chunk_scores, starts)
            order = np.lexsort((document_ids, -document_scores))[:TOP_K]
            candidates: list[dict[str, Any]] = []
            for rank, raw_document_index in enumerate(order, 1):
                document_index = int(raw_document_index)
                start = int(starts[document_index])
                end = int(ends[document_index])
                representative = start + int(np.argmax(chunk_scores[start:end]))
                candidates.append(
                    {
                        "rank": rank,
                        "document_id": str(document_ids[document_index]),
                        "score": float(document_scores[document_index]),
                        "representative_chunk_id": metadata[representative]["chunk_id"],
                    }
                )
            reference_id = query["reference_document_id"]
            rows.append(
                {
                    "question_id": query["question_id"],
                    "reference_document_id": reference_id,
                    "reference_document_rank": next(
                        (
                            item["rank"]
                            for item in candidates
                            if item["document_id"] == reference_id
                        ),
                        None,
                    ),
                    "top_documents": candidates,
                }
            )
            if (query_index + 1) % 100 == 0:
                print(
                    f"[{config.name}] retrieved {query_index + 1}/{len(queries)} queries",
                    flush=True,
                )
    finally:
        del document_tensor, query_tensor
    return rows, time.perf_counter() - started


def write_group_results(
    output_dir: Path,
    config: ChunkConfig,
    metadata: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    vectors_path: Path,
    metadata_path: Path,
    encoding: dict[str, Any],
    retrieval_seconds: float,
    reused_rankings: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = output_dir / "dense_rankings_top100.jsonl"
    atomic_jsonl(rankings_path, rows)
    summary = {
        "status": "complete",
        "evaluation_split": "TRAIN only",
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "query_instruction": None,
        "chunk_config": asdict(config),
        "document_count": EXPECTED_DOCUMENTS,
        "chunk_count": len(metadata),
        "top_k": TOP_K,
        "tie_break": "score descending, then document_id ascending",
        "document_score": "maximum normalized query/chunk dot product",
        "metrics": metrics_from_rows(rows),
        "retrieval_seconds": retrieval_seconds,
        "encoding": encoding,
        "reused_rankings_from": display_path(reused_rankings)
        if reused_rankings is not None
        else None,
        "artifacts": {
            "rankings": display_path(rankings_path),
            "document_vectors": display_path(vectors_path),
            "query_vectors": display_path(QUERY_CACHE / "embeddings.npy"),
            "chunk_metadata": display_path(metadata_path),
        },
    }
    atomic_json(output_dir / "metrics.json", summary)
    return summary


def run_config(
    config: ChunkConfig,
    args: argparse.Namespace,
    corpus_sha256: str,
    queries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    index_path = args.bm25_index_dir / f"{config.name}.sqlite"
    metadata = load_bm25_metadata(index_path, args.corpus, config)
    valid_chunks = {row["chunk_id"] for row in metadata}
    cache_dir = args.cache_root / f"{MODEL_SLUG}_{config.name}"
    output_dir = args.results_dir / config.name
    metadata_path = cache_dir / "chunk_metadata.jsonl"
    vectors_path = cache_dir / "embeddings.npy"
    manifest_path = cache_dir / "manifest.json"

    if not args.force_rerun and not args.force_reencode:
        candidates = [output_dir / "dense_rankings_top100.jsonl"]
        known = KNOWN_COMPLETE_RANKINGS.get(config.name)
        if known is not None:
            candidates.append(known)
        for rankings_path in candidates:
            if not result_config_is_compatible(rankings_path, config):
                continue
            rows = validate_rankings(rankings_path, queries, valid_chunks)
            if rows is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                if not metadata_file_matches(metadata_path, metadata):
                    atomic_jsonl(metadata_path, metadata)
                summary = write_group_results(
                    output_dir,
                    config,
                    metadata,
                    rows,
                    vectors_path,
                    metadata_path,
                    {"reused": True, "rankings_only": True},
                    0.0,
                    rankings_path,
                )
                print(f"[{config.name}] reused complete rankings", flush=True)
                return summary

    cache_dir.mkdir(parents=True, exist_ok=True)
    if not metadata_file_matches(metadata_path, metadata):
        atomic_jsonl(metadata_path, metadata)
    compatible = cache_is_compatible(
        manifest_path,
        vectors_path,
        metadata_path,
        corpus_sha256,
        config,
        len(metadata),
    )
    model: SentenceTransformer | None = None
    encoding: dict[str, Any]
    if compatible and not args.force_reencode:
        print(f"[{config.name}] reusing document vectors", flush=True)
        encoding = {"reused": True}
    else:
        validate_chunk_bodies(args.corpus, config, metadata)
        model = load_model()
        torch.cuda.reset_peak_memory_stats()
        print(
            f"[{config.name}] encoding {len(metadata)} chunks, "
            f"batch_size={args.batch_size}",
            flush=True,
        )
        encoding = encode_documents(
            model,
            args.corpus,
            config,
            metadata,
            args.batch_size,
            vectors_path,
        )
        atomic_json(
            manifest_path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "status": "complete",
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "offline": True,
                "corpus_sha256": corpus_sha256,
                "metadata_sha256": sha256_file(metadata_path),
                "chunk_config": asdict(config),
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

    query_vectors, model = load_or_encode_queries(model, list(queries), args.batch_size)
    if model is not None:
        del model
        torch.cuda.empty_cache()
    document_vectors = validate_vectors(vectors_path, len(metadata))
    rows, retrieval_seconds = retrieve(
        config, document_vectors, query_vectors, metadata, queries
    )
    summary = write_group_results(
        output_dir,
        config,
        metadata,
        rows,
        vectors_path,
        metadata_path,
        encoding,
        retrieval_seconds,
        None,
    )
    del document_vectors, query_vectors, rows, metadata, valid_chunks
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> int:
    args = parse_args()
    args.corpus = args.corpus.resolve()
    args.queries = args.queries.resolve()
    args.bm25_index_dir = args.bm25_index_dir.resolve()
    args.cache_root = args.cache_root.resolve()
    args.results_dir = args.results_dir.resolve()
    if not args.corpus.is_file() or not args.queries.is_file():
        raise FileNotFoundError(
            f"missing corpus or TRAIN queries: {args.corpus}, {args.queries}"
        )
    queries = list(jsonl_rows(args.queries))
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError(
            f"expected {EXPECTED_QUERIES} TRAIN queries, found {len(queries)}"
        )
    if any(
        not isinstance(row.get("question_id"), str)
        or not isinstance(row.get("query"), str)
        or not isinstance(row.get("reference_document_id"), str)
        for row in queries
    ):
        raise ValueError("TRAIN query fields are incomplete")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    corpus_sha256 = sha256_file(args.corpus)
    summaries: list[dict[str, Any]] = []
    for config_name in args.configs:
        config = CONFIGS[config_name]
        print(f"[{config.name}] starting", flush=True)
        try:
            summaries.append(run_config(config, args, corpus_sha256, queries))
        except Exception as error:
            raise RuntimeError(f"failed chunk_config={config.name}: {error}") from error
        atomic_json(
            args.results_dir / "summary.json",
            {
                "status": "in_progress"
                if len(summaries) < len(args.configs)
                else "complete",
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "evaluation_split": "TRAIN only",
                "query_count": len(queries),
                "requested_configs": list(args.configs),
                "completed_configs": [
                    summary["chunk_config"]["name"] for summary in summaries
                ],
                "results": summaries,
            },
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{config.name}] saved; released vectors and index", flush=True)
    print(args.results_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
