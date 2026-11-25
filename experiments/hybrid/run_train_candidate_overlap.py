#!/usr/bin/env python3
"""Compare BM25 and dense Top-100 candidate coverage on TechQA TRAIN."""

from __future__ import annotations

import argparse
import json
import platform
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import (  # noqa: E402
    jsonl_rows,
    retrieve_top_documents,
)
from experiments.dense.run_dense_fixed_600_0 import (  # noqa: E402
    count_truncated,
    document_layout,
    encode,
    load_model,
    model_revision,
    synchronize,
    validate_embeddings,
)


DEFAULT_QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
DEFAULT_BM25_INDEX = (
    PROJECT_ROOT
    / "data/processed/techqa/bm25_chunk_indexes/fixed_600_0.sqlite"
)
DEFAULT_DENSE_CACHE = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/bge-small-en-v1.5_fixed_600_0"
)
DEFAULT_RESULTS = PROJECT_ROOT / "experiments/hybrid/results/fixed_600_0"
DIAGNOSTIC_KS = (10, 20, 50, 100)
TOP_K = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--bm25-index", type=Path, default=DEFAULT_BM25_INDEX)
    parser.add_argument("--dense-cache", type=Path, default=DEFAULT_DENSE_CACHE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--query-batch-size", type=int, default=64)
    return parser.parse_args()


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


def load_cache(
    cache_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    with (cache_dir / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    required_manifest = {
        "model_name": "BAAI/bge-small-en-v1.5",
        "chunk_config": {
            "name": "fixed_600_0",
            "strategy": "fixed",
            "size": 600,
            "overlap": 0,
            "step": 600,
        },
        "chunk_count": 182_835,
        "document_count": 28_482,
        "normalized": True,
        "dtype": "float32",
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Dense cache manifest mismatch for {key}: "
                f"expected {expected!r}, found {manifest.get(key)!r}"
            )
    embeddings = np.load(cache_dir / "embeddings.npy", mmap_mode="r")
    metadata = list(jsonl_rows(cache_dir / "chunk_metadata.jsonl"))
    if embeddings.shape != (len(metadata), manifest["embedding_dimension"]):
        raise ValueError(
            f"Dense cache shape mismatch: {embeddings.shape}, metadata={len(metadata)}"
        )
    validate_embeddings(embeddings)
    return manifest, embeddings, metadata


def dense_top_documents(
    chunk_scores: np.ndarray,
    document_scores: np.ndarray,
    order: np.ndarray,
    document_ids: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    metadata: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    documents = []
    for rank, document_index in enumerate(order[:TOP_K], start=1):
        start = int(starts[document_index])
        end = int(ends[document_index])
        representative_index = start + int(np.argmax(chunk_scores[start:end]))
        documents.append(
            {
                "rank": rank,
                "document_id": str(document_ids[document_index]),
                "score": float(document_scores[document_index]),
                "representative_chunk_id": metadata[representative_index]["chunk_id"],
            }
        )
    return documents


def reference_rank(
    documents: list[dict[str, Any]], reference_document_id: str
) -> int | None:
    return next(
        (
            item["rank"]
            for item in documents
            if item["document_id"] == reference_document_id
        ),
        None,
    )


def overlap_statistics(
    bm25_rows: list[dict[str, Any]], dense_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(bm25_rows) != len(dense_rows):
        raise ValueError("BM25 and dense result row counts differ")
    output = []
    for cutoff in DIAGNOSTIC_KS:
        both = bm25_only = dense_only = neither = 0
        for bm25_row, dense_row in zip(bm25_rows, dense_rows, strict=True):
            if bm25_row["question_id"] != dense_row["question_id"]:
                raise ValueError("BM25 and dense query ordering differs")
            bm25_rank = bm25_row["reference_document_rank"]
            dense_rank = dense_row["reference_document_rank"]
            bm25_hit = bm25_rank is not None and bm25_rank <= cutoff
            dense_hit = dense_rank is not None and dense_rank <= cutoff
            if bm25_hit and dense_hit:
                both += 1
            elif bm25_hit:
                bm25_only += 1
            elif dense_hit:
                dense_only += 1
            else:
                neither += 1
        denominator = len(bm25_rows)
        union_hits = both + bm25_only + dense_only
        output.append(
            {
                "cutoff": cutoff,
                "both_hit": both,
                "bm25_only_hit": bm25_only,
                "dense_only_hit": dense_only,
                "neither_hit": neither,
                "bm25_hit_count": both + bm25_only,
                "dense_hit_count": both + dense_only,
                "union_hit_count": union_hits,
                "bm25_recall": (both + bm25_only) / denominator,
                "dense_recall": (both + dense_only) / denominator,
                "union_recall": union_hits / denominator,
            }
        )
    return output


def main() -> int:
    args = parse_args()
    args.queries = args.queries.resolve()
    args.bm25_index = args.bm25_index.resolve()
    args.dense_cache = args.dense_cache.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    queries = list(jsonl_rows(args.queries))
    if len(queries) != 450:
        raise ValueError(f"Expected 450 TRAIN queries, found {len(queries)}")
    manifest, embeddings, metadata = load_cache(args.dense_cache)
    document_ids, starts, ends, _ = document_layout(metadata)
    valid_document_ids = set(document_ids.tolist())

    print(
        f"loading {manifest['model_name']}@{manifest['model_revision']} on {args.device}",
        flush=True,
    )
    model = load_model(
        manifest["model_name"], manifest["model_revision"], args.device
    )
    resolved_revision = model_revision(model, manifest["model_name"])
    if resolved_revision != manifest["model_revision"]:
        raise ValueError(
            f"Model revision differs from cache: {resolved_revision} != "
            f"{manifest['model_revision']}"
        )
    query_texts = [row["query"] for row in queries]
    query_truncated_count = count_truncated(
        model, query_texts, args.query_batch_size
    )
    query_vectors, query_encoding_seconds = encode(
        model, query_texts, args.query_batch_size, args.device
    )
    validate_embeddings(query_vectors)

    document_tensor = torch.tensor(
        np.asarray(embeddings), dtype=torch.float32, device=args.device
    )
    query_tensor = torch.as_tensor(
        query_vectors, dtype=torch.float32, device=args.device
    )
    bm25_connection = sqlite3.connect(f"file:{args.bm25_index}?mode=ro", uri=True)
    bm25_connection.execute("PRAGMA query_only=ON")
    bm25_rows = []
    dense_rows = []
    bm25_seconds = 0.0
    dense_seconds = 0.0

    try:
        for query_index, query in enumerate(queries):
            question_id = query["question_id"]
            reference_id = query["reference_document_id"]
            if reference_id not in valid_document_ids:
                raise ValueError(f"Reference document missing: {reference_id}")

            started = time.perf_counter()
            bm25_documents = retrieve_top_documents(
                bm25_connection, query["query"], TOP_K
            )
            bm25_seconds += time.perf_counter() - started

            synchronize(args.device)
            started = time.perf_counter()
            chunk_scores = torch.mv(
                document_tensor, query_tensor[query_index]
            ).cpu().numpy()
            document_scores = np.maximum.reduceat(chunk_scores, starts)
            order = np.lexsort((document_ids, -document_scores))
            dense_documents = dense_top_documents(
                chunk_scores,
                document_scores,
                order,
                document_ids,
                starts,
                ends,
                metadata,
            )
            synchronize(args.device)
            dense_seconds += time.perf_counter() - started

            if len(bm25_documents) != TOP_K or len(dense_documents) != TOP_K:
                raise ValueError(f"Did not return Top-{TOP_K} for {question_id}")
            for route, documents in (
                ("BM25", bm25_documents),
                ("Dense", dense_documents),
            ):
                returned_ids = [item["document_id"] for item in documents]
                if len(set(returned_ids)) != TOP_K:
                    raise ValueError(f"{route} returned duplicate documents for {question_id}")
                invalid = set(returned_ids) - valid_document_ids
                if invalid:
                    raise ValueError(f"{route} returned invalid IDs: {sorted(invalid)}")

            common = {
                "question_id": question_id,
                "reference_document_id": reference_id,
                "reference_rank_scope": "top_100; null means not in top 100",
            }
            bm25_rows.append(
                {
                    **common,
                    "reference_document_rank": reference_rank(
                        bm25_documents, reference_id
                    ),
                    "top_documents": bm25_documents,
                }
            )
            dense_rows.append(
                {
                    **common,
                    "reference_document_rank": reference_rank(
                        dense_documents, reference_id
                    ),
                    "top_documents": dense_documents,
                }
            )
            if (query_index + 1) % 50 == 0:
                print(f"processed {query_index + 1}/450 queries", flush=True)
    finally:
        bm25_connection.close()

    statistics = overlap_statistics(bm25_rows, dense_rows)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    bm25_path = args.results_dir / "bm25_top100.jsonl"
    dense_path = args.results_dir / "dense_top100.jsonl"
    summary_path = args.results_dir / "candidate_overlap_summary.json"
    write_jsonl(bm25_path, bm25_rows)
    write_jsonl(dense_path, dense_rows)
    summary = {
        "evaluation_split": "TRAIN only",
        "query_count": len(queries),
        "chunk_config": "fixed_600_0",
        "diagnostic_cutoffs": list(DIAGNOSTIC_KS),
        "maximum_saved_candidates_per_route": TOP_K,
        "statistics": statistics,
        "bm25": {
            "index": str(args.bm25_index.relative_to(PROJECT_ROOT)),
            "ranking": "existing SQLite FTS5 BM25 chunk ordering; maximum chunk per document",
            "total_retrieval_seconds": bm25_seconds,
            "average_retrieval_seconds": bm25_seconds / len(queries),
        },
        "dense": {
            "cache_manifest": str(
                (args.dense_cache / "manifest.json").relative_to(PROJECT_ROOT)
            ),
            "model_name": manifest["model_name"],
            "model_revision": manifest["model_revision"],
            "device": args.device,
            "cuda_device_name": (
                torch.cuda.get_device_name(0) if args.device == "cuda" else None
            ),
            "query_instruction": None,
            "query_truncated_count": query_truncated_count,
            "query_encoding_seconds": query_encoding_seconds,
            "total_retrieval_seconds": dense_seconds,
            "average_retrieval_seconds": dense_seconds / len(queries),
            "ranking": "normalized exact dot product; maximum chunk per document; document_id tie-break",
        },
        "artifacts": {
            "bm25_rankings": str(bm25_path.relative_to(PROJECT_ROOT)),
            "dense_rankings": str(dense_path.relative_to(PROJECT_ROOT)),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sqlite": sqlite3.sqlite_version,
        },
    }
    write_json(summary_path, summary)
    print(json.dumps(statistics, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
