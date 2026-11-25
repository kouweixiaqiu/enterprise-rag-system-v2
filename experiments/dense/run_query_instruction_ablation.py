#!/usr/bin/env python3
"""Evaluate the BGE recommended query instruction on TechQA TRAIN."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import jsonl_rows  # noqa: E402
from experiments.dense.run_dense_fixed_600_0 import (  # noqa: E402
    count_truncated,
    document_layout,
    encode,
    load_model,
    model_revision,
    synchronize,
    validate_embeddings,
)
from experiments.hybrid.run_train_candidate_overlap import (  # noqa: E402
    dense_top_documents,
    load_cache,
)


QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
TOP_K = 100
DEFAULT_QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
DEFAULT_CACHE = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/bge-small-en-v1.5_fixed_600_0"
)
DEFAULT_BASELINE_METRICS = (
    PROJECT_ROOT / "experiments/dense/results/fixed_600_0/metrics.json"
)
DEFAULT_RESULTS = (
    PROJECT_ROOT / "experiments/dense/results/fixed_600_0_query_instruction"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--dense-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS
    )
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


def evaluate(
    queries: list[dict[str, Any]],
    query_vectors: np.ndarray,
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    document_ids, starts, ends, _ = document_layout(metadata)
    document_id_to_index = {
        document_id: index for index, document_id in enumerate(document_ids)
    }
    document_tensor = torch.tensor(
        np.asarray(embeddings), dtype=torch.float32, device=device
    )
    query_tensor = torch.as_tensor(query_vectors, dtype=torch.float32, device=device)
    rows = []
    retrieval_seconds = 0.0

    for query_index, query in enumerate(queries):
        reference_id = query["reference_document_id"]
        reference_document_index = document_id_to_index.get(reference_id)
        if reference_document_index is None:
            raise ValueError(f"Reference document missing from cache: {reference_id}")

        synchronize(device)
        started = time.perf_counter()
        chunk_scores = torch.mv(
            document_tensor, query_tensor[query_index]
        ).cpu().numpy()
        document_scores = np.maximum.reduceat(chunk_scores, starts)
        order = np.lexsort((document_ids, -document_scores))
        top_documents = dense_top_documents(
            chunk_scores,
            document_scores,
            order,
            document_ids,
            starts,
            ends,
            metadata,
        )
        synchronize(device)
        retrieval_seconds += time.perf_counter() - started

        exact_rank = int(
            np.flatnonzero(order == reference_document_index)[0]
        ) + 1
        if len(top_documents) != TOP_K:
            raise ValueError(f"Did not return Top-{TOP_K} for {query['question_id']}")
        returned_ids = [item["document_id"] for item in top_documents]
        if len(set(returned_ids)) != TOP_K:
            raise ValueError(f"Duplicate Top-{TOP_K} documents for {query['question_id']}")
        rows.append(
            {
                "question_id": query["question_id"],
                "reference_document_id": reference_id,
                "reference_document_rank": exact_rank if exact_rank <= TOP_K else None,
                "reference_document_exact_rank": exact_rank,
                "reference_rank_scope": "top_100; null means not in top 100",
                "top_documents": top_documents,
            }
        )
        if (query_index + 1) % 50 == 0:
            print(f"retrieved {query_index + 1}/450 queries", flush=True)

    denominator = len(rows)
    retrieval = {
        "query_count": denominator,
        "queries_with_100_documents": sum(
            len(row["top_documents"]) == TOP_K for row in rows
        ),
        "recall_at_1": sum(
            row["reference_document_exact_rank"] == 1 for row in rows
        )
        / denominator,
        "recall_at_5": sum(
            row["reference_document_exact_rank"] <= 5 for row in rows
        )
        / denominator,
        "mrr_at_10": sum(
            1.0 / row["reference_document_exact_rank"]
            if row["reference_document_exact_rank"] <= 10
            else 0.0
            for row in rows
        )
        / denominator,
        "exact_search_seconds": retrieval_seconds,
        "average_search_seconds": retrieval_seconds / denominator,
        "exact_reference_ranks": {
            row["question_id"]: row["reference_document_exact_rank"]
            for row in rows
            if row["question_id"] in {"TRAIN_Q003", "TRAIN_Q024"}
        },
    }
    return rows, retrieval


def comparison_row(
    label: str,
    recall_at_1: float,
    recall_at_5: float,
    mrr_at_10: float,
    query_truncated_count: int,
    exact_ranks: dict[str, int],
) -> dict[str, Any]:
    return {
        "setting": label,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "mrr_at_10": mrr_at_10,
        "query_truncated_count": query_truncated_count,
        "train_q003_reference_document_rank": exact_ranks["TRAIN_Q003"],
        "train_q024_reference_document_rank": exact_ranks["TRAIN_Q024"],
    }


def main() -> int:
    args = parse_args()
    args.queries = args.queries.resolve()
    args.dense_cache = args.dense_cache.resolve()
    args.baseline_metrics = args.baseline_metrics.resolve()
    args.results_dir = args.results_dir.resolve()
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    queries = list(jsonl_rows(args.queries))
    if len(queries) != 450:
        raise ValueError(f"Expected 450 TRAIN queries, found {len(queries)}")
    manifest, embeddings, metadata = load_cache(args.dense_cache)
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

    instructed_queries = [QUERY_INSTRUCTION + row["query"] for row in queries]
    query_truncated_count = count_truncated(
        model, instructed_queries, args.query_batch_size
    )
    query_vectors, query_encoding_seconds = encode(
        model,
        instructed_queries,
        args.query_batch_size,
        args.device,
    )
    validate_embeddings(query_vectors)
    rows, retrieval = evaluate(
        queries,
        query_vectors,
        embeddings,
        metadata,
        args.device,
    )

    with args.baseline_metrics.open(encoding="utf-8") as stream:
        baseline = json.load(stream)
    baseline_retrieval = baseline["retrieval"]
    baseline_encoding = baseline["encoding"]
    comparison = {
        "evaluation_split": "TRAIN only",
        "query_count": len(queries),
        "chunk_config": "fixed_600_0",
        "model_name": manifest["model_name"],
        "model_revision": manifest["model_revision"],
        "recommended_query_instruction": QUERY_INSTRUCTION,
        "recommended_query_instruction_repr": repr(QUERY_INSTRUCTION),
        "document_instruction": None,
        "results": [
            comparison_row(
                "without query instruction",
                baseline_retrieval["recall_at_1"],
                baseline_retrieval["recall_at_5"],
                baseline_retrieval["mrr_at_10"],
                baseline_encoding["query_truncated_count"],
                baseline_retrieval["exact_reference_ranks"],
            ),
            comparison_row(
                "with recommended query instruction",
                retrieval["recall_at_1"],
                retrieval["recall_at_5"],
                retrieval["mrr_at_10"],
                query_truncated_count,
                retrieval["exact_reference_ranks"],
            ),
        ],
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = args.results_dir / "query_rankings_top100.jsonl"
    metrics_path = args.results_dir / "metrics.json"
    comparison_path = args.results_dir / "comparison_with_no_instruction.json"
    write_jsonl(rankings_path, rows)
    metrics = {
        "evaluation_split": "TRAIN only",
        "query_count": len(queries),
        "chunk_config": "fixed_600_0",
        "model_name": manifest["model_name"],
        "model_revision": manifest["model_revision"],
        "document_cache_reused": True,
        "document_cache_manifest": str(
            (args.dense_cache / "manifest.json").relative_to(PROJECT_ROOT)
        ),
        "query_instruction": QUERY_INSTRUCTION,
        "query_instruction_repr": repr(QUERY_INSTRUCTION),
        "query_construction": "instruction prefix concatenated directly before original query text",
        "document_instruction": None,
        "query_truncated_count": query_truncated_count,
        "query_encoding_seconds": query_encoding_seconds,
        "device": args.device,
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if args.device == "cuda" else None
        ),
        "similarity": "normalized exact dot product",
        "document_aggregation": "maximum chunk score per document before Top-100",
        "document_tie_break": "document_id ascending",
        "retrieval": retrieval,
        "artifacts": {
            "rankings": str(rankings_path.relative_to(PROJECT_ROOT)),
            "comparison": str(comparison_path.relative_to(PROJECT_ROOT)),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    write_json(metrics_path, metrics)
    write_json(comparison_path, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {rankings_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
