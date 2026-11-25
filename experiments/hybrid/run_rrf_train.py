#!/usr/bin/env python3
"""Run document-level weighted RRF over saved TechQA TRAIN rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BM25 = (
    PROJECT_ROOT / "experiments/hybrid/results/fixed_600_0/bm25_top100.jsonl"
)
DEFAULT_DENSE = (
    PROJECT_ROOT / "experiments/hybrid/results/fixed_600_0/dense_top100.jsonl"
)
DEFAULT_RESULTS = PROJECT_ROOT / "experiments/hybrid/results/fixed_600_0_rrf"
RRF_K = 60
QUERY_COUNT = 450


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25-rankings", type=Path, default=DEFAULT_BM25)
    parser.add_argument("--dense-rankings", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


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


def validate_inputs(
    bm25_rows: list[dict[str, Any]], dense_rows: list[dict[str, Any]]
) -> None:
    if len(bm25_rows) != QUERY_COUNT or len(dense_rows) != QUERY_COUNT:
        raise ValueError(
            f"Expected {QUERY_COUNT} rows per route, found "
            f"BM25={len(bm25_rows)}, Dense={len(dense_rows)}"
        )
    for bm25, dense in zip(bm25_rows, dense_rows, strict=True):
        if (
            bm25["question_id"] != dense["question_id"]
            or bm25["reference_document_id"] != dense["reference_document_id"]
        ):
            raise ValueError("BM25 and Dense query rows are not aligned")
        for route_name, row in (("BM25", bm25), ("Dense", dense)):
            documents = row["top_documents"]
            if len(documents) != 100:
                raise ValueError(
                    f"{route_name} does not have Top-100 for {row['question_id']}"
                )
            ids = [item["document_id"] for item in documents]
            if len(set(ids)) != 100:
                raise ValueError(
                    f"{route_name} has duplicate documents for {row['question_id']}"
                )
            if [item["rank"] for item in documents] != list(range(1, 101)):
                raise ValueError(
                    f"{route_name} ranks are not 1..100 for {row['question_id']}"
                )


def metrics_from_ranks(ranks: list[int | None]) -> dict[str, float | int]:
    denominator = len(ranks)
    return {
        "query_count": denominator,
        "recall_at_1": sum(rank == 1 for rank in ranks) / denominator,
        "recall_at_5": sum(
            rank is not None and rank <= 5 for rank in ranks
        )
        / denominator,
        "mrr_at_10": sum(
            1.0 / rank if rank is not None and rank <= 10 else 0.0
            for rank in ranks
        )
        / denominator,
    }


def baseline_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    return metrics_from_ranks(
        [row["reference_document_rank"] for row in rows]
    )


def fuse_query(
    bm25_documents: list[dict[str, Any]],
    dense_documents: list[dict[str, Any]],
    depth: int,
    bm25_weight: float,
    dense_weight: float,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for route_name, documents, weight in (
        ("bm25", bm25_documents[:depth], bm25_weight),
        ("dense", dense_documents[:depth], dense_weight),
    ):
        for item in documents:
            document_id = item["document_id"]
            candidate = candidates.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "dense_rank": None,
                },
            )
            candidate[f"{route_name}_rank"] = item["rank"]
            candidate["rrf_score"] += weight / (RRF_K + item["rank"])

    ordered = sorted(
        candidates.values(), key=lambda item: (-item["rrf_score"], item["document_id"])
    )
    for rank, item in enumerate(ordered, start=1):
        item["rank"] = rank
    return ordered


def run_config(
    name: str,
    bm25_rows: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    depth: int,
    bm25_weight: float,
    dense_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    reference_ranks: list[int | None] = []
    candidate_counts = []
    for bm25, dense in zip(bm25_rows, dense_rows, strict=True):
        fused = fuse_query(
            bm25["top_documents"],
            dense["top_documents"],
            depth,
            bm25_weight,
            dense_weight,
        )
        reference_id = bm25["reference_document_id"]
        reference_rank = next(
            (
                item["rank"]
                for item in fused
                if item["document_id"] == reference_id
            ),
            None,
        )
        reference_ranks.append(reference_rank)
        candidate_counts.append(len(fused))
        rows.append(
            {
                "question_id": bm25["question_id"],
                "reference_document_id": reference_id,
                "reference_document_rank": reference_rank,
                "reference_rank_scope": "full fused candidate union; null means absent",
                "fused_documents": fused,
            }
        )
    metrics = {
        **metrics_from_ranks(reference_ranks),
        "name": name,
        "rrf_k": RRF_K,
        "input_depth_per_route": depth,
        "bm25_weight": bm25_weight,
        "dense_weight": dense_weight,
        "candidate_union_count_min": min(candidate_counts),
        "candidate_union_count_max": max(candidate_counts),
        "candidate_union_count_mean": sum(candidate_counts) / len(candidate_counts),
    }
    return rows, metrics


def add_dense_delta(
    metrics: dict[str, Any], dense_metrics: dict[str, Any]
) -> dict[str, Any]:
    return {
        **metrics,
        "delta_vs_dense": {
            metric: metrics[metric] - dense_metrics[metric]
            for metric in ("recall_at_1", "recall_at_5", "mrr_at_10")
        },
    }


def main() -> int:
    args = parse_args()
    args.bm25_rankings = args.bm25_rankings.resolve()
    args.dense_rankings = args.dense_rankings.resolve()
    args.results_dir = args.results_dir.resolve()
    bm25_rows = read_jsonl(args.bm25_rankings)
    dense_rows = read_jsonl(args.dense_rankings)
    validate_inputs(bm25_rows, dense_rows)

    bm25_metrics = baseline_metrics(bm25_rows)
    dense_metrics = baseline_metrics(dense_rows)
    runs: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    equal_metrics = []
    for depth in (20, 50):
        name = f"rrf_k60_depth{depth}_bm25_1_dense_1"
        rows, metrics = run_config(
            name, bm25_rows, dense_rows, depth, 1.0, 1.0
        )
        runs.append((rows, metrics))
        equal_metrics.append(metrics)

    selected_depth = min(
        equal_metrics,
        key=lambda item: (-item["mrr_at_10"], item["input_depth_per_route"]),
    )["input_depth_per_route"]
    for bm25_weight, dense_weight in ((1.0, 2.0), (2.0, 1.0)):
        name = (
            f"rrf_k60_depth{selected_depth}_bm25_{int(bm25_weight)}_"
            f"dense_{int(dense_weight)}"
        )
        rows, metrics = run_config(
            name,
            bm25_rows,
            dense_rows,
            selected_depth,
            bm25_weight,
            dense_weight,
        )
        runs.append((rows, metrics))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    hybrid_results = []
    artifacts = {}
    for rows, metrics in runs:
        rankings_path = args.results_dir / f"{metrics['name']}.jsonl"
        write_jsonl(rankings_path, rows)
        enriched = add_dense_delta(metrics, dense_metrics)
        hybrid_results.append(enriched)
        artifacts[metrics["name"]] = str(rankings_path.relative_to(PROJECT_ROOT))

    summary = {
        "evaluation_split": "TRAIN only",
        "query_count": QUERY_COUNT,
        "chunk_config": "fixed_600_0",
        "dense_setting": "BAAI/bge-small-en-v1.5 without query instruction",
        "rrf": {
            "k": RRF_K,
            "rank_origin": 1,
            "formula": "sum(route_weight / (k + route_rank))",
            "candidate_merge_key": "document_id",
            "missing_route_contribution": 0,
            "sort": "rrf_score descending, document_id ascending",
        },
        "depth_selection": {
            "candidates": [20, 50],
            "criterion": "highest MRR@10; smaller depth on exact tie",
            "selected_depth": selected_depth,
        },
        "baselines": [
            add_dense_delta({"name": "BM25", **bm25_metrics}, dense_metrics),
            add_dense_delta({"name": "Dense", **dense_metrics}, dense_metrics),
        ],
        "hybrid_results": hybrid_results,
        "source_rankings": {
            "bm25": str(args.bm25_rankings.relative_to(PROJECT_ROOT)),
            "dense": str(args.dense_rankings.relative_to(PROJECT_ROOT)),
        },
        "artifacts": artifacts,
    }
    summary_path = args.results_dir / "rrf_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
