#!/usr/bin/env python3
"""Validate the completed Qwen3-4B TRAIN ranking and save key metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    PROJECT_ROOT
    / "experiments/dense/results/qwen3_reranker_4b_gguf_top20_fixed_600_0_train"
)
RANKINGS = RESULT_DIR / "rankings.jsonl"
SUMMARY = RESULT_DIR / "summary.json"
QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
EXPECTED_QUERIES = 450
EXPECTED_PPLX_HITS_AT_5 = 335


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def reference_rank(reference_id: str, ranking: Sequence[dict[str, Any]]) -> int | None:
    return next(
        (
            rank
            for rank, candidate in enumerate(ranking, 1)
            if candidate["document_id"] == reference_id
        ),
        None,
    )


def metrics(ranks: Sequence[int | None]) -> dict[str, Any]:
    hits = sum(rank is not None and rank <= 5 for rank in ranks)
    return {
        "hits_at_5": hits,
        "recall_at_5": hits / len(ranks),
        "mrr_at_10": sum(
            1.0 / rank if rank is not None and rank <= 10 else 0.0
            for rank in ranks
        )
        / len(ranks),
    }


def main() -> int:
    args = parse_args()
    rows = list(jsonl_rows(RANKINGS))
    query_rows = list(jsonl_rows(QUERIES))
    queries = {row["question_id"]: row for row in query_rows}
    if len(rows) != EXPECTED_QUERIES or len(query_rows) != EXPECTED_QUERIES:
        raise ValueError("expected exactly 450 rankings and TRAIN queries")
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError("duplicate TRAIN question IDs")
    row_ids = [row.get("question_id") for row in rows]
    if len(set(row_ids)) != EXPECTED_QUERIES or set(row_ids) != set(queries):
        raise ValueError("ranking question IDs are incomplete or duplicated")

    pplx_ranks: list[int | None] = []
    qwen_ranks: list[int | None] = []
    for row in rows:
        question_id = row["question_id"]
        query = queries[question_id]
        if row.get("reference_document_id") != query["reference_document_id"]:
            raise ValueError(f"reference mismatch for {question_id}")
        pplx = row.get("pplx_baseline")
        qwen = row.get("qwen_top_20")
        if not isinstance(pplx, list) or not isinstance(qwen, list):
            raise ValueError(f"missing rankings for {question_id}")
        if len(pplx) != 50 or len(qwen) != 50:
            raise ValueError(f"ranking depth is not 50 for {question_id}")
        pplx_ids = [candidate["document_id"] for candidate in pplx]
        qwen_ids = [candidate["document_id"] for candidate in qwen]
        if len(set(pplx_ids)) != 50 or len(set(qwen_ids)) != 50:
            raise ValueError(f"duplicate document for {question_id}")
        if set(pplx_ids[:20]) != set(qwen_ids[:20]):
            raise ValueError(f"Top-20 set changed for {question_id}")
        if pplx_ids[20:] != qwen_ids[20:]:
            raise ValueError(f"rank 21-50 order changed for {question_id}")
        if row.get("scores_valid") is not True:
            raise ValueError(f"invalid score marker for {question_id}")
        reference_id = query["reference_document_id"]
        pplx_ranks.append(reference_rank(reference_id, pplx))
        qwen_ranks.append(reference_rank(reference_id, qwen))

    pplx_metrics = metrics(pplx_ranks)
    qwen_metrics = metrics(qwen_ranks)
    if pplx_metrics["hits_at_5"] != EXPECTED_PPLX_HITS_AT_5:
        raise ValueError("PPLX baseline is not 335/450")
    new_into_top_5 = sum(
        (before is None or before > 5) and after is not None and after <= 5
        for before, after in zip(pplx_ranks, qwen_ranks, strict=True)
    )
    dropped_out_of_top_5 = sum(
        before is not None and before <= 5 and (after is None or after > 5)
        for before, after in zip(pplx_ranks, qwen_ranks, strict=True)
    )

    saved_summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    if saved_summary.get("status") != "complete":
        raise ValueError("saved full-run summary is not complete")
    saved_metrics = saved_summary.get("metrics", {})
    if saved_metrics.get("pplx_baseline", {}).get("hits_at_5") != 335:
        raise ValueError("saved summary PPLX metric mismatch")
    if saved_metrics.get("qwen_4b_top_20", {}).get("hits_at_5") != qwen_metrics["hits_at_5"]:
        raise ValueError("saved summary Qwen metric mismatch")

    output = {
        "status": "validated_complete",
        "query_count": EXPECTED_QUERIES,
        "rankings_file": str(RANKINGS),
        "full_run_summary_file": str(SUMMARY),
        "pplx_baseline": pplx_metrics,
        "qwen_4b_top_20": qwen_metrics,
        "recall_at_5_change": {
            "hit_count": qwen_metrics["hits_at_5"] - pplx_metrics["hits_at_5"],
            "percentage_points": 100.0
            * (qwen_metrics["recall_at_5"] - pplx_metrics["recall_at_5"]),
        },
        "new_into_top_5": new_into_top_5,
        "dropped_out_of_top_5": dropped_out_of_top_5,
        "net_top_5_change": new_into_top_5 - dropped_out_of_top_5,
    }
    atomic_json(args.output, output)
    print("POSTPROCESS_METRICS=" + json.dumps(output, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
