#!/usr/bin/env python3
"""Offline Qwen3 Top-20 reranking of all 450 saved PPLX TRAIN rankings."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import jsonl_rows
from experiments.dense.run_qwen3_reranker_smoke import (
    BATCH_SIZE,
    BGE_RERANK_ROWS,
    CANDIDATE_COUNT,
    INSTRUCTION,
    MAX_LENGTH,
    MODEL_SNAPSHOT,
    QwenReranker,
    TRAIN_QUERIES,
    atomic_json,
    atomic_jsonl,
    load_passages,
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/dense/results/qwen3_reranker_0.6b_top20_fixed_600_0_train"
)
EXPECTED_QUERIES = 450
RANKING_DEPTH = 50


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    prior_rows = list(jsonl_rows(BGE_RERANK_ROWS))
    query_rows = list(jsonl_rows(TRAIN_QUERIES))
    queries = {row["question_id"]: row for row in query_rows}
    if len(prior_rows) != EXPECTED_QUERIES or len(queries) != EXPECTED_QUERIES:
        raise ValueError("expected exactly 450 saved reranker rows and TRAIN queries")
    if len(query_rows) != len(queries):
        raise ValueError("duplicate TRAIN question IDs")
    for row in prior_rows:
        question_id = row["question_id"]
        query = queries.get(question_id)
        if query is None or query["reference_document_id"] != row["reference_document_id"]:
            raise ValueError(f"TRAIN query mismatch for {question_id}")
        baseline = row["baseline"]["ranking"][:RANKING_DEPTH]
        bge = row["rerank_top_20"]["ranking"][:RANKING_DEPTH]
        if len(baseline) != RANKING_DEPTH or len(bge) != RANKING_DEPTH:
            raise ValueError(f"incomplete saved ranking for {question_id}")
        baseline_ids = {candidate["document_id"] for candidate in baseline}
        if len(baseline_ids) != RANKING_DEPTH:
            raise ValueError(f"duplicate PPLX documents for {question_id}")
        if baseline_ids != {candidate["document_id"] for candidate in bge}:
            raise ValueError(f"saved BGE candidate set differs for {question_id}")
    return prior_rows, queries


def reference_rank(reference_id: str, rows: Sequence[dict[str, Any]]) -> int | None:
    return next(
        (rank for rank, row in enumerate(rows, 1) if row["document_id"] == reference_id),
        None,
    )


def metric_summary(ranks: Sequence[int | None]) -> dict[str, Any]:
    hits_at_5 = sum(rank is not None and rank <= 5 for rank in ranks)
    return {
        "query_count": len(ranks),
        "hits_at_5": hits_at_5,
        "recall_at_5": hits_at_5 / len(ranks),
        "mrr_at_10": sum(
            1.0 / rank if rank is not None and rank <= 10 else 0.0 for rank in ranks
        )
        / len(ranks),
    }


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "total_seconds": sum(values),
        "mean_seconds_per_query": float(np.mean(values)),
        "p95_seconds_per_query": ordered[p95_index],
    }


def main() -> int:
    wall_started = time.perf_counter()
    prior_rows, queries = load_inputs()
    passages = load_passages(prior_rows)
    print(
        f"offline Qwen full TRAIN: queries={EXPECTED_QUERIES} "
        f"pairs={EXPECTED_QUERIES * CANDIDATE_COUNT} batch_size={BATCH_SIZE}",
        flush=True,
    )
    reranker = QwenReranker()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    output_rows: list[dict[str, Any]] = []
    qwen_times: list[float] = []
    bge_times: list[float] = []
    pplx_ranks: list[int | None] = []
    bge_ranks: list[int | None] = []
    qwen_ranks: list[int | None] = []
    longest_input = 0
    scoring_started = time.perf_counter()
    for query_index, prior in enumerate(prior_rows, 1):
        query = queries[prior["question_id"]]
        baseline = prior["baseline"]["ranking"][:RANKING_DEPTH]
        bge = prior["rerank_top_20"]["ranking"][:RANKING_DEPTH]
        head = baseline[:CANDIDATE_COUNT]
        tail = baseline[CANDIDATE_COUNT:]
        documents = [passages[row["representative_chunk_id"]] for row in head]

        torch.cuda.synchronize()
        query_started = time.perf_counter()
        scores, query_longest = reranker.score(query["query"], documents)
        torch.cuda.synchronize()
        query_seconds = time.perf_counter() - query_started
        longest_input = max(longest_input, query_longest)
        if len(scores) != CANDIDATE_COUNT or not np.isfinite(scores).all():
            raise ValueError(f"invalid scores for {query['question_id']}")
        if not all(0.0 <= score <= 1.0 for score in scores):
            raise ValueError(f"out-of-range scores for {query['question_id']}")

        scored_head = [
            {
                "document_id": candidate["document_id"],
                "original_rank": original_rank,
                "pplx_score": candidate["pplx_score"],
                "representative_chunk_id": candidate["representative_chunk_id"],
                "qwen_relevance_score": float(score),
            }
            for original_rank, (candidate, score) in enumerate(
                zip(head, scores, strict=True), 1
            )
        ]
        scored_head.sort(
            key=lambda row: (-row["qwen_relevance_score"], row["original_rank"])
        )
        qwen = scored_head + [
            {
                "document_id": candidate["document_id"],
                "original_rank": original_rank,
                "pplx_score": candidate["pplx_score"],
                "representative_chunk_id": candidate["representative_chunk_id"],
                "qwen_relevance_score": None,
            }
            for original_rank, candidate in enumerate(tail, CANDIDATE_COUNT + 1)
        ]
        for rank, candidate in enumerate(qwen, 1):
            candidate["rank"] = rank

        baseline_top20 = {row["document_id"] for row in head}
        qwen_top20 = {row["document_id"] for row in qwen[:CANDIDATE_COUNT]}
        baseline_top50 = {row["document_id"] for row in baseline}
        qwen_top50 = {row["document_id"] for row in qwen}
        if baseline_top20 != qwen_top20 or len(qwen_top20) != CANDIDATE_COUNT:
            raise ValueError(f"Top-20 set changed for {query['question_id']}")
        if baseline_top50 != qwen_top50 or len(qwen_top50) != RANKING_DEPTH:
            raise ValueError(f"Top-50 set changed for {query['question_id']}")
        if [row["document_id"] for row in qwen[CANDIDATE_COUNT:]] != [
            row["document_id"] for row in tail
        ]:
            raise ValueError(f"rank 21-50 order changed for {query['question_id']}")

        reference_id = query["reference_document_id"]
        pplx_rank = reference_rank(reference_id, baseline)
        bge_rank = reference_rank(reference_id, bge)
        qwen_rank = reference_rank(reference_id, qwen)
        pplx_ranks.append(pplx_rank)
        bge_ranks.append(bge_rank)
        qwen_ranks.append(qwen_rank)
        qwen_times.append(query_seconds)
        bge_times.append(float(prior["scoring_seconds"]["top_20"]))
        output_rows.append(
            {
                "question_id": query["question_id"],
                "reference_document_id": reference_id,
                "reference_ranks": {
                    "pplx_baseline": pplx_rank,
                    "bge_top_20": bge_rank,
                    "qwen_top_20": qwen_rank,
                },
                "qwen_scoring_seconds": query_seconds,
                "top_20_document_set_unchanged": True,
                "rank_21_50_order_unchanged": True,
                "scores_valid": True,
                "pplx_baseline": baseline,
                "bge_top_20": bge,
                "qwen_top_20": qwen,
            }
        )
        if query_index % 25 == 0 or query_index == EXPECTED_QUERIES:
            print(f"scored {query_index}/{EXPECTED_QUERIES} queries", flush=True)

    scoring_seconds = time.perf_counter() - scoring_started
    metrics = {
        "pplx_baseline": metric_summary(pplx_ranks),
        "bge_top_20": metric_summary(bge_ranks),
        "qwen_top_20": metric_summary(qwen_ranks),
    }
    if metrics["pplx_baseline"]["hits_at_5"] != 335:
        raise ValueError(
            f"PPLX baseline changed: {metrics['pplx_baseline']['hits_at_5']} != 335"
        )
    new_into_top_5 = sum(
        (before is None or before > 5) and after is not None and after <= 5
        for before, after in zip(pplx_ranks, qwen_ranks, strict=True)
    )
    dropped_out_of_top_5 = sum(
        before is not None and before <= 5 and (after is None or after > 5)
        for before, after in zip(pplx_ranks, qwen_ranks, strict=True)
    )
    summary = {
        "status": "complete",
        "evaluation_split": "TRAIN only",
        "query_count": EXPECTED_QUERIES,
        "candidate_documents_per_query": RANKING_DEPTH,
        "reranked_documents_per_query": CANDIDATE_COUNT,
        "pair_count": EXPECTED_QUERIES * CANDIDATE_COUNT,
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "revision": MODEL_SNAPSHOT.name,
        "model_snapshot": str(MODEL_SNAPSHOT),
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "longest_input_tokens": longest_input,
        "instruction": INSTRUCTION,
        "score_method": "softmax(no_logit, yes_logit)[yes]",
        "all_top_20_document_sets_unchanged": True,
        "all_rank_21_50_orders_unchanged": True,
        "all_scores_valid": True,
        "pplx_335_of_450_confirmed": True,
        "metrics": metrics,
        "rerank_timing": {
            "pplx_baseline": {
                "mean_seconds_per_query": 0.0,
                "p95_seconds_per_query": 0.0,
                "note": "no reranking",
            },
            "bge_top_20": timing_summary(bge_times),
            "qwen_top_20": timing_summary(qwen_times),
        },
        "qwen_vs_pplx_top_5": {
            "new_into_top_5": new_into_top_5,
            "dropped_out_of_top_5": dropped_out_of_top_5,
            "net_change": new_into_top_5 - dropped_out_of_top_5,
        },
        "model_load_seconds": reranker.load_seconds,
        "qwen_scoring_seconds": scoring_seconds,
        "experiment_wall_seconds": time.perf_counter() - wall_started,
        "peak_gpu_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_gpu_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
    }
    atomic_jsonl(OUTPUT_DIR / "rankings.jsonl", output_rows)
    atomic_json(OUTPUT_DIR / "summary.json", summary)
    print("QWEN_FULL_RESULT=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
