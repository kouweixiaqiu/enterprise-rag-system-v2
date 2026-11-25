#!/usr/bin/env python3
"""Rerank all 450 saved PPLX TRAIN Top-20 candidates via local llama-server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import jsonl_rows  # noqa: E402
from experiments.dense.run_qwen3_reranker_full_train import (  # noqa: E402
    EXPECTED_QUERIES,
    RANKING_DEPTH,
    load_inputs,
    metric_summary,
    reference_rank,
    timing_summary,
)
from experiments.dense.run_qwen3_reranker_smoke import (  # noqa: E402
    CANDIDATE_COUNT,
    INSTRUCTION,
    MODEL_SNAPSHOT,
    PREFIX,
    SUFFIX,
    format_instruction,
    load_passages,
)


MODEL_NAME = "Qwen3-Reranker-4B-Q4_K_M.gguf"
MODEL_PATH = Path(
    "/home/fuxuanyi/models/qwen3-reranker-4b/Qwen3-Reranker-4B-Q4_K_M.gguf"
)
LLAMA_SERVER_BUILD = "1 (b4d6c7d8f)"
DEFAULT_ENDPOINT = "http://127.0.0.1:18080/v1/rerank"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/dense/results/qwen3_reranker_4b_gguf_top20_fixed_600_0_train"
)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREFLIGHT = OUTPUT_DIR / "preflight.json"
RANKINGS = OUTPUT_DIR / "rankings.jsonl"
SUMMARY = OUTPUT_DIR / "summary.json"
QWEN_06_SUMMARY = (
    PROJECT_ROOT
    / "experiments/dense/results/qwen3_reranker_0.6b_top20_fixed_600_0_train/summary.json"
)
EXPECTED_PPLX_HITS_AT_5 = 335
EXPECTED_QWEN_06_HITS_AT_5 = 306
TOKEN_MARGIN = 32
TOKEN_ROUNDING = 64
SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate all inputs and compute the required per-slot context, then stop.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("QWEN4B_RERANK_ENDPOINT", DEFAULT_ENDPOINT),
    )
    parser.add_argument(
        "--validate-question",
        metavar="QUESTION_ID",
        help="Score and checkpoint only this question, then stop.",
    )
    parser.add_argument(
        "--validate-longest",
        action="store_true",
        help="Also score the single longest input recorded by preflight.",
    )
    return parser.parse_args()


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
    atomic_text(
        path,
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
    )


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def question_fingerprint(
    prior: dict[str, Any], query: dict[str, Any], passages: dict[str, str]
) -> str:
    digest = hashlib.sha256()
    digest.update(query["question_id"].encode())
    digest.update(b"\0")
    digest.update(query["query"].encode())
    for candidate in prior["baseline"]["ranking"][:RANKING_DEPTH]:
        digest.update(b"\0")
        digest.update(candidate["document_id"].encode())
        digest.update(b"\0")
        chunk_id = candidate["representative_chunk_id"]
        digest.update(chunk_id.encode())
        if chunk_id in passages:
            digest.update(b"\0")
            digest.update(passages[chunk_id].encode())
    return digest.hexdigest()


def input_fingerprint(
    prior_rows: Sequence[dict[str, Any]],
    queries: dict[str, dict[str, Any]],
    passages: dict[str, str],
) -> str:
    digest = hashlib.sha256()
    for prior in prior_rows:
        digest.update(
            question_fingerprint(prior, queries[prior["question_id"]], passages).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def load_tokenizer() -> AutoTokenizer:
    required = ("tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (MODEL_SNAPSHOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"local Qwen3 tokenizer cache is incomplete: {missing}")
    return AutoTokenizer.from_pretrained(
        MODEL_SNAPSHOT,
        local_files_only=True,
        trust_remote_code=False,
    )


def run_preflight() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"missing local GGUF: {MODEL_PATH}")
    prior_rows, queries = load_inputs()
    passages = load_passages(prior_rows)
    tokenizer = load_tokenizer()
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

    longest = 0
    longest_case: dict[str, Any] | None = None
    pair_count = 0
    for prior in prior_rows:
        question_id = prior["question_id"]
        query = queries[question_id]["query"]
        for candidate in prior["baseline"]["ranking"][:CANDIDATE_COUNT]:
            chunk_id = candidate["representative_chunk_id"]
            middle = tokenizer.encode(
                format_instruction(query, passages[chunk_id]),
                add_special_tokens=False,
            )
            token_count = len(prefix_tokens) + len(middle) + len(suffix_tokens)
            pair_count += 1
            if token_count > longest:
                longest = token_count
                longest_case = {
                    "question_id": question_id,
                    "document_id": candidate["document_id"],
                    "representative_chunk_id": chunk_id,
                    "tokens": token_count,
                }
    if pair_count != EXPECTED_QUERIES * CANDIDATE_COUNT:
        raise ValueError(f"unexpected pair count: {pair_count}")

    # This is the same official Qwen3 reranker template/tokenizer used by the
    # completed 0.6B run. Matching its recorded maximum guards against template
    # drift before the GGUF server is launched.
    qwen_06 = json.loads(QWEN_06_SUMMARY.read_text(encoding="utf-8"))
    recorded_longest = int(qwen_06["longest_input_tokens"])
    if longest != recorded_longest:
        raise ValueError(
            f"input template/tokenization drift: {longest} != {recorded_longest}"
        )
    if qwen_06["metrics"]["qwen_top_20"]["hits_at_5"] != EXPECTED_QWEN_06_HITS_AT_5:
        raise ValueError("saved Qwen 0.6B comparator changed")

    required_slot_context = round_up(longest + TOKEN_MARGIN, TOKEN_ROUNDING)
    result = {
        "status": "passed",
        "evaluation_split": "TRAIN only",
        "query_count": EXPECTED_QUERIES,
        "candidate_documents_per_query": CANDIDATE_COUNT,
        "pair_count": pair_count,
        "input_template": "official Qwen3 reranker yes/no template",
        "instruction": INSTRUCTION,
        "longest_input_tokens": longest,
        "longest_case": longest_case,
        "token_margin": TOKEN_MARGIN,
        "required_slot_context": required_slot_context,
        "input_fingerprint_sha256": input_fingerprint(prior_rows, queries, passages),
        "model_path": str(MODEL_PATH),
        "model_size_bytes": MODEL_PATH.stat().st_size,
        "reference_labels_used": False,
    }
    atomic_json(PREFLIGHT, result)
    print(f"REQUIRED_SLOT_CONTEXT={required_slot_context}")
    print("PREFLIGHT_RESULT=" + json.dumps(result, ensure_ascii=False))
    return result


def rerank(endpoint: str, query: str, documents: list[str]) -> list[float]:
    body = json.dumps(
        {"model": MODEL_NAME, "query": query, "documents": documents},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"llama-server request failed: {exc.reason}") from None
    rows = result.get("results")
    if not isinstance(rows, list) or len(rows) != len(documents):
        raise ValueError(f"invalid rerank response: {result}")
    scores: list[float | None] = [None] * len(documents)
    for row in rows:
        index = row.get("index")
        score = row.get("relevance_score")
        if (
            not isinstance(index, int)
            or not 0 <= index < len(documents)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
            or scores[index] is not None
        ):
            raise ValueError(f"invalid rerank result row: {row}")
        scores[index] = float(score)
    if any(score is None for score in scores):
        raise ValueError("rerank response omitted a candidate")
    return [float(score) for score in scores]


def checkpoint_path(question_id: str) -> Path:
    return CHECKPOINT_DIR / f"{question_id}.json"


def validate_checkpoint(
    row: dict[str, Any],
    prior: dict[str, Any],
    query: dict[str, Any],
    passages: dict[str, str],
) -> None:
    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("question_id") != query["question_id"]
        or row.get("reference_document_id") != query["reference_document_id"]
        or row.get("model") != MODEL_NAME
        or row.get("scores_valid") is not True
        or row.get("top_20_document_set_unchanged") is not True
        or row.get("rank_21_50_order_unchanged") is not True
        or row.get("input_fingerprint_sha256")
        != question_fingerprint(prior, query, passages)
    ):
        raise ValueError(f"incompatible checkpoint for {query['question_id']}")
    baseline = prior["baseline"]["ranking"][:RANKING_DEPTH]
    qwen = row.get("qwen_top_20")
    if not isinstance(qwen, list) or len(qwen) != RANKING_DEPTH:
        raise ValueError(f"incomplete checkpoint for {query['question_id']}")
    if {item["document_id"] for item in qwen[:CANDIDATE_COUNT]} != {
        item["document_id"] for item in baseline[:CANDIDATE_COUNT]
    }:
        raise ValueError(f"checkpoint Top-20 changed for {query['question_id']}")
    if [item["document_id"] for item in qwen[CANDIDATE_COUNT:]] != [
        item["document_id"] for item in baseline[CANDIDATE_COUNT:]
    ]:
        raise ValueError(f"checkpoint tail changed for {query['question_id']}")


def build_row(
    prior: dict[str, Any],
    query: dict[str, Any],
    scores: list[float],
    scoring_seconds: float,
    passages: dict[str, str],
) -> dict[str, Any]:
    baseline = prior["baseline"]["ranking"][:RANKING_DEPTH]
    head = baseline[:CANDIDATE_COUNT]
    tail = baseline[CANDIDATE_COUNT:]
    scored_head = [
        {
            "document_id": candidate["document_id"],
            "original_rank": original_rank,
            "pplx_score": candidate["pplx_score"],
            "representative_chunk_id": candidate["representative_chunk_id"],
            "qwen_relevance_score": score,
        }
        for original_rank, (candidate, score) in enumerate(
            zip(head, scores, strict=True), 1
        )
    ]
    scored_head.sort(key=lambda item: (-item["qwen_relevance_score"], item["original_rank"]))
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

    baseline_top20 = {item["document_id"] for item in head}
    if baseline_top20 != {item["document_id"] for item in qwen[:CANDIDATE_COUNT]}:
        raise ValueError(f"Top-20 set changed for {query['question_id']}")
    if [item["document_id"] for item in qwen[CANDIDATE_COUNT:]] != [
        item["document_id"] for item in tail
    ]:
        raise ValueError(f"rank 21-50 order changed for {query['question_id']}")

    reference_id = query["reference_document_id"]
    return {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "question_id": query["question_id"],
        "reference_document_id": reference_id,
        "input_fingerprint_sha256": question_fingerprint(prior, query, passages),
        "reference_ranks": {
            "pplx_baseline": reference_rank(reference_id, baseline),
            "qwen_4b_top_20": reference_rank(reference_id, qwen),
        },
        "qwen_scoring_seconds": scoring_seconds,
        "top_20_document_set_unchanged": True,
        "rank_21_50_order_unchanged": True,
        "scores_valid": True,
        "pplx_baseline": baseline,
        "qwen_top_20": qwen,
    }


def load_or_score(
    endpoint: str,
    prior_rows: Sequence[dict[str, Any]],
    queries: dict[str, dict[str, Any]],
    passages: dict[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    resumed = 0
    for query_index, prior in enumerate(prior_rows, 1):
        question_id = prior["question_id"]
        query = queries[question_id]
        path = checkpoint_path(question_id)
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            validate_checkpoint(row, prior, query, passages)
            output.append(row)
            resumed += 1
            print(f"reused {query_index}/{EXPECTED_QUERIES}: {question_id}", flush=True)
            continue

        head = prior["baseline"]["ranking"][:CANDIDATE_COUNT]
        documents = [passages[item["representative_chunk_id"]] for item in head]
        started = time.perf_counter()
        scores = rerank(endpoint, query["query"], documents)
        elapsed = time.perf_counter() - started
        row = build_row(prior, query, scores, elapsed, passages)
        atomic_json(path, row)
        output.append(row)
        print(
            f"scored {query_index}/{EXPECTED_QUERIES}: {question_id} "
            f"seconds={elapsed:.3f}",
            flush=True,
        )
    print(f"checkpoint_resume_count={resumed}", flush=True)
    return output


def summarize(rows: Sequence[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    pplx_ranks = [row["reference_ranks"]["pplx_baseline"] for row in rows]
    qwen_ranks = [row["reference_ranks"]["qwen_4b_top_20"] for row in rows]
    timings = [float(row["qwen_scoring_seconds"]) for row in rows]
    metrics = {
        "pplx_baseline": metric_summary(pplx_ranks),
        "qwen_4b_top_20": metric_summary(qwen_ranks),
    }
    if metrics["pplx_baseline"]["hits_at_5"] != EXPECTED_PPLX_HITS_AT_5:
        raise ValueError("PPLX 335/450 baseline changed")
    qwen_06 = json.loads(QWEN_06_SUMMARY.read_text(encoding="utf-8"))
    qwen_06_hits = qwen_06["metrics"]["qwen_top_20"]["hits_at_5"]
    if qwen_06_hits != EXPECTED_QWEN_06_HITS_AT_5:
        raise ValueError("Qwen 0.6B 306/450 comparator changed")

    new_into_top_5 = sum(
        (before is None or before > 5) and after is not None and after <= 5
        for before, after in zip(pplx_ranks, qwen_ranks, strict=True)
    )
    dropped_out_of_top_5 = sum(
        before is not None and before <= 5 and (after is None or after > 5)
        for before, after in zip(pplx_ranks, qwen_ranks, strict=True)
    )
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    return {
        "status": "complete",
        "evaluation_split": "TRAIN only",
        "query_count": EXPECTED_QUERIES,
        "candidate_documents_per_query": RANKING_DEPTH,
        "reranked_documents_per_query": CANDIDATE_COUNT,
        "pair_count": EXPECTED_QUERIES * CANDIDATE_COUNT,
        "model": MODEL_NAME,
        "model_path": str(MODEL_PATH),
        "llama_server_build": LLAMA_SERVER_BUILD,
        "score_method": "llama-server /v1/rerank relevance_score",
        "longest_input_tokens": preflight["longest_input_tokens"],
        "required_slot_context": preflight["required_slot_context"],
        "all_top_20_document_sets_unchanged": True,
        "all_rank_21_50_orders_unchanged": True,
        "all_scores_valid": True,
        "metrics": metrics,
        "comparators": {
            "pplx_baseline": metrics["pplx_baseline"],
            "qwen_0.6b_top_20": qwen_06["metrics"]["qwen_top_20"],
        },
        "qwen_4b_vs_pplx_top_5": {
            "new_into_top_5": new_into_top_5,
            "dropped_out_of_top_5": dropped_out_of_top_5,
            "net_change": new_into_top_5 - dropped_out_of_top_5,
        },
        "rerank_timing": timing_summary(timings),
        "experiment_wall_seconds": wall_seconds,
    }


def run_targeted_validation(args: argparse.Namespace) -> None:
    if not PREFLIGHT.is_file():
        raise FileNotFoundError("preflight.json is missing; run with --preflight first")
    prior_rows, queries = load_inputs()
    passages = load_passages(prior_rows)
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("input_fingerprint_sha256") != input_fingerprint(
        prior_rows, queries, passages
    ):
        raise ValueError("saved preflight does not match current TRAIN inputs")

    reusable: list[str] = []
    for prior in prior_rows:
        question_id = prior["question_id"]
        path = checkpoint_path(question_id)
        if not path.is_file():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        validate_checkpoint(row, prior, queries[question_id], passages)
        reusable.append(question_id)
    print("VALIDATION_REUSABLE_CHECKPOINTS=" + json.dumps(reusable), flush=True)

    if args.validate_question:
        prior_by_id = {row["question_id"]: row for row in prior_rows}
        prior = prior_by_id.get(args.validate_question)
        if prior is None:
            raise ValueError(f"unknown TRAIN question: {args.validate_question}")
        query = queries[args.validate_question]
        path = checkpoint_path(args.validate_question)
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            validate_checkpoint(row, prior, query, passages)
            print(f"VALIDATION_QUESTION_REUSED={args.validate_question}", flush=True)
        else:
            head = prior["baseline"]["ranking"][:CANDIDATE_COUNT]
            documents = [passages[item["representative_chunk_id"]] for item in head]
            started = time.perf_counter()
            scores = rerank(args.endpoint, query["query"], documents)
            elapsed = time.perf_counter() - started
            row = build_row(prior, query, scores, elapsed, passages)
            validate_checkpoint(row, prior, query, passages)
            atomic_json(path, row)
            print(
                "VALIDATION_QUESTION_RESULT="
                + json.dumps(
                    {
                        "question_id": args.validate_question,
                        "seconds": elapsed,
                        "checkpoint": str(path),
                        "reference_ranks": row["reference_ranks"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    if args.validate_longest:
        longest = preflight["longest_case"]
        query = queries[longest["question_id"]]
        passage = passages[longest["representative_chunk_id"]]
        started = time.perf_counter()
        scores = rerank(args.endpoint, query["query"], [passage])
        elapsed = time.perf_counter() - started
        print(
            "VALIDATION_LONGEST_RESULT="
            + json.dumps(
                {
                    **longest,
                    "score": scores[0],
                    "seconds": elapsed,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
def main() -> int:
    args = parse_args()
    if args.preflight:
        run_preflight()
        return 0
    if args.validate_question or args.validate_longest:
        run_targeted_validation(args)
        return 0
    if not PREFLIGHT.is_file():
        raise FileNotFoundError("preflight.json is missing; run with --preflight first")

    wall_started = time.perf_counter()
    prior_rows, queries = load_inputs()
    passages = load_passages(prior_rows)
    current_fingerprint = input_fingerprint(prior_rows, queries, passages)
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("input_fingerprint_sha256") != current_fingerprint:
        raise ValueError("saved preflight does not match current TRAIN inputs")
    print(
        f"Qwen3 4B GGUF TRAIN: queries={EXPECTED_QUERIES} "
        f"pairs={EXPECTED_QUERIES * CANDIDATE_COUNT} endpoint={args.endpoint}",
        flush=True,
    )
    rows = load_or_score(args.endpoint, prior_rows, queries, passages)
    if len(rows) != EXPECTED_QUERIES:
        raise ValueError(f"incomplete output: {len(rows)}")
    summary = summarize(rows, time.perf_counter() - wall_started)
    atomic_jsonl(RANKINGS, rows)
    atomic_json(SUMMARY, summary)
    print("QWEN4B_FULL_RESULT=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
