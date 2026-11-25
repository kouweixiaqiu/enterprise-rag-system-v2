#!/usr/bin/env python3
"""Run Qwen3-Reranker-4B BF16 Top-20 reranking on all 450 TRAIN queries."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.dense.run_qwen3_reranker_full_train import (  # noqa: E402
    EXPECTED_QUERIES,
    RANKING_DEPTH,
    load_inputs,
    reference_rank,
    timing_summary,
)
from experiments.dense.run_qwen3_reranker_smoke import (  # noqa: E402
    CANDIDATE_COUNT,
    INSTRUCTION,
    PREFIX,
    SUFFIX,
    format_instruction,
    load_passages,
)
from experiments.chunk.run_bm25_chunk_experiments import jsonl_rows  # noqa: E402


MODEL_NAME = "Qwen/Qwen3-Reranker-4B"
REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
MODEL_SNAPSHOT = Path(
    "/home/fuxuanyi/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-Reranker-4B/snapshots/"
    + REVISION
)
GGUF_RANKINGS = (
    PROJECT_ROOT
    / "experiments/dense/results/"
    "qwen3_reranker_4b_gguf_top20_fixed_600_0_train/rankings.jsonl"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/dense/results/"
    "qwen3_reranker_4b_bf16_top20_fixed_600_0_train"
)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RANKINGS = OUTPUT_DIR / "rankings.jsonl"
SUMMARY = OUTPUT_DIR / "summary.json"
COMPARISON = OUTPUT_DIR / "comparison.csv"
SMOKE = OUTPUT_DIR / "smoke_test.json"
MAX_LENGTH = 2048
BATCH_SIZE = 1
GPU_MODEL_BUDGET = "4800MiB"
CPU_MODEL_BUDGET = "5000MiB"
SCHEMA_VERSION = 1


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


def input_fingerprint(prior: dict[str, Any], query: dict[str, Any]) -> str:
    payload = {
        "question_id": query["question_id"],
        "query": query["query"],
        "candidates": [
            [candidate["document_id"], candidate["representative_chunk_id"]]
            for candidate in prior["baseline"]["ranking"][:RANKING_DEPTH]
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def checkpoint_path(question_id: str) -> Path:
    return CHECKPOINT_DIR / f"{question_id}.json"


class BF16Reranker:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for BF16 GPU/CPU offload")
        required = (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        )
        missing = [name for name in required if not (MODEL_SNAPSHOT / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete local BF16 snapshot: {missing}")

        load_started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_SNAPSHOT,
            local_files_only=True,
            trust_remote_code=False,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_SNAPSHOT,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
            max_memory={0: GPU_MODEL_BUDGET, "cpu": CPU_MODEL_BUDGET},
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).eval()
        self.load_seconds = time.perf_counter() - load_started
        self.device_map = dict(self.model.hf_device_map)
        if "disk" in set(self.device_map.values()):
            raise RuntimeError("unexpected disk offload; BF16 CPU offload was required")
        parameter_dtypes = {str(parameter.dtype) for parameter in self.model.parameters()}
        if parameter_dtypes != {"torch.bfloat16"}:
            raise ValueError(f"model is not purely BF16: {sorted(parameter_dtypes)}")
        self.input_device = self.model.get_input_embeddings().weight.device
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        if (self.false_token_id, self.true_token_id) != (2152, 9693):
            raise ValueError(
                f"unexpected no/yes token IDs: {self.false_token_id}/{self.true_token_id}"
            )
        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)

    def score(self, query: str, documents: Sequence[str]) -> tuple[list[float], int]:
        scores: list[float] = []
        longest = 0
        with torch.inference_mode():
            for document in documents:
                middle = self.tokenizer.encode(
                    format_instruction(query, document),
                    add_special_tokens=False,
                    truncation=True,
                    max_length=MAX_LENGTH
                    - len(self.prefix_tokens)
                    - len(self.suffix_tokens),
                )
                input_ids = self.prefix_tokens + middle + self.suffix_tokens
                longest = max(longest, len(input_ids))
                batch = {
                    "input_ids": torch.tensor(
                        [input_ids], dtype=torch.long, device=self.input_device
                    ),
                    "attention_mask": torch.ones(
                        (1, len(input_ids)), dtype=torch.long, device=self.input_device
                    ),
                }
                logits = self.model(
                    **batch,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1, :]
                binary_logits = torch.stack(
                    [logits[:, self.false_token_id], logits[:, self.true_token_id]],
                    dim=1,
                ).to(torch.float32)
                score = torch.softmax(binary_logits, dim=1)[0, 1]
                scores.append(float(score.cpu()))
        return scores, longest


def build_row(
    prior: dict[str, Any],
    query: dict[str, Any],
    scores: list[float],
    seconds: float,
    longest: int,
) -> dict[str, Any]:
    baseline = prior["baseline"]["ranking"][:RANKING_DEPTH]
    head = baseline[:CANDIDATE_COUNT]
    tail = baseline[CANDIDATE_COUNT:]
    if len(scores) != CANDIDATE_COUNT or not np.isfinite(scores).all():
        raise ValueError(f"invalid scores for {query['question_id']}")
    if not all(0.0 <= score <= 1.0 for score in scores):
        raise ValueError(f"out-of-range scores for {query['question_id']}")
    scored = [
        {
            "document_id": candidate["document_id"],
            "original_rank": original_rank,
            "pplx_score": candidate["pplx_score"],
            "representative_chunk_id": candidate["representative_chunk_id"],
            "qwen_bf16_relevance_score": score,
        }
        for original_rank, (candidate, score) in enumerate(
            zip(head, scores, strict=True), 1
        )
    ]
    scored.sort(
        key=lambda candidate: (
            -candidate["qwen_bf16_relevance_score"],
            candidate["original_rank"],
        )
    )
    ranking = scored + [
        {
            "document_id": candidate["document_id"],
            "original_rank": original_rank,
            "pplx_score": candidate["pplx_score"],
            "representative_chunk_id": candidate["representative_chunk_id"],
            "qwen_bf16_relevance_score": None,
        }
        for original_rank, candidate in enumerate(tail, CANDIDATE_COUNT + 1)
    ]
    for rank, candidate in enumerate(ranking, 1):
        candidate["rank"] = rank
    reference_id = query["reference_document_id"]
    row = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "revision": REVISION,
        "question_id": query["question_id"],
        "reference_document_id": reference_id,
        "input_fingerprint_sha256": input_fingerprint(prior, query),
        "reference_ranks": {
            "pplx_baseline": reference_rank(reference_id, baseline),
            "qwen_4b_bf16_top_20": reference_rank(reference_id, ranking),
        },
        "qwen_bf16_scoring_seconds": seconds,
        "longest_input_tokens": longest,
        "top_20_document_set_unchanged": True,
        "rank_21_50_order_unchanged": True,
        "scores_valid": True,
        "pplx_baseline": baseline,
        "qwen_4b_bf16_top_20": ranking,
    }
    validate_row(row, prior, query)
    return row


def validate_row(row: dict[str, Any], prior: dict[str, Any], query: dict[str, Any]) -> None:
    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("model") != MODEL_NAME
        or row.get("revision") != REVISION
        or row.get("question_id") != query["question_id"]
        or row.get("reference_document_id") != query["reference_document_id"]
        or row.get("input_fingerprint_sha256") != input_fingerprint(prior, query)
        or row.get("scores_valid") is not True
    ):
        raise ValueError(f"incompatible row for {query['question_id']}")
    baseline = prior["baseline"]["ranking"][:RANKING_DEPTH]
    ranking = row.get("qwen_4b_bf16_top_20")
    if not isinstance(ranking, list) or len(ranking) != RANKING_DEPTH:
        raise ValueError(f"incomplete row for {query['question_id']}")
    if {item["document_id"] for item in baseline[:20]} != {
        item["document_id"] for item in ranking[:20]
    }:
        raise ValueError(f"Top-20 set changed for {query['question_id']}")
    if [item["document_id"] for item in baseline[20:]] != [
        item["document_id"] for item in ranking[20:]
    ]:
        raise ValueError(f"rank 21-50 changed for {query['question_id']}")


def metric_summary(ranks: Sequence[int | None]) -> dict[str, Any]:
    result: dict[str, Any] = {"query_count": len(ranks)}
    for cutoff in (1, 5, 10):
        hits = sum(rank is not None and rank <= cutoff for rank in ranks)
        result[f"hits_at_{cutoff}"] = hits
        result[f"recall_at_{cutoff}"] = hits / len(ranks)
    result["mrr_at_10"] = sum(
        1.0 / rank if rank is not None and rank <= 10 else 0.0 for rank in ranks
    ) / len(ranks)
    return result


def write_comparison(metrics: dict[str, dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = COMPARISON.with_name(f".{COMPARISON.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["system", "hits_at_1", "recall_at_1", "hits_at_5", "recall_at_5", "hits_at_10", "recall_at_10", "mrr_at_10"]
        )
        for name in ("pplx_baseline", "qwen_4b_gguf_top_20", "qwen_4b_bf16_top_20"):
            row = metrics[name]
            writer.writerow(
                [
                    name,
                    row["hits_at_1"],
                    row["recall_at_1"],
                    row["hits_at_5"],
                    row["recall_at_5"],
                    row["hits_at_10"],
                    row["recall_at_10"],
                    row["mrr_at_10"],
                ]
            )
    os.replace(temporary, COMPARISON)


def main() -> int:
    wall_started = time.perf_counter()
    prior_rows, queries = load_inputs()
    passages = load_passages(prior_rows)
    gguf_rows = {row["question_id"]: row for row in jsonl_rows(GGUF_RANKINGS)}
    if len(gguf_rows) != EXPECTED_QUERIES:
        raise ValueError("expected 450 completed GGUF rows")

    print(
        f"loading {MODEL_NAME}@{REVISION} as BF16 with batch_size={BATCH_SIZE} "
        f"gpu_budget={GPU_MODEL_BUDGET} cpu_budget={CPU_MODEL_BUDGET}",
        flush=True,
    )
    reranker = BF16Reranker()
    print(
        "device_map=" + json.dumps(reranker.device_map, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    output: list[dict[str, Any]] = []
    resumed = 0
    for query_index, prior in enumerate(prior_rows, 1):
        question_id = prior["question_id"]
        query = queries[question_id]
        path = checkpoint_path(question_id)
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            validate_row(row, prior, query)
            output.append(row)
            resumed += 1
            print(f"reused {query_index}/{EXPECTED_QUERIES}: {question_id}", flush=True)
            continue

        head = prior["baseline"]["ranking"][:CANDIDATE_COUNT]
        documents = [passages[item["representative_chunk_id"]] for item in head]
        torch.cuda.synchronize()
        started = time.perf_counter()
        scores, longest = reranker.score(query["query"], documents)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        row = build_row(prior, query, scores, seconds, longest)
        atomic_json(path, row)
        output.append(row)
        print(
            f"scored {query_index}/{EXPECTED_QUERIES}: {question_id} "
            f"seconds={seconds:.3f}",
            flush=True,
        )
        if query_index == 1:
            atomic_json(
                SMOKE,
                {
                    "status": "passed",
                    "question_id": question_id,
                    "pair_count": CANDIDATE_COUNT,
                    "scores_valid": True,
                    "top_20_document_set_unchanged": True,
                    "reference_ranks": row["reference_ranks"],
                    "seconds": seconds,
                    "longest_input_tokens": longest,
                    "peak_gpu_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
                    "peak_gpu_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
                },
            )
            print("one-query smoke test passed; continuing full TRAIN", flush=True)

    if len(output) != EXPECTED_QUERIES:
        raise ValueError(f"incomplete output: {len(output)}")

    pplx_ranks = [row["reference_ranks"]["pplx_baseline"] for row in output]
    bf16_ranks = [row["reference_ranks"]["qwen_4b_bf16_top_20"] for row in output]
    gguf_ranks = [
        gguf_rows[row["question_id"]]["reference_ranks"]["qwen_4b_top_20"]
        for row in output
    ]
    metrics = {
        "pplx_baseline": metric_summary(pplx_ranks),
        "qwen_4b_gguf_top_20": metric_summary(gguf_ranks),
        "qwen_4b_bf16_top_20": metric_summary(bf16_ranks),
    }
    if metrics["pplx_baseline"]["hits_at_5"] != 335:
        raise ValueError("PPLX baseline changed from 335/450")
    if metrics["qwen_4b_gguf_top_20"]["hits_at_5"] != 316:
        raise ValueError("GGUF comparator changed from 316/450")
    new_into_top_5 = sum(
        (before is None or before > 5) and after is not None and after <= 5
        for before, after in zip(pplx_ranks, bf16_ranks, strict=True)
    )
    dropped_out_of_top_5 = sum(
        before is not None and before <= 5 and (after is None or after > 5)
        for before, after in zip(pplx_ranks, bf16_ranks, strict=True)
    )
    timings = [float(row["qwen_bf16_scoring_seconds"]) for row in output]
    summary = {
        "status": "complete",
        "evaluation_split": "TRAIN only",
        "query_count": EXPECTED_QUERIES,
        "pair_count": EXPECTED_QUERIES * CANDIDATE_COUNT,
        "model": MODEL_NAME,
        "revision": REVISION,
        "model_snapshot": str(MODEL_SNAPSHOT),
        "dtype": "bfloat16",
        "batch_size": BATCH_SIZE,
        "device_map": reranker.device_map,
        "gpu_model_budget": GPU_MODEL_BUDGET,
        "cpu_model_budget": CPU_MODEL_BUDGET,
        "instruction": INSTRUCTION,
        "score_method": "softmax(no_logit, yes_logit)[yes]",
        "all_top_20_document_sets_unchanged": True,
        "all_rank_21_50_orders_unchanged": True,
        "all_scores_valid": True,
        "metrics": metrics,
        "qwen_4b_bf16_vs_pplx_top_5": {
            "new_into_top_5": new_into_top_5,
            "dropped_out_of_top_5": dropped_out_of_top_5,
            "net_change": new_into_top_5 - dropped_out_of_top_5,
        },
        "rerank_timing": timing_summary(timings),
        "model_load_seconds": reranker.load_seconds,
        "experiment_wall_seconds": time.perf_counter() - wall_started,
        "checkpoint_resume_count": resumed,
        "peak_gpu_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_gpu_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "process_peak_ram_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2),
    }
    atomic_jsonl(RANKINGS, output)
    atomic_json(SUMMARY, summary)
    write_comparison(metrics)
    print("QWEN4B_BF16_RESULT=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
