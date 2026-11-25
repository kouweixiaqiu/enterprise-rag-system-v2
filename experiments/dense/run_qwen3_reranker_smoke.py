#!/usr/bin/env python3
"""Offline 4-query Qwen3 reranker smoke test over saved PPLX top-20 candidates."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import fixed_chunks, jsonl_rows  # noqa: E402


MODEL_SNAPSHOT = Path(
    "/home/fuxuanyi/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-Reranker-0.6B/snapshots/"
    "e61197ed45024b0ed8a2d74b80b4d909f1255473"
)
BGE_RERANK_ROWS = (
    PROJECT_ROOT
    / "experiments/dense/results/pplx_bge_reranker_fixed_600_0_train/rankings.jsonl"
)
TRAIN_QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/dense/results/qwen3_reranker_0.6b_smoke_top20_train"
)
QUERY_COUNT = 4
CANDIDATE_COUNT = 20
BATCH_SIZE = 4
MAX_LENGTH = 2048
INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on '
    'the Query and the Instruct provided. Note that the answer can only be "yes" or '
    '"no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    os.replace(temporary, path)


def format_instruction(query: str, document: str) -> str:
    return (
        f"<Instruct>: {INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


def load_inputs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    prior_rows = list(jsonl_rows(BGE_RERANK_ROWS))[:QUERY_COUNT]
    queries = {row["question_id"]: row for row in jsonl_rows(TRAIN_QUERIES)}
    if len(prior_rows) != QUERY_COUNT:
        raise ValueError("expected four prior BGE reranker rows")
    for row in prior_rows:
        query = queries.get(row["question_id"])
        if query is None or query["reference_document_id"] != row["reference_document_id"]:
            raise ValueError(f"TRAIN query mismatch for {row['question_id']}")
        candidates = row["baseline"]["ranking"][:CANDIDATE_COUNT]
        if len(candidates) != CANDIDATE_COUNT:
            raise ValueError(f"not enough candidates for {row['question_id']}")
        ids = [candidate["document_id"] for candidate in candidates]
        if len(set(ids)) != CANDIDATE_COUNT:
            raise ValueError(f"duplicate candidate document for {row['question_id']}")
    return prior_rows, queries


def load_passages(prior_rows: Sequence[dict[str, Any]]) -> dict[str, str]:
    needed = {
        candidate["representative_chunk_id"]: candidate["document_id"]
        for row in prior_rows
        for candidate in row["baseline"]["ranking"][:CANDIDATE_COUNT]
    }
    needed_documents = set(needed.values())
    passages: dict[str, str] = {}
    for document in jsonl_rows(CORPUS):
        document_id = document["document_id"]
        if document_id not in needed_documents:
            continue
        for chunk_index, body in enumerate(fixed_chunks(document["text"], 600, 0)):
            chunk_id = f"{document_id}::{chunk_index:06d}"
            if chunk_id in needed:
                if needed[chunk_id] != document_id:
                    raise ValueError(f"representative/document mismatch: {chunk_id}")
                passages[chunk_id] = f"{document['title']}\n\n{body}"
    missing = sorted(set(needed) - set(passages))
    if missing:
        raise ValueError(f"missing representative passages: {missing[:5]}")
    return passages


class QwenReranker:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        required = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
        missing = [name for name in required if not (MODEL_SNAPSHOT / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete local Qwen cache: {missing}")
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
        ).to("cuda").eval()
        self.load_seconds = time.perf_counter() - load_started
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
            for start in range(0, len(documents), BATCH_SIZE):
                pairs = [
                    format_instruction(query, document)
                    for document in documents[start : start + BATCH_SIZE]
                ]
                encoded = self.tokenizer(
                    pairs,
                    padding=False,
                    truncation="longest_first",
                    return_attention_mask=False,
                    max_length=MAX_LENGTH - len(self.prefix_tokens) - len(self.suffix_tokens),
                )
                for index, token_ids in enumerate(encoded["input_ids"]):
                    encoded["input_ids"][index] = (
                        self.prefix_tokens + token_ids + self.suffix_tokens
                    )
                    longest = max(longest, len(encoded["input_ids"][index]))
                batch = self.tokenizer.pad(
                    encoded,
                    padding=True,
                    return_tensors="pt",
                    max_length=MAX_LENGTH,
                )
                batch = {key: value.to("cuda") for key, value in batch.items()}
                logits = self.model(**batch).logits[:, -1, :]
                yes_logits = logits[:, self.true_token_id]
                no_logits = logits[:, self.false_token_id]
                binary_logits = torch.stack([no_logits, yes_logits], dim=1)
                probabilities = torch.nn.functional.log_softmax(binary_logits, dim=1)[:, 1].exp()
                scores.extend(probabilities.to(torch.float32).cpu().tolist())
        return scores, longest


def main() -> int:
    wall_started = time.perf_counter()
    prior_rows, queries = load_inputs()
    passages = load_passages(prior_rows)
    print(
        f"offline Qwen smoke: queries={QUERY_COUNT} pairs={QUERY_COUNT * CANDIDATE_COUNT} "
        f"batch_size={BATCH_SIZE}",
        flush=True,
    )
    reranker = QwenReranker()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    scoring_started = time.perf_counter()

    output_rows: list[dict[str, Any]] = []
    longest_input = 0
    for prior in prior_rows:
        query = queries[prior["question_id"]]
        candidates = prior["baseline"]["ranking"][:CANDIDATE_COUNT]
        documents = [passages[row["representative_chunk_id"]] for row in candidates]
        query_started = time.perf_counter()
        scores, query_longest = reranker.score(query["query"], documents)
        torch.cuda.synchronize()
        query_seconds = time.perf_counter() - query_started
        longest_input = max(longest_input, query_longest)
        if len(scores) != CANDIDATE_COUNT or not np.isfinite(scores).all():
            raise ValueError(f"invalid scores for {query['question_id']}")
        if not all(0.0 <= score <= 1.0 for score in scores):
            raise ValueError(f"out-of-range scores for {query['question_id']}")

        scored = [
            {
                "document_id": candidate["document_id"],
                "original_rank": original_rank,
                "representative_chunk_id": candidate["representative_chunk_id"],
                "qwen_relevance_score": float(score),
            }
            for original_rank, (candidate, score) in enumerate(
                zip(candidates, scores, strict=True), 1
            )
        ]
        reranked = sorted(
            scored,
            key=lambda row: (-row["qwen_relevance_score"], row["original_rank"]),
        )
        for reranked_rank, row in enumerate(reranked, 1):
            row["reranked_rank"] = reranked_rank
        original_ids = {row["document_id"] for row in scored}
        reranked_ids = {row["document_id"] for row in reranked}
        if original_ids != reranked_ids or len(original_ids) != CANDIDATE_COUNT:
            raise ValueError(f"top-20 candidate set changed for {query['question_id']}")

        reference_id = query["reference_document_id"]
        original_reference_rank = prior["baseline"]["reference_document_rank"]
        reranked_reference_rank = next(
            (
                row["reranked_rank"]
                for row in reranked
                if row["document_id"] == reference_id
            ),
            original_reference_rank,
        )
        output_rows.append(
            {
                "question_id": query["question_id"],
                "reference_document_id": reference_id,
                "reference_rank_before": original_reference_rank,
                "reference_rank_after": reranked_reference_rank,
                "scoring_seconds": query_seconds,
                "top_20_document_set_unchanged": True,
                "scores_valid": True,
                "reranked_top_20": reranked,
            }
        )
        print(
            f"{query['question_id']}: reference {original_reference_rank} -> "
            f"{reranked_reference_rank}, seconds={query_seconds:.3f}",
            flush=True,
        )

    torch.cuda.synchronize()
    scoring_seconds = time.perf_counter() - scoring_started
    summary = {
        "status": "passed",
        "mode": "offline_smoke_test",
        "evaluation_split": "TRAIN only",
        "model": "Qwen/Qwen3-Reranker-0.6B",
        "revision": MODEL_SNAPSHOT.name,
        "model_snapshot": str(MODEL_SNAPSHOT),
        "query_count": QUERY_COUNT,
        "candidate_documents_per_query": CANDIDATE_COUNT,
        "pair_count": QUERY_COUNT * CANDIDATE_COUNT,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "longest_input_tokens": longest_input,
        "instruction": INSTRUCTION,
        "score_method": "softmax(no_logit, yes_logit)[yes]",
        "all_top_20_document_sets_unchanged": True,
        "all_scores_valid": True,
        "model_load_seconds": reranker.load_seconds,
        "scoring_seconds": scoring_seconds,
        "wall_seconds": time.perf_counter() - wall_started,
        "peak_gpu_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_gpu_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
        "reference_rank_changes": [
            {
                "question_id": row["question_id"],
                "before": row["reference_rank_before"],
                "after": row["reference_rank_after"],
            }
            for row in output_rows
        ],
    }
    atomic_jsonl(OUTPUT_DIR / "rankings.jsonl", output_rows)
    atomic_json(OUTPUT_DIR / "summary.json", summary)
    print("QWEN_SMOKE_RESULT=" + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
