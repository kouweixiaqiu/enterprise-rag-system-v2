#!/usr/bin/env python3
"""Rerank seven saved TRAIN Top-50 document chains with one Qwen3 model load."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import (  # noqa: E402
    chunks_for,
    jsonl_rows,
)
from experiments.chunk.run_bm25_parameter_grid import CONFIGS  # noqa: E402
from experiments.dense.run_qwen3_reranker_smoke import (  # noqa: E402
    INSTRUCTION,
    PREFIX,
    SUFFIX,
    format_instruction,
)


EXPECTED_QUERIES = 450
CANDIDATE_COUNT = 50
EVALUATION_CUTOFFS = (1, 5, 10, 20)
SCHEMA_VERSION = 1
TRAIN_QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
DEFAULT_MODEL_SNAPSHOT = Path(
    "/home/fuxuanyi/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-Reranker-4B/snapshots/"
    "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/dense/results/qwen3_reranker_4b_bf16_top50_seven_chain_train"
)


@dataclass(frozen=True)
class Chain:
    name: str
    chunk_config: str
    source: str
    rankings_path: Path
    description: str


RRF_ROOT = PROJECT_ROOT / "experiments/hybrid/results/pplx_bm25_rrf_grid_train"
PPLX_ROOT = (
    PROJECT_ROOT
    / "experiments/dense/results/pplx_embed_v1_0.6b_six_chunk_train"
)
CHAINS = (
    Chain(
        "fixed_1200_0_rrf",
        "fixed_1200_0",
        "rrf",
        RRF_ROOT
        / "fixed_1200_0/k1_1.6_b_0.5/"
        "bm25_d100_pplx_d100_pplxw3_bm25w1_k60/rrf_rankings.jsonl",
        "k1=1.6,b=0.5,B100/P100,PPLX:BM25=3:1,k=60",
    ),
    Chain(
        "fixed_1200_300_rrf",
        "fixed_1200_300",
        "rrf",
        RRF_ROOT
        / "fixed_1200_300/k1_1.2_b_1/"
        "bm25_d100_pplx_d100_pplxw3_bm25w1_k60/rrf_rankings.jsonl",
        "k1=1.2,b=1.0,B100/P100,PPLX:BM25=3:1,k=60",
    ),
    Chain(
        "fixed_600_0_rrf",
        "fixed_600_0",
        "rrf",
        RRF_ROOT
        / "fixed_600_0/k1_1.6_b_1/"
        "bm25_d50_pplx_d100_pplxw3_bm25w1_k60/rrf_rankings.jsonl",
        "k1=1.6,b=1.0,B50/P100,PPLX:BM25=3:1,k=60",
    ),
    Chain(
        "fixed_600_150_rrf",
        "fixed_600_150",
        "rrf",
        RRF_ROOT
        / "fixed_600_150/k1_1.6_b_1/"
        "bm25_d100_pplx_d100_pplxw3_bm25w1_k60/rrf_rankings.jsonl",
        "k1=1.6,b=1.0,B100/P100,PPLX:BM25=3:1,k=60",
    ),
    Chain(
        "paragraph_merge_1200_0_rrf",
        "paragraph_merge_1200_0",
        "rrf",
        RRF_ROOT
        / "paragraph_merge_1200_0/k1_1.6_b_1/"
        "bm25_d50_pplx_d100_pplxw2_bm25w1_k60/rrf_rankings.jsonl",
        "k1=1.6,b=1.0,B50/P100,PPLX:BM25=2:1,k=60",
    ),
    Chain(
        "paragraph_merge_600_0_rrf",
        "paragraph_merge_600_0",
        "rrf",
        RRF_ROOT
        / "paragraph_merge_600_0/k1_1.2_b_0.5/"
        "bm25_d100_pplx_d100_pplxw3_bm25w1_k60/rrf_rankings.jsonl",
        "k1=1.2,b=0.5,B100/P100,PPLX:BM25=3:1,k=60",
    ),
    Chain(
        "fixed_600_0_pplx",
        "fixed_600_0",
        "pplx",
        PPLX_ROOT / "fixed_600_0/dense_rankings_top100.jsonl",
        "PPLX original ranking",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use 'auto' for the established 4B GPU/CPU offload, or a torch device.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gpu-memory-budget", default="4800MiB")
    parser.add_argument("--cpu-memory-budget", default="5000MiB")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help="Validation only: score the first N aligned TRAIN questions; 0 uses all 450.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Randomly sample this many TRAIN questions once and reuse them for all chains.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--chain",
        action="append",
        choices=tuple(chain.name for chain in CHAINS),
        help="Run only this chain; repeat for multiple chains. Default: all seven.",
    )
    parser.add_argument(
        "--reuse-checkpoints-from",
        type=Path,
        help=(
            "Existing single-chain result directory whose checkpoints are copied into "
            "the new output before scoring. Requires exactly one --chain."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate selected inputs and save the question list without loading the model.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_length <= 0:
        parser.error("--max-length must be positive")
    if args.query_limit < 0 or args.query_limit > EXPECTED_QUERIES:
        parser.error(f"--query-limit must be between 0 and {EXPECTED_QUERIES}")
    if args.sample_size < 0 or args.sample_size > EXPECTED_QUERIES:
        parser.error(f"--sample-size must be between 0 and {EXPECTED_QUERIES}")
    if args.query_limit and args.sample_size:
        parser.error("--query-limit and --sample-size cannot be used together")
    if args.reuse_checkpoints_from and (not args.chain or len(args.chain) != 1):
        parser.error("--reuse-checkpoints-from requires exactly one --chain")
    return args


def selected_chains(args: argparse.Namespace) -> tuple[Chain, ...]:
    if not args.chain:
        return CHAINS
    requested = set(args.chain)
    return tuple(chain for chain in CHAINS if chain.name in requested)


def stage_reused_checkpoints(
    args: argparse.Namespace,
    chains: Sequence[Chain],
) -> dict[str, set[str]]:
    reused = {chain.name: set() for chain in chains}
    if args.reuse_checkpoints_from is None:
        return reused
    chain = chains[0]
    source_dir = args.reuse_checkpoints_from.resolve() / "checkpoints"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing reused checkpoint directory: {source_dir}")
    source_files = sorted(source_dir.glob("TRAIN_Q*.json"))
    if len(source_files) != 30:
        raise ValueError(
            f"expected exactly 30 reusable checkpoints in {source_dir}, found {len(source_files)}"
        )
    target_dir = args.output_dir / chain.name / "checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    for source in source_files:
        row = json.loads(source.read_text(encoding="utf-8"))
        question_id = row.get("question_id")
        if question_id != source.stem or row.get("chain") != chain.name:
            raise ValueError(f"invalid reusable checkpoint: {source}")
        target = target_dir / source.name
        if target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing != row:
                raise ValueError(f"existing checkpoint differs from reusable source: {target}")
        else:
            shutil.copy2(source, target)
        reused[chain.name].add(question_id)
    return reused


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def __call__(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as output:
            output.write(line + "\n")


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def atomic_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, path)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def reference_rank(reference_id: str, documents: Sequence[dict[str, Any]]) -> int | None:
    return next(
        (index for index, row in enumerate(documents, 1) if row["document_id"] == reference_id),
        None,
    )


def candidate_chunk_id(candidate: dict[str, Any], source: str) -> str:
    if source == "pplx":
        value = candidate.get("representative_chunk_id")
    else:
        value = candidate.get("pplx_representative_chunk_id") or candidate.get(
            "bm25_representative_chunk_id"
        )
    if not isinstance(value, str) or not value:
        raise ValueError(f"candidate has no usable representative chunk: {candidate}")
    return value


def source_documents(row: dict[str, Any], source: str) -> list[dict[str, Any]]:
    key = "top_documents" if source == "pplx" else "fused_documents"
    documents = row.get(key)
    if not isinstance(documents, list):
        raise ValueError(f"{row.get('question_id')} has no {key} list")
    return documents


def load_queries() -> tuple[list[str], dict[str, dict[str, Any]]]:
    rows = list(jsonl_rows(TRAIN_QUERIES))
    by_id = {row.get("question_id"): row for row in rows}
    if len(rows) != EXPECTED_QUERIES or len(by_id) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} unique TRAIN questions")
    if not all(
        isinstance(row.get("query"), str)
        and isinstance(row.get("reference_document_id"), str)
        for row in rows
    ):
        raise ValueError("TRAIN query file has invalid query/reference fields")
    return [row["question_id"] for row in rows], by_id


def select_question_ids(
    args: argparse.Namespace,
    question_order: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    if args.sample_size:
        question_ids = random.Random(args.seed).sample(
            list(question_order), args.sample_size
        )
        mode = "seeded_random_sample"
    elif args.query_limit:
        question_ids = list(question_order[: args.query_limit])
        mode = "first_n_validation"
    else:
        question_ids = list(question_order)
        mode = "all_train"
    manifest = {
        "mode": mode,
        "source_query_count": len(question_order),
        "sample_size": len(question_ids),
        "seed": args.seed if args.sample_size else None,
        "question_ids": question_ids,
    }
    path = args.output_dir / "sample_question_ids.json"
    if path.is_file():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError(
                f"saved sample list differs from current selection: {path}"
            )
    else:
        atomic_json(path, manifest)
    return question_ids, manifest


def load_and_validate_ranking(
    chain: Chain,
    question_order: Sequence[str],
    queries: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not chain.rankings_path.is_file():
        raise FileNotFoundError(f"missing ranking for {chain.name}: {chain.rankings_path}")
    rows = list(jsonl_rows(chain.rankings_path))
    by_id = {row.get("question_id"): row for row in rows}
    if len(rows) != EXPECTED_QUERIES or len(by_id) != EXPECTED_QUERIES:
        raise ValueError(f"{chain.name}: expected 450 unique ranking rows")
    if set(by_id) != set(question_order):
        raise ValueError(f"{chain.name}: ranking question IDs do not match TRAIN")
    for question_id in question_order:
        row = by_id[question_id]
        if row.get("reference_document_id") != queries[question_id]["reference_document_id"]:
            raise ValueError(f"{chain.name}: reference mismatch for {question_id}")
        candidates = source_documents(row, chain.source)[:CANDIDATE_COUNT]
        if len(candidates) != CANDIDATE_COUNT:
            raise ValueError(f"{chain.name}: {question_id} has fewer than 50 candidates")
        document_ids = [candidate.get("document_id") for candidate in candidates]
        if not all(isinstance(value, str) and value for value in document_ids):
            raise ValueError(f"{chain.name}: invalid document ID for {question_id}")
        if len(set(document_ids)) != CANDIDATE_COUNT:
            raise ValueError(f"{chain.name}: duplicate Top-50 document for {question_id}")
        for candidate in candidates:
            chunk_id = candidate_chunk_id(candidate, chain.source)
            if not chunk_id.startswith(candidate["document_id"] + "::"):
                raise ValueError(
                    f"{chain.name}: representative/document mismatch for {question_id}: "
                    f"{chunk_id} vs {candidate['document_id']}"
                )
    return by_id


def needed_chunks(
    chain: Chain,
    ranking: dict[str, dict[str, Any]],
    question_ids: Sequence[str],
) -> set[str]:
    return {
        candidate_chunk_id(candidate, chain.source)
        for question_id in question_ids
        for candidate in source_documents(ranking[question_id], chain.source)[:CANDIDATE_COUNT]
    }


def load_passages(chunk_config: str, needed: set[str]) -> dict[str, str]:
    config = CONFIGS[chunk_config]
    passages: dict[str, str] = {}
    needed_documents = {chunk_id.split("::", 1)[0] for chunk_id in needed}
    for document in jsonl_rows(CORPUS):
        document_id = document.get("document_id")
        if document_id not in needed_documents:
            continue
        title = document.get("title")
        text = document.get("text")
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError(f"invalid corpus text for {document_id}")
        for index, body in enumerate(chunks_for(text, config)):
            chunk_id = f"{document_id}::{index:06d}"
            if chunk_id in needed:
                passages[chunk_id] = f"{title}\n\n{body}"
    missing = sorted(needed - set(passages))
    if missing:
        raise ValueError(
            f"{chunk_config}: {len(missing)} representative chunks cannot be rebuilt; "
            f"first={missing[:3]}"
        )
    return passages


class QwenReranker:
    def __init__(self, args: argparse.Namespace) -> None:
        if (args.device == "auto" or args.device.startswith("cuda")) and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is unavailable for --device {args.device}")
        required = ("config.json", "tokenizer.json", "tokenizer_config.json")
        missing = [name for name in required if not (args.model_snapshot / name).is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete local model snapshot: {missing}")
        if not any(args.model_snapshot.glob("*.safetensors")):
            raise FileNotFoundError(
                f"local model snapshot has no safetensors weights: {args.model_snapshot}"
            )
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_snapshot,
            local_files_only=True,
            trust_remote_code=False,
            padding_side="left",
        )
        load_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": torch_dtype(args.dtype),
            "attn_implementation": args.attn_implementation,
        }
        if args.device == "auto":
            load_kwargs.update(
                {
                    "device_map": "auto",
                    "max_memory": {0: args.gpu_memory_budget, "cpu": args.cpu_memory_budget},
                    "low_cpu_mem_usage": True,
                    "use_safetensors": True,
                }
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_snapshot, **load_kwargs
            ).eval()
            self.device_map = dict(self.model.hf_device_map)
            if "disk" in set(self.device_map.values()):
                raise RuntimeError("unexpected disk offload; GPU/CPU offload is required")
            self.input_device = self.model.get_input_embeddings().weight.device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                args.model_snapshot, **load_kwargs
            ).to(args.device).eval()
            self.device_map = {"model": args.device}
            self.input_device = torch.device(args.device)
        self.load_seconds = time.perf_counter() - started
        self.device = args.device
        self.batch_size = args.batch_size
        self.max_length = args.max_length
        self.false_token_id = self.tokenizer.convert_tokens_to_ids("no")
        self.true_token_id = self.tokenizer.convert_tokens_to_ids("yes")
        if self.false_token_id == self.tokenizer.unk_token_id or self.true_token_id == self.tokenizer.unk_token_id:
            raise ValueError("model tokenizer does not provide standalone no/yes tokens")
        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        if len(self.prefix_tokens) + len(self.suffix_tokens) >= self.max_length:
            raise ValueError("--max-length is too small for the Qwen reranker template")

    def score(self, query: str, documents: Sequence[str]) -> tuple[list[float], int]:
        scores: list[float] = []
        longest = 0
        middle_limit = self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        with torch.inference_mode():
            for start in range(0, len(documents), self.batch_size):
                pairs = [
                    format_instruction(query, document)
                    for document in documents[start : start + self.batch_size]
                ]
                encoded = self.tokenizer(
                    pairs,
                    padding=False,
                    truncation="longest_first",
                    return_attention_mask=False,
                    max_length=middle_limit,
                )
                for index, token_ids in enumerate(encoded["input_ids"]):
                    encoded["input_ids"][index] = (
                        self.prefix_tokens + token_ids + self.suffix_tokens
                    )
                    longest = max(longest, len(encoded["input_ids"][index]))
                batch = self.tokenizer.pad(encoded, padding=True, return_tensors="pt")
                batch = {key: value.to(self.input_device) for key, value in batch.items()}
                logits = self.model(
                    **batch,
                    use_cache=False,
                    logits_to_keep=1,
                ).logits[:, -1, :]
                binary_logits = torch.stack(
                    [logits[:, self.false_token_id], logits[:, self.true_token_id]],
                    dim=1,
                ).to(torch.float32)
                probabilities = torch.softmax(binary_logits, dim=1)[:, 1]
                scores.extend(probabilities.cpu().tolist())
        return scores, longest

    def synchronize(self) -> None:
        if torch.cuda.is_available() and any(
            value == 0 or str(value).startswith("cuda")
            for value in self.device_map.values()
        ):
            torch.cuda.synchronize()


def input_fingerprint(
    chain: Chain,
    query: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    value = {
        "schema_version": SCHEMA_VERSION,
        "chain": asdict(chain),
        "question_id": query["question_id"],
        "query": query["query"],
        "model_snapshot": str(args.model_snapshot.resolve()),
        "dtype": args.dtype,
        "max_length": args.max_length,
        "instruction": INSTRUCTION,
        "candidates": [
            [candidate["document_id"], candidate_chunk_id(candidate, chain.source)]
            for candidate in candidates
        ],
    }
    value["chain"]["rankings_path"] = str(chain.rankings_path.resolve())
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_checkpoint(
    row: dict[str, Any],
    fingerprint: str,
    candidate_ids: set[str],
) -> None:
    ranking = row.get("reranked_documents")
    if (
        row.get("schema_version") != SCHEMA_VERSION
        or row.get("input_fingerprint_sha256") != fingerprint
        or not isinstance(ranking, list)
        or len(ranking) != CANDIDATE_COUNT
        or {candidate.get("document_id") for candidate in ranking} != candidate_ids
        or row.get("scores_valid") is not True
    ):
        raise ValueError(f"incompatible checkpoint for {row.get('question_id')}")


def score_question(
    reranker: QwenReranker,
    chain: Chain,
    source_row: dict[str, Any],
    query: dict[str, Any],
    passages: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates = source_documents(source_row, chain.source)[:CANDIDATE_COUNT]
    chunk_ids = [candidate_chunk_id(candidate, chain.source) for candidate in candidates]
    documents = [passages[chunk_id] for chunk_id in chunk_ids]
    fingerprint = input_fingerprint(chain, query, candidates, args)
    reranker.synchronize()
    started = time.perf_counter()
    scores, longest = reranker.score(query["query"], documents)
    reranker.synchronize()
    seconds = time.perf_counter() - started
    if len(scores) != CANDIDATE_COUNT or not np.isfinite(scores).all():
        raise ValueError(f"invalid scores for {chain.name}/{query['question_id']}")
    if not all(0.0 <= score <= 1.0 for score in scores):
        raise ValueError(f"out-of-range scores for {chain.name}/{query['question_id']}")
    ranking = []
    for original_rank, (candidate, chunk_id, score) in enumerate(
        zip(candidates, chunk_ids, scores, strict=True), 1
    ):
        ranking.append(
            {
                "document_id": candidate["document_id"],
                "original_rank": original_rank,
                "scored_chunk_id": chunk_id,
                "rerank_score": float(score),
            }
        )
    ranking.sort(key=lambda row: (-row["rerank_score"], row["original_rank"]))
    for rank, candidate in enumerate(ranking, 1):
        candidate["rank"] = rank
    before_rank = reference_rank(query["reference_document_id"], candidates)
    after_rank = reference_rank(query["reference_document_id"], ranking)
    row = {
        "schema_version": SCHEMA_VERSION,
        "chain": chain.name,
        "chunk_config": chain.chunk_config,
        "ranking_source": chain.source,
        "question_id": query["question_id"],
        "reference_document_id": query["reference_document_id"],
        "input_fingerprint_sha256": fingerprint,
        "reference_rank_before": before_rank,
        "reference_rank_after": after_rank,
        "scoring_seconds": seconds,
        "longest_input_tokens": longest,
        "candidate_set_unchanged": True,
        "scores_valid": True,
        "reranked_documents": ranking,
    }
    validate_checkpoint(row, fingerprint, {candidate["document_id"] for candidate in candidates})
    return row


def metric_summary(ranks: Sequence[int | None]) -> dict[str, Any]:
    output: dict[str, Any] = {"query_count": len(ranks)}
    for cutoff in EVALUATION_CUTOFFS:
        hits = sum(rank is not None and rank <= cutoff for rank in ranks)
        output[f"hits_at_{cutoff}"] = hits
        output[f"recall_at_{cutoff}"] = hits / len(ranks)
    return output


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "total_seconds": float(sum(values)),
        "mean_seconds_per_query": float(np.mean(values)),
        "p95_seconds_per_query": float(ordered[p95_index]),
    }


def summarize_chain(chain: Chain, rows: Sequence[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    before = [row["reference_rank_before"] for row in rows]
    after = [row["reference_rank_after"] for row in rows]
    new_ids = [
        row["question_id"]
        for row in rows
        if (row["reference_rank_before"] is None or row["reference_rank_before"] > 5)
        and row["reference_rank_after"] is not None
        and row["reference_rank_after"] <= 5
    ]
    dropped_ids = [
        row["question_id"]
        for row in rows
        if row["reference_rank_before"] is not None
        and row["reference_rank_before"] <= 5
        and (row["reference_rank_after"] is None or row["reference_rank_after"] > 5)
    ]
    return {
        "status": "complete",
        "chain": chain.name,
        "chunk_config": chain.chunk_config,
        "ranking_source": chain.source,
        "source_configuration": chain.description,
        "source_rankings": display_path(chain.rankings_path),
        "query_count": len(rows),
        "candidate_documents_per_query": CANDIDATE_COUNT,
        "metrics_before": metric_summary(before),
        "metrics_after": metric_summary(after),
        "top_5_change": {
            "new_hits": len(new_ids),
            "dropped_hits": len(dropped_ids),
            "net_change": len(new_ids) - len(dropped_ids),
            "new_question_ids": new_ids,
            "dropped_question_ids": dropped_ids,
        },
        "timing": timing_summary([float(row["scoring_seconds"]) for row in rows]),
        "wall_seconds": elapsed,
        "all_candidate_sets_unchanged": True,
        "all_scores_valid": True,
    }


def run_chain(
    chain_index: int,
    chain_count: int,
    chain: Chain,
    ranking: dict[str, dict[str, Any]],
    question_ids: Sequence[str],
    queries: dict[str, dict[str, Any]],
    reranker: QwenReranker,
    args: argparse.Namespace,
    logger: RunLogger,
    progress_every: int,
    base_reused_question_ids: set[str],
) -> dict[str, Any]:
    started = time.perf_counter()
    chain_dir = args.output_dir / chain.name
    checkpoints = chain_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    needed = needed_chunks(chain, ranking, question_ids)
    passages = load_passages(chain.chunk_config, needed)
    logger(
        f"[{chain_index}/{chain_count}] 开始 {chain.name}: questions={len(question_ids)}, "
        f"pairs={len(question_ids) * CANDIDATE_COUNT}, source={display_path(chain.rankings_path)}"
    )
    rows: list[dict[str, Any]] = []
    work_total = len(set(question_ids) - base_reused_question_ids)
    work_completed = 0
    for index, question_id in enumerate(question_ids, 1):
        source_row = ranking[question_id]
        query = queries[question_id]
        candidates = source_documents(source_row, chain.source)[:CANDIDATE_COUNT]
        fingerprint = input_fingerprint(chain, query, candidates, args)
        checkpoint = checkpoints / f"{question_id}.json"
        if checkpoint.is_file():
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
            validate_checkpoint(
                row,
                fingerprint,
                {candidate["document_id"] for candidate in candidates},
            )
        else:
            row = score_question(reranker, chain, source_row, query, passages, args)
            atomic_json(checkpoint, row)
        rows.append(row)
        if question_id not in base_reused_question_ids:
            work_completed += 1
            if work_completed % progress_every == 0 or work_completed == work_total:
                logger(
                    f"[{chain_index}/{chain_count}] {chain.name}: "
                    f"已完成 {work_completed}/{work_total}"
                )
    row_ids = [row.get("question_id") for row in rows]
    if len(rows) != len(question_ids) or len(set(row_ids)) != len(question_ids):
        raise ValueError(f"{chain.name}: merged rankings contain missing/duplicate questions")
    if set(row_ids) != set(question_ids):
        raise ValueError(f"{chain.name}: merged ranking question IDs do not match selection")
    for row in rows:
        documents = row.get("reranked_documents")
        if not isinstance(documents, list) or len(documents) != CANDIDATE_COUNT:
            raise ValueError(f"{chain.name}: invalid merged Top-50 for {row.get('question_id')}")
        if not all(isinstance(doc.get("scored_chunk_id"), str) for doc in documents):
            raise ValueError(f"{chain.name}: missing scored chunk ID for {row.get('question_id')}")
    summary = summarize_chain(chain, rows, time.perf_counter() - started)
    summary["reused_checkpoint_count"] = len(set(question_ids) & base_reused_question_ids)
    summary["supplement_question_count"] = work_total
    summary["merged_unique_question_count"] = len(set(row_ids))
    atomic_jsonl(chain_dir / "rankings.jsonl", rows)
    atomic_json(chain_dir / "summary.json", summary)
    logger(
        f"[{chain_index}/{chain_count}] 完成 {chain.name}: elapsed={summary['wall_seconds']:.1f}s, "
        f"merged_unique={summary['merged_unique_question_count']}, "
        f"output={display_path(chain_dir)}"
    )
    return summary


def fixed_600_comparison(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["chain"]: row for row in summaries}
    pplx = by_name["fixed_600_0_pplx"]
    rrf = by_name["fixed_600_0_rrf"]
    return {
        "query_count": pplx["query_count"],
        "systems": {
            "pplx_original": pplx["metrics_before"],
            "pplx_reranked": pplx["metrics_after"],
            "rrf_reranked": rrf["metrics_after"],
        },
    }


def write_summary_tables(
    output_dir: Path,
    summaries: Sequence[dict[str, Any]],
    comparison: dict[str, Any] | None,
) -> None:
    atomic_csv(
        output_dir / "summary.csv",
        (
            "chain",
            "chunk_config",
            "ranking_source",
            "query_count",
            "r5_before_hits",
            "r5_after_hits",
            "top5_new_hits",
            "top5_dropped_hits",
            "top5_net_change",
            "top5_new_question_ids",
            "top5_dropped_question_ids",
            "rankings_path",
        ),
        (
            (
                row["chain"],
                row["chunk_config"],
                row["ranking_source"],
                row["query_count"],
                row["metrics_before"]["hits_at_5"],
                row["metrics_after"]["hits_at_5"],
                row["top_5_change"]["new_hits"],
                row["top_5_change"]["dropped_hits"],
                row["top_5_change"]["net_change"],
                "|".join(row["top_5_change"]["new_question_ids"]),
                "|".join(row["top_5_change"]["dropped_question_ids"]),
                row["source_rankings"],
            )
            for row in summaries
        ),
    )
    if comparison is None:
        return
    by_name = {row["chain"]: row for row in summaries}
    pplx = by_name["fixed_600_0_pplx"]
    rrf = by_name["fixed_600_0_rrf"]
    atomic_csv(
        output_dir / "fixed_600_0_comparison.csv",
        ("system", "query_count", "r5_hits", "top5_new_hits", "top5_dropped_hits", "top5_net_change"),
        (
            (
                "pplx_original",
                comparison["query_count"],
                comparison["systems"]["pplx_original"]["hits_at_5"],
                0,
                0,
                0,
            ),
            (
                "pplx_reranked",
                comparison["query_count"],
                comparison["systems"]["pplx_reranked"]["hits_at_5"],
                pplx["top_5_change"]["new_hits"],
                pplx["top_5_change"]["dropped_hits"],
                pplx["top_5_change"]["net_change"],
            ),
            (
                "rrf_reranked",
                comparison["query_count"],
                comparison["systems"]["rrf_reranked"]["hits_at_5"],
                rrf["top_5_change"]["new_hits"],
                rrf["top_5_change"]["dropped_hits"],
                rrf["top_5_change"]["net_change"],
            ),
        ),
    )


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.model_snapshot = args.model_snapshot.resolve()
    chains = selected_chains(args)
    logger = RunLogger(args.output_dir / "run.log")
    completed = 0
    wall_started = time.perf_counter()
    try:
        question_order, queries = load_queries()
        rankings = {
            chain.name: load_and_validate_ranking(chain, question_order, queries)
            for chain in chains
        }
        question_ids, sample_manifest = select_question_ids(args, question_order)
        reused_question_ids = stage_reused_checkpoints(args, chains)
        logger(
            f"批次开始: chains={len(chains)}, questions={len(question_ids)}, "
            f"sample_mode={sample_manifest['mode']}, seed={sample_manifest['seed']}, "
            f"model={args.model_snapshot}, device={args.device}, dtype={args.dtype}, "
            f"batch_size={args.batch_size}, max_length={args.max_length}, "
            f"gpu_budget={args.gpu_memory_budget}, cpu_budget={args.cpu_memory_budget}"
        )
        if args.reuse_checkpoints_from is not None:
            logger(
                f"复用检查点: chain={chains[0].name}, count={len(reused_question_ids[chains[0].name])}, "
                f"source={args.reuse_checkpoints_from.resolve()}"
            )
        if args.validate_only:
            validation = {
                "status": "passed",
                "chains": len(chains),
                "candidate_documents_per_query": CANDIDATE_COUNT,
                "sample": sample_manifest,
                "model_snapshot": str(args.model_snapshot),
                "device": args.device,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "input_rankings": [display_path(chain.rankings_path) for chain in chains],
                "reused_checkpoint_counts": {
                    name: len(ids) for name, ids in reused_question_ids.items()
                },
            }
            atomic_json(args.output_dir / "validation.json", validation)
            logger(
                f"入口验证完成: chains={len(chains)}, questions={len(question_ids)}, "
                f"sample={display_path(args.output_dir / 'sample_question_ids.json')}"
            )
            return 0
        reranker = QwenReranker(args)
        logger(
            f"模型加载完成: seconds={reranker.load_seconds:.1f}; 本批次只加载一次; "
            f"device_map={json.dumps(reranker.device_map, ensure_ascii=False, sort_keys=True)}"
        )
        if torch.cuda.is_available() and any(
            value == 0 or str(value).startswith("cuda")
            for value in reranker.device_map.values()
        ):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        summaries: list[dict[str, Any]] = []
        progress_every = 10 if args.sample_size else 50
        for chain_index, chain in enumerate(chains, 1):
            summary = run_chain(
                chain_index,
                len(chains),
                chain,
                rankings[chain.name],
                question_ids,
                queries,
                reranker,
                args,
                logger,
                progress_every,
                reused_question_ids[chain.name],
            )
            summaries.append(summary)
            completed += 1

        summary_names = {row["chain"] for row in summaries}
        comparison = (
            fixed_600_comparison(summaries)
            if {"fixed_600_0_pplx", "fixed_600_0_rrf"} <= summary_names
            else None
        )
        if comparison is not None:
            atomic_json(args.output_dir / "fixed_600_0_comparison.json", comparison)
        write_summary_tables(args.output_dir, summaries, comparison)
        batch_summary: dict[str, Any] = {
            "status": "complete",
            "evaluation_split": "TRAIN only",
            "validation_query_limit": args.query_limit,
            "sample": sample_manifest,
            "completed_chains": completed,
            "expected_chains": len(chains),
            "model_snapshot": str(args.model_snapshot),
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "attn_implementation": args.attn_implementation,
            "gpu_memory_budget": args.gpu_memory_budget,
            "cpu_memory_budget": args.cpu_memory_budget,
            "device_map": reranker.device_map,
            "instruction": INSTRUCTION,
            "score_method": "softmax(no_logit, yes_logit)[yes]",
            "model_load_seconds": reranker.load_seconds,
            "wall_seconds": time.perf_counter() - wall_started,
            "chains": summaries,
        }
        if comparison is not None:
            batch_summary["fixed_600_0_comparison"] = comparison
        if torch.cuda.is_available() and any(
            value == 0 or str(value).startswith("cuda")
            for value in reranker.device_map.values()
        ):
            batch_summary["peak_gpu_memory_allocated_gib"] = (
                torch.cuda.max_memory_allocated() / (1024**3)
            )
            batch_summary["peak_gpu_memory_reserved_gib"] = (
                torch.cuda.max_memory_reserved() / (1024**3)
            )
        atomic_json(args.output_dir / "summary.json", batch_summary)
        logger(
            f"批次结束: 完成数 {completed}/{len(chains)}, elapsed={batch_summary['wall_seconds']:.1f}s, "
            f"output={display_path(args.output_dir)}"
        )
        return 0
    except BaseException as error:
        logger(f"ERROR: 完成数 {completed}/{len(chains)}; {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
