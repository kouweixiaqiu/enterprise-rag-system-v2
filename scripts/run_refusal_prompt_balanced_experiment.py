#!/usr/bin/env python3
"""Run a balanced 36 answerable / 36 no-answer refusal-prompt experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_train_q000_siliconflow import (  # noqa: E402
    MAX_TOKENS,
    MODEL,
    TEMPERATURE,
    RetryableSiliconFlowError,
    request_completion,
)
from scripts.run_train_generation_from_rankings import (  # noqa: E402
    citation_map,
    hard_request_deadline,
    load_api_key,
    render_prompt,
)


NEGATIVE_AUDIT = ROOT / (
    "experiments/generation/results/"
    "siliconflow_train150_unanswerable_pplx_fixed_600_0_top5_neighbors_refusal/"
    "evidence_audit_top100/no_answer_top100.jsonl"
)
NEGATIVE_CONTEXTS = ROOT / (
    "experiments/generation/results/"
    "siliconflow_train150_unanswerable_pplx_fixed_600_0_top5_neighbors_refusal/"
    "contexts.jsonl"
)
NEGATIVE_BASELINE = ROOT / (
    "experiments/generation/results/"
    "siliconflow_train150_unanswerable_pplx_fixed_600_0_top5_neighbors_refusal/"
    "generations.jsonl"
)
POSITIVE_CONTEXTS = ROOT / (
    "experiments/generation/results/"
    "siliconflow_train450_pplx_fixed_600_0_top5_prompt_v2_representative_with_neighbors/"
    "contexts.jsonl"
)
POSITIVE_BASELINE = ROOT / (
    "experiments/generation/results/"
    "siliconflow_train30_answerable_pplx_fixed_600_0_top5_neighbors_refusal/"
    "generations.jsonl"
)
RAW_TRAIN = ROOT / "data/raw/techqa/TechQA/training_and_dev/training_Q_A.json"
BASELINE_PROMPT = ROOT / "configs/generation/answer_with_citations_refusal.prompt"
BALANCED_PROMPT = ROOT / (
    "configs/generation/answer_with_citations_refusal_balanced.prompt"
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/generation/results/"
    "refusal_prompt_balanced_36_answerable_36_no_answer"
)
DEFAULT_KEY_FILE = Path.home() / ".config/enterprise-rag/siliconflow.env"
SEED = 42
PER_CLASS = 36
TUNE_PER_CLASS = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    parser.add_argument("--max-retry-delay-seconds", type=float, default=60.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().lower()


def parse_decision(answer: str) -> str:
    first = next((line.strip() for line in answer.splitlines() if line.strip()), "")
    match = re.fullmatch(r"DECISION:\s*(ANSWER|INSUFFICIENT_CONTEXT)", first)
    return match.group(1) if match else "INVALID_FORMAT"


def build_dataset() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    raw = {row["QUESTION_ID"]: row for row in json.loads(RAW_TRAIN.read_text(encoding="utf-8"))}
    negative_ids = [row["question_id"] for row in read_jsonl(NEGATIVE_AUDIT)]
    if len(negative_ids) != PER_CLASS or len(set(negative_ids)) != PER_CLASS:
        raise ValueError("expected 36 distinct audited Top-100 no-answer IDs")

    negative_contexts = {row["question_id"]: row for row in read_jsonl(NEGATIVE_CONTEXTS)}
    positive_contexts = {row["question_id"]: row for row in read_jsonl(POSITIVE_CONTEXTS)}
    reusable_positive = {row["question_id"] for row in read_jsonl(POSITIVE_BASELINE)}
    eligible_positive = []
    for question_id, context in positive_contexts.items():
        answer = normalized(raw[question_id]["ANSWER"])
        supplied = normalized(" ".join(source["text"] for source in context["sources"]))
        if answer and answer in supplied:
            eligible_positive.append(question_id)
    reusable_exact = sorted(set(eligible_positive) & reusable_positive)
    if len(reusable_exact) > PER_CLASS:
        reusable_exact = reusable_exact[:PER_CLASS]
    remaining = sorted(set(eligible_positive) - set(reusable_exact))
    rng = random.Random(SEED)
    rng.shuffle(remaining)
    positive_ids = reusable_exact + remaining[: PER_CLASS - len(reusable_exact)]
    if len(positive_ids) != PER_CLASS or len(set(positive_ids)) != PER_CLASS:
        raise ValueError("could not select 36 distinct exact-answer-in-context positives")

    def split_ids(ids: list[str]) -> dict[str, str]:
        shuffled = list(ids)
        random.Random(SEED).shuffle(shuffled)
        return {
            question_id: ("tune" if index < TUNE_PER_CLASS else "holdout")
            for index, question_id in enumerate(shuffled)
        }

    negative_split = split_ids(negative_ids)
    positive_split = split_ids(positive_ids)
    dataset = []
    contexts: dict[str, dict[str, Any]] = {}
    for label, ids, source, expected, split in (
        ("no_answer", negative_ids, negative_contexts, "INSUFFICIENT_CONTEXT", negative_split),
        ("answerable", positive_ids, positive_contexts, "ANSWER", positive_split),
    ):
        for question_id in ids:
            context = source[question_id]
            key = f"{label}:{question_id}"
            contexts[key] = context
            dataset.append(
                {
                    "case_key": key,
                    "question_id": question_id,
                    "class": label,
                    "split": split[question_id],
                    "expected_decision": expected,
                    "question": context["question"],
                    "source_document_ids": [s["document_id"] for s in context["sources"]],
                    "source_chunk_ids": [s["included_chunk_ids"] for s in context["sources"]],
                }
            )
    return dataset, contexts


def baseline_reuse() -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for label, path in (("no_answer", NEGATIVE_BASELINE), ("answerable", POSITIVE_BASELINE)):
        for row in read_jsonl(path):
            values[f"{label}:{row['question_id']}"] = row
    return values


def run(args: argparse.Namespace) -> None:
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dataset, contexts = build_dataset()
    write_json(output / "dataset.json", {"seed": SEED, "cases": dataset})
    if args.prepare_only:
        print(f"prepared {len(dataset)} cases at {output}", flush=True)
        return

    prompts = {
        "baseline": BASELINE_PROMPT.read_text(encoding="utf-8"),
        "balanced": BALANCED_PROMPT.read_text(encoding="utf-8"),
    }
    api_key = load_api_key(args.api_key_file)
    if not api_key:
        raise RuntimeError(f"SILICONFLOW_API_KEY missing from environment and {args.api_key_file}")
    reused = baseline_reuse()

    for variant, template in prompts.items():
        variant_dir = output / variant
        generation_path = variant_dir / "generations.jsonl"
        existing = {
            row["case_key"]: row
            for row in (read_jsonl(generation_path) if generation_path.exists() else [])
        }
        prompt_hash = hashlib.sha256(template.encode()).hexdigest()
        ordered: list[dict[str, Any]] = []
        for index, case in enumerate(dataset, 1):
            key = case["case_key"]
            context = contexts[key]
            prompt = render_prompt(template, context["question"], context["sources"])
            if key in existing and existing[key].get("prompt_sha256") == prompt_hash:
                row = existing[key]
                print(f"[{variant}] reused checkpoint {index}/72 {case['question_id']}", flush=True)
            elif variant == "baseline" and key in reused:
                old = reused[key]
                row = {
                    **case,
                    "model": old["model"],
                    "temperature": old["temperature"],
                    "max_tokens": old["max_tokens"],
                    "prompt_sha256": prompt_hash,
                    "decision": old.get("decision") or parse_decision(old["answer"]),
                    "answer": old["answer"],
                    "sources": context["sources"],
                    "citation_map": old.get("citation_map", []),
                    "usage": old.get("usage", {}),
                    "reused_from_existing_run": True,
                }
                print(f"[{variant}] reused prior {index}/72 {case['question_id']}", flush=True)
            else:
                retries = 0
                while True:
                    try:
                        with hard_request_deadline(args.request_timeout_seconds):
                            answer, usage = request_completion(
                                api_key, prompt, timeout=max(1, int(args.request_timeout_seconds))
                            )
                        break
                    except (RetryableSiliconFlowError, TimeoutError) as exc:
                        retries += 1
                        delay = min(
                            args.max_retry_delay_seconds,
                            args.retry_delay_seconds * (2 ** min(retries - 1, 6)),
                        )
                        print(
                            f"[{variant}] transient {case['question_id']} retry={retries} "
                            f"in {delay:g}s: {exc}", flush=True
                        )
                        time.sleep(delay)
                row = {
                    **case,
                    "model": MODEL,
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "prompt_sha256": prompt_hash,
                    "decision": parse_decision(answer),
                    "answer": answer,
                    "sources": context["sources"],
                    "citation_map": citation_map(answer, context["sources"]),
                    "usage": usage,
                    "retry_count": retries,
                    "reused_from_existing_run": False,
                }
                print(
                    f"[{variant}] generated {index}/72 {case['question_id']} "
                    f"decision={row['decision']}", flush=True
                )
            existing[key] = row
            ordered = [existing[item["case_key"]] for item in dataset if item["case_key"] in existing]
            write_jsonl(generation_path, ordered)

        summary: dict[str, Any] = {"variant": variant, "total": len(ordered), "by_split": {}}
        for split in ("tune", "holdout", "all"):
            selected = ordered if split == "all" else [row for row in ordered if row["split"] == split]
            counts = {
                "no_answer_correct_refusal": sum(
                    row["class"] == "no_answer" and row["decision"] == "INSUFFICIENT_CONTEXT"
                    for row in selected
                ),
                "no_answer_incorrect_answer": sum(
                    row["class"] == "no_answer" and row["decision"] == "ANSWER"
                    for row in selected
                ),
                "answerable_answered": sum(
                    row["class"] == "answerable" and row["decision"] == "ANSWER"
                    for row in selected
                ),
                "answerable_false_refusal": sum(
                    row["class"] == "answerable" and row["decision"] == "INSUFFICIENT_CONTEXT"
                    for row in selected
                ),
                "invalid_format": sum(row["decision"] == "INVALID_FORMAT" for row in selected),
            }
            summary["by_split"][split] = {"case_count": len(selected), **counts}
        write_json(variant_dir / "summary.json", summary)
        print(f"[{variant}] complete: {variant_dir}", flush=True)

    print(f"experiment complete: {output}", flush=True)


if __name__ == "__main__":
    run(parse_args())
