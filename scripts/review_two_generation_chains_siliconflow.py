#!/usr/bin/env python3
"""Anonymously review two aligned 450-question generation chains via SiliconFlow."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import threading
import time
from pathlib import Path
from typing import Any

from review_three_chain_siliconflow import (
    API_KEY_FILE,
    MODEL,
    PROMPT,
    RetryableError,
    evidence_text,
    load_key,
    request,
    rows,
)


EXPECTED_QUESTIONS = 450
ANONYMIZATION_SEED = 20261125


def atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def extract_question(prompt: str) -> str:
    marker = "Question:\n"
    evidence_marker = "\n\nEvidence:\n"
    if marker not in prompt or evidence_marker not in prompt:
        raise ValueError("generation prompt does not contain Question/Evidence markers")
    return prompt.split(marker, 1)[1].split(evidence_marker, 1)[0].strip()


def load_chain(path: Path, label: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} generations file is missing: {path}")
    values = list(rows(path))
    data = {row["question_id"]: row for row in values}
    if len(values) != len(data):
        raise ValueError(f"{label} contains duplicate question IDs: {path}")
    if len(data) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"{label} has {len(data)} questions, expected {EXPECTED_QUESTIONS}: {path}"
        )
    return data


def build_items(
    left_path: Path,
    right_path: Path,
    left_label: str,
    right_label: str,
) -> list[dict[str, Any]]:
    chains = {
        left_label: load_chain(left_path, left_label),
        right_label: load_chain(right_path, right_label),
    }
    question_sets = [set(data) for data in chains.values()]
    if question_sets[0] != question_sets[1]:
        missing_left = sorted(question_sets[1] - question_sets[0])
        missing_right = sorted(question_sets[0] - question_sets[1])
        raise ValueError(
            "two chains have different question sets: "
            f"missing_from_{left_label}={missing_left[:10]}, "
            f"missing_from_{right_label}={missing_right[:10]}"
        )

    items: list[dict[str, Any]] = []
    for label, data in chains.items():
        for question_id in sorted(data):
            row = data[question_id]
            if not row.get("gold_answer"):
                raise ValueError(f"{label}/{question_id} has no reference answer")
            if not isinstance(row.get("sources"), list) or not row["sources"]:
                raise ValueError(f"{label}/{question_id} has no supplied evidence")
            items.append(
                {
                    "chain": label,
                    "question_id": question_id,
                    "question": extract_question(row["prompt"]),
                    "gold": row["gold_answer"],
                    "answer": row["answer"],
                    "sources": row["sources"],
                }
            )

    random.Random(ANONYMIZATION_SEED).shuffle(items)
    for index, item in enumerate(items, 1):
        item["anonymous_id"] = f"J{index:04d}"
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-generations", type=Path, required=True)
    parser.add_argument("--right-generations", type=Path, required=True)
    parser.add_argument("--left-label", default="fixed_600_0_dense")
    parser.add_argument("--right-label", default="paragraph_merge_1200_0_rrf")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retry-delay", type=float, default=8.0)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.left_label == args.right_label:
        parser.error("the two labels must differ")

    api_key = load_key()
    if not api_key:
        raise SystemExit(
            f"SILICONFLOW_API_KEY missing from environment and {API_KEY_FILE}"
        )

    items = build_items(
        args.left_generations.resolve(),
        args.right_generations.resolve(),
        args.left_label,
        args.right_label,
    )
    expected = EXPECTED_QUESTIONS * 2
    if len(items) != expected:
        raise ValueError(f"built {len(items)} review items, expected {expected}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "initial_reviews.jsonl"
    mapping_path = args.output_dir / "anonymous_mapping.jsonl"
    expected_mapping = [
        {
            "anonymous_id": item["anonymous_id"],
            "question_id": item["question_id"],
            "chain": item["chain"],
        }
        for item in items
    ]
    if mapping_path.is_file():
        saved_mapping = list(rows(mapping_path))
        if saved_mapping != expected_mapping:
            raise ValueError(
                f"existing anonymous mapping does not match this run: {mapping_path}"
            )
    else:
        atomic_jsonl(mapping_path, expected_mapping)

    completed = (
        {row["anonymous_id"]: row for row in rows(output_path)}
        if output_path.is_file()
        else {}
    )
    valid_ids = {item["anonymous_id"] for item in items}
    unknown = sorted(set(completed) - valid_ids)
    if unknown:
        raise ValueError(f"review checkpoint contains unknown IDs: {unknown[:10]}")
    pending = [item for item in items if item["anonymous_id"] not in completed]
    pending.sort(key=lambda item: (item["question_id"], item["anonymous_id"]))
    lock = threading.Lock()

    def judge(item: dict[str, Any]) -> dict[str, Any]:
        base_prompt = PROMPT.format(
            question=item["question"],
            gold=item["gold"],
            answer=item["answer"],
            evidence=evidence_text(item["sources"]),
        )
        retry = 0
        while True:
            try:
                retry_note = "" if retry == 0 else (
                    f"\nThis is validation attempt {retry + 1}. Return valid JSON "
                    "with every required string non-empty."
                )
                judgment, usage = request(
                    api_key, base_prompt + retry_note, args.timeout
                )
                return {
                    "anonymous_id": item["anonymous_id"],
                    "question_id": item["question_id"],
                    "chain": item["chain"],
                    **judgment,
                    "judge_model": MODEL,
                    "usage": usage,
                }
            except RetryableError as exc:
                retry += 1
                delay = min(60.0, args.retry_delay * (2 ** min(retry - 1, 3)))
                with lock:
                    print(
                        f"retry {item['anonymous_id']} #{retry} in {delay:g}s: {exc}",
                        flush=True,
                    )
                time.sleep(delay)

    print(
        f"loaded={expected} reused={len(completed)} pending={len(pending)} "
        f"workers={args.workers}",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(judge, item): item for item in pending}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed[row["anonymous_id"]] = row
            ordered = [
                completed[item["anonymous_id"]]
                for item in items
                if item["anonymous_id"] in completed
            ]
            atomic_jsonl(output_path, ordered)
            done = len(completed)
            if done % 25 == 0 or done == expected:
                print(f"reviewed {done}/{expected}", flush=True)

    print(f"initial_reviews={output_path}", flush=True)
    print(f"anonymous_mapping={mapping_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
