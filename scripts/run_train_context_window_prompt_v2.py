#!/usr/bin/env python3
"""Regenerate the fixed 20 TRAIN cases with the neighbor context and prompt v2."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_train_q000_siliconflow import (  # noqa: E402
    MAX_TOKENS,
    MODEL,
    TEMPERATURE,
    request_answer,
)
from scripts.run_train_context_window_ablation import (  # noqa: E402
    RANKING_DEPTH,
    SAMPLE_SIZE,
    load_gold_after_generation,
    load_inputs,
    render_prompt,
)


TEMPLATE = (
    PROJECT_ROOT
    / "configs/generation/answer_with_citations_direct_conditions.prompt"
)
BASELINE_GENERATIONS = (
    PROJECT_ROOT
    / "experiments/generation/results/"
    "siliconflow_train_context_window_seed42_20/generations.jsonl"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/generation/results/"
    "siliconflow_train_context_window_prompt_v2_seed42_20"
)
GENERATIONS = OUTPUT_DIR / "generations.jsonl"
MANIFEST = OUTPUT_DIR / "manifest.json"
VARIANT = "representative_with_neighbors"


def jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


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
    lines = (
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
    )
    atomic_text(path, "\n".join(lines) + "\n")


def baseline_neighbor_sources() -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for row in jsonl_rows(BASELINE_GENERATIONS):
        if row.get("variant") != VARIANT:
            continue
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or question_id in output:
            raise ValueError("invalid baseline neighbor checkpoint")
        output[question_id] = row["sources"]
    if len(output) != SAMPLE_SIZE:
        raise ValueError("expected 20 baseline neighbor cases")
    return output


def main() -> int:
    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        print("SILICONFLOW_API_KEY is not set or is empty.", file=sys.stderr)
        return 2

    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("{{question}}") != 1 or template.count("{{evidence}}") != 1:
        raise ValueError("generation template must contain exactly two placeholders")

    sample, rankings, documents = load_inputs()
    baseline_sources = baseline_neighbor_sources()
    prompt_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    sample_ids = [query["question_id"] for query in sample]
    if set(sample_ids) != set(baseline_sources):
        raise ValueError("sample IDs differ from the baseline run")

    existing: dict[str, dict[str, Any]] = {}
    if GENERATIONS.is_file():
        for row in jsonl_rows(GENERATIONS):
            question_id = row.get("question_id")
            if (
                question_id not in sample_ids
                or row.get("variant") != VARIANT
                or row.get("model") != MODEL
                or row.get("temperature") != TEMPERATURE
                or row.get("max_tokens") != MAX_TOKENS
                or row.get("prompt_sha256") != prompt_sha256
                or question_id in existing
            ):
                raise ValueError("existing prompt-v2 checkpoint is incompatible")
            existing[question_id] = row

    completed: list[dict[str, Any]] = []
    for query in sample:
        question_id = query["question_id"]
        prompt, sources = render_prompt(
            template,
            query,
            rankings[question_id],
            documents,
            VARIANT,
        )
        if sources != baseline_sources[question_id]:
            raise ValueError(f"neighbor evidence changed for {question_id}")

        if question_id in existing:
            completed.append(existing[question_id])
            print(f"reused {len(completed)}/{SAMPLE_SIZE}: {question_id}", flush=True)
            continue

        try:
            answer = request_answer(api_key, prompt)
        except RuntimeError as exc:
            print(f"Stopped at {question_id}: {exc}", file=sys.stderr)
            return 1
        completed.append(
            {
                "question_id": question_id,
                "variant": VARIANT,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "prompt_sha256": prompt_sha256,
                "answer": answer,
                "sources": sources,
            }
        )
        atomic_jsonl(GENERATIONS, completed)
        print(f"completed {len(completed)}/{SAMPLE_SIZE}: {question_id}", flush=True)

    gold = load_gold_after_generation(set(sample_ids))
    for row in completed:
        row["gold_answer"] = gold[row["question_id"]]
    atomic_jsonl(GENERATIONS, completed)
    atomic_json(
        MANIFEST,
        {
            "status": "generated_pending_support_audit",
            "evaluation_split": "TRAIN only",
            "seed": 42,
            "sample_size": SAMPLE_SIZE,
            "question_ids": sample_ids,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "prompt_sha256": prompt_sha256,
            "prompt_template": str(TEMPLATE.relative_to(PROJECT_ROOT)),
            "retriever": "PPLX fixed_600_0 original Top-5",
            "variant": VARIANT,
            "baseline_passed": 8,
            "requests_completed": len(completed),
            "generation_file": str(GENERATIONS.relative_to(PROJECT_ROOT)),
        },
    )
    print(f"generations={GENERATIONS}", flush=True)
    print(f"manifest={MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
