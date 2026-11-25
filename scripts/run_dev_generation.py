#!/usr/bin/env python3
"""Generate cited answers for all 147 deduplicated TechQA DEV questions."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from datetime import datetime
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
    jsonl_rows,
    render_prompt,
)


EXPECTED_QUESTION_COUNT = 147
VARIANT = "representative_with_neighbors"
TEMPLATE = (
    PROJECT_ROOT
    / "configs/generation/answer_with_citations_direct_conditions.prompt"
)
QUERIES = PROJECT_ROOT / "data/processed/techqa/dev_queries.jsonl"
CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
RANKINGS = (
    PROJECT_ROOT
    / "experiments/dense/results/pplx_embed_v1_0.6b_fixed_600_0_dev/"
    "rankings_top50.jsonl"
)
RAW_DEV = (
    PROJECT_ROOT
    / "data/raw/techqa/TechQA/training_and_dev/dev_Q_A.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/generation/results/"
    "siliconflow_dev_pplx_top5_neighbors_prompt_v2_147"
)
GENERATIONS = OUTPUT_DIR / "generations.jsonl"
CONTEXTS = OUTPUT_DIR / "contexts.jsonl"
MANIFEST = OUTPUT_DIR / "manifest.json"
RUN_LOG = OUTPUT_DIR / "run.log"
EXIT_CODE = OUTPUT_DIR / "exit_code.txt"


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


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line + "\n")


def unique_index(
    rows: Iterable[dict[str, Any]], key: str, description: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value or value in output:
            raise ValueError(f"invalid or duplicate {description}: {value!r}")
        output[value] = row
    return output


def load_inputs() -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    queries = list(jsonl_rows(QUERIES))
    if len(queries) != EXPECTED_QUESTION_COUNT:
        raise ValueError(
            f"expected {EXPECTED_QUESTION_COUNT} deduplicated DEV questions, "
            f"found {len(queries)}"
        )
    query_index = unique_index(queries, "question_id", "DEV question ID")
    rankings = unique_index(
        jsonl_rows(RANKINGS), "question_id", "DEV ranking question ID"
    )
    if set(rankings) != set(query_index):
        missing = sorted(set(query_index) - set(rankings))
        extra = sorted(set(rankings) - set(query_index))
        raise ValueError(
            f"DEV query/ranking IDs differ; missing={missing}, extra={extra}"
        )

    needed_document_ids: set[str] = set()
    for query in queries:
        question_id = query["question_id"]
        ranking = rankings[question_id]
        if ranking.get("reference_document_id") != query.get(
            "reference_document_id"
        ):
            raise ValueError(f"query/ranking reference mismatch for {question_id}")
        candidates = ranking.get("top_documents", [])[:RANKING_DEPTH]
        if len(candidates) != RANKING_DEPTH or len(
            {candidate.get("document_id") for candidate in candidates}
        ) != RANKING_DEPTH:
            raise ValueError(f"invalid PPLX Top-5 for {question_id}")
        needed_document_ids.update(
            candidate["document_id"] for candidate in candidates
        )

    documents = {
        row["document_id"]: row
        for row in jsonl_rows(CORPUS)
        if row.get("document_id") in needed_document_ids
    }
    if set(documents) != needed_document_ids:
        missing = sorted(needed_document_ids - set(documents))
        raise ValueError(f"candidate documents missing from corpus: {missing}")
    return queries, rankings, documents


def build_contexts(
    template: str,
    queries: list[dict[str, Any]],
    rankings: dict[str, dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for query in queries:
        question_id = query["question_id"]
        ranking = rankings[question_id]
        prompt, sources = render_prompt(
            template, query, ranking, documents, VARIANT
        )
        contexts.append(
            {
                "question_id": question_id,
                "query": query["query"],
                "variant": VARIANT,
                "top_documents": ranking["top_documents"][:RANKING_DEPTH],
                "sources": sources,
                "prompt": prompt,
            }
        )
    return contexts


def load_gold_after_generation(question_ids: set[str]) -> dict[str, str]:
    records = json.loads(RAW_DEV.read_text(encoding="utf-8"))
    output: dict[str, str] = {}
    for row in records:
        question_id = row.get("QUESTION_ID")
        answer = row.get("ANSWER")
        if question_id not in question_ids or not isinstance(answer, str):
            continue
        if question_id in output and output[question_id] != answer:
            raise ValueError(f"conflicting DEV reference answers for {question_id}")
        output[question_id] = answer
    if set(output) != question_ids:
        missing = sorted(question_ids - set(output))
        raise ValueError(f"DEV reference answers missing for: {missing}")
    return output


def load_existing(
    question_ids: set[str],
    contexts: dict[str, dict[str, Any]],
    prompt_sha256: str,
) -> dict[str, dict[str, Any]]:
    existing: dict[str, dict[str, Any]] = {}
    if not GENERATIONS.is_file():
        return existing
    for row in jsonl_rows(GENERATIONS):
        question_id = row.get("question_id")
        if (
            question_id not in question_ids
            or question_id in existing
            or row.get("variant") != VARIANT
            or row.get("model") != MODEL
            or row.get("temperature") != TEMPERATURE
            or row.get("max_tokens") != MAX_TOKENS
            or row.get("prompt_sha256") != prompt_sha256
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
            or row.get("sources") != contexts[question_id]["sources"]
        ):
            raise ValueError("existing DEV generation checkpoint is incompatible")
        existing[question_id] = row
    return existing


def write_manifest(
    status: str,
    prompt_sha256: str,
    question_ids: list[str],
    completed: int,
) -> None:
    atomic_json(
        MANIFEST,
        {
            "status": status,
            "evaluation_split": "DEV answerable, exact TRAIN duplicates excluded",
            "question_count": EXPECTED_QUESTION_COUNT,
            "question_ids": question_ids,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "prompt_sha256": prompt_sha256,
            "prompt_template": str(TEMPLATE.relative_to(PROJECT_ROOT)),
            "retriever": "PPLX fixed_600_0 original Top-5",
            "variant": VARIANT,
            "requests_completed": completed,
            "generation_file": str(GENERATIONS.relative_to(PROJECT_ROOT)),
            "contexts_file": str(CONTEXTS.relative_to(PROJECT_ROOT)),
            "run_log": str(RUN_LOG.relative_to(PROJECT_ROOT)),
            "exit_code_file": str(EXIT_CODE.relative_to(PROJECT_ROOT)),
        },
    )


def main() -> int:
    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        log("SILICONFLOW_API_KEY is not set or is empty.")
        return 2

    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("{{question}}") != 1 or template.count("{{evidence}}") != 1:
        raise ValueError("generation template must contain exactly two placeholders")
    prompt_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    queries, rankings, documents = load_inputs()
    context_rows = build_contexts(template, queries, rankings, documents)
    atomic_jsonl(CONTEXTS, context_rows)
    contexts = {row["question_id"]: row for row in context_rows}
    question_ids = [query["question_id"] for query in queries]
    question_id_set = set(question_ids)
    existing = load_existing(question_id_set, contexts, prompt_sha256)
    log(
        f"Starting DEV generation: total={EXPECTED_QUESTION_COUNT}, "
        f"reusable={len(existing)}, remaining={EXPECTED_QUESTION_COUNT - len(existing)}"
    )
    write_manifest("running", prompt_sha256, question_ids, len(existing))

    successful = dict(existing)
    for processed, context in enumerate(context_rows, 1):
        question_id = context["question_id"]
        if question_id in successful:
            log(
                f"reused input {processed}/{EXPECTED_QUESTION_COUNT}: "
                f"{question_id} (saved={len(successful)})"
            )
            continue
        try:
            answer = request_answer(api_key, context["prompt"])
        except RuntimeError as exc:
            log(f"Stopped at {question_id}: {exc}")
            write_manifest("failed", prompt_sha256, question_ids, len(successful))
            return 1
        successful[question_id] = {
            "question_id": question_id,
            "variant": VARIANT,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "prompt_sha256": prompt_sha256,
            "answer": answer,
            "sources": context["sources"],
        }
        checkpoint = [
            successful[saved_question_id]
            for saved_question_id in question_ids
            if saved_question_id in successful
        ]
        atomic_jsonl(GENERATIONS, checkpoint)
        write_manifest("running", prompt_sha256, question_ids, len(successful))
        log(
            f"completed {len(successful)}/{EXPECTED_QUESTION_COUNT}: {question_id}"
        )

    # Reference answers are intentionally loaded only after every generation is done.
    gold = load_gold_after_generation(question_id_set)
    completed = [successful[question_id] for question_id in question_ids]
    for row in completed:
        row["gold_answer"] = gold[row["question_id"]]
    atomic_jsonl(GENERATIONS, completed)
    write_manifest(
        "generated_pending_human_review",
        prompt_sha256,
        question_ids,
        len(completed),
    )
    log(f"Completed {len(completed)}/{EXPECTED_QUESTION_COUNT} DEV generations")
    log(f"generations={GENERATIONS}")
    log(f"contexts={CONTEXTS}")
    log(f"manifest={MANIFEST}")
    return 0


if __name__ == "__main__":
    try:
        result = main()
    except KeyboardInterrupt:
        log("Interrupted by user.")
        result = 130
    except Exception:
        log("Unhandled error:\n" + traceback.format_exc().rstrip())
        result = 1
    atomic_text(EXIT_CODE, f"{result}\n")
    raise SystemExit(result)
