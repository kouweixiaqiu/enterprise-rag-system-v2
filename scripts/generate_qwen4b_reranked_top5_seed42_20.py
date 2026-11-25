#!/usr/bin/env python3
"""Generate the fixed 20 TRAIN answers from Qwen3-4B reranked Top-5 evidence."""

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
    CORPUS,
    QUERIES,
    RAW_TRAIN,
    SAMPLE_SIZE,
    fixed_sample,
    source_for,
)


TEMPLATE = (
    PROJECT_ROOT / "configs/generation/answer_with_citations_direct_conditions.prompt"
)
QWEN_RANKINGS = (
    PROJECT_ROOT
    / "experiments/dense/results/qwen3_reranker_4b_gguf_top20_fixed_600_0_train/rankings.jsonl"
)
BASELINE_GENERATIONS = (
    PROJECT_ROOT
    / "experiments/generation/results/"
    "siliconflow_train_context_window_prompt_v2_seed42_20/generations.jsonl"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/generation/results/"
    "siliconflow_train_qwen4b_top5_prompt_v2_neighbors_seed42_20"
)
GENERATIONS = OUTPUT_DIR / "generations.jsonl"
MANIFEST = OUTPUT_DIR / "manifest.json"
VARIANT = "qwen3_4b_top5_representative_with_neighbors"
BASELINE_PASSED = 9


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
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
    atomic_text(
        path,
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
    )


def render_prompt(
    template: str, query: dict[str, Any], candidates: list[dict[str, Any]], documents: dict[str, dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    if len(candidates) != 5 or len({row["document_id"] for row in candidates}) != 5:
        raise ValueError(f"invalid reranked Top-5 for {query['question_id']}")
    sources = [
        source_for(number, candidate, documents[candidate["document_id"]], "representative_with_neighbors")
        for number, candidate in enumerate(candidates, 1)
    ]
    evidence = "\n\n".join(
        f"[{source['number']}]\nTitle: {source['title']}\n\n{source['text']}"
        for source in sources
    )
    prompt = template.replace("{{question}}", query["query"]).replace(
        "{{evidence}}", evidence
    )
    if "{{question}}" in prompt or "{{evidence}}" in prompt:
        raise ValueError("rendered prompt contains an unresolved placeholder")
    return prompt, sources


def ranking_fingerprint(candidates: list[dict[str, Any]]) -> str:
    payload = [
        [candidate["document_id"], candidate["representative_chunk_id"]]
        for candidate in candidates
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def load_gold(question_ids: set[str]) -> dict[str, str]:
    records = json.loads(RAW_TRAIN.read_text(encoding="utf-8"))
    output = {
        row["QUESTION_ID"]: row["ANSWER"]
        for row in records
        if row.get("QUESTION_ID") in question_ids and isinstance(row.get("ANSWER"), str)
    }
    if set(output) != question_ids:
        raise ValueError("a sampled TRAIN gold answer is missing")
    return output


def main() -> int:
    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        print("SILICONFLOW_API_KEY is not set or is empty.", file=sys.stderr)
        return 2
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("{{question}}") != 1 or template.count("{{evidence}}") != 1:
        raise ValueError("generation template must contain exactly two placeholders")
    prompt_sha256 = hashlib.sha256(template.encode()).hexdigest()

    query_rows = list(jsonl_rows(QUERIES))
    sample = fixed_sample(query_rows)
    sample_ids = [row["question_id"] for row in sample]
    rankings = {row["question_id"]: row for row in jsonl_rows(QWEN_RANKINGS)}
    if len(rankings) != 450 or any(question_id not in rankings for question_id in sample_ids):
        raise ValueError("completed Qwen3-4B rankings are incomplete")
    baseline_rows = list(jsonl_rows(BASELINE_GENERATIONS))
    baseline = {row["question_id"]: row for row in baseline_rows}
    if len(baseline_rows) != SAMPLE_SIZE or set(baseline) != set(sample_ids):
        raise ValueError("prompt-v2 neighbor baseline does not match the fixed sample")

    top5_by_id: dict[str, list[dict[str, Any]]] = {}
    needed_documents: set[str] = set()
    for query in sample:
        question_id = query["question_id"]
        ranking = rankings[question_id]
        if ranking["reference_document_id"] != query["reference_document_id"]:
            raise ValueError(f"query/ranking mismatch for {question_id}")
        top5 = ranking["qwen_top_20"][:5]
        top5_by_id[question_id] = top5
        needed_documents.update(candidate["document_id"] for candidate in top5)
    documents = {
        row["document_id"]: row
        for row in jsonl_rows(CORPUS)
        if row["document_id"] in needed_documents
    }
    if set(documents) != needed_documents:
        raise ValueError("a reranked Top-5 document is missing from the corpus")

    existing: dict[str, dict[str, Any]] = {}
    if GENERATIONS.is_file():
        for row in jsonl_rows(GENERATIONS):
            question_id = row.get("question_id")
            expected_fingerprint = ranking_fingerprint(top5_by_id[question_id]) if question_id in top5_by_id else None
            if (
                question_id not in sample_ids
                or question_id in existing
                or row.get("variant") != VARIANT
                or row.get("model") != MODEL
                or row.get("temperature") != TEMPERATURE
                or row.get("max_tokens") != MAX_TOKENS
                or row.get("prompt_sha256") != prompt_sha256
                or row.get("ranking_fingerprint_sha256") != expected_fingerprint
            ):
                raise ValueError("existing Qwen3-4B generation checkpoint is incompatible")
            existing[question_id] = row

    completed: list[dict[str, Any]] = []
    reused_ids: list[str] = []
    regenerated_ids: list[str] = []
    for query in sample:
        question_id = query["question_id"]
        top5 = top5_by_id[question_id]
        fingerprint = ranking_fingerprint(top5)
        if question_id in existing:
            row = existing[question_id]
            completed.append(row)
            (reused_ids if row["reused_existing_answer"] else regenerated_ids).append(question_id)
            print(f"checkpoint reused {len(completed)}/{SAMPLE_SIZE}: {question_id}", flush=True)
            continue

        prompt, sources = render_prompt(template, query, top5, documents)
        old = baseline[question_id]
        old_document_ids = [source["document_id"] for source in old["sources"]]
        new_document_ids = [source["document_id"] for source in sources]
        reuse_answer = old_document_ids == new_document_ids
        if reuse_answer:
            if old["sources"] != sources:
                raise ValueError(f"unchanged Top-5 evidence changed for {question_id}")
            answer = old["answer"]
            reused_ids.append(question_id)
            action = "answer reused"
        else:
            try:
                answer = request_answer(api_key, prompt)
            except RuntimeError as exc:
                print(f"Stopped at {question_id}: {exc}", file=sys.stderr)
                return 1
            regenerated_ids.append(question_id)
            action = "answer generated"
        completed.append(
            {
                "question_id": question_id,
                "variant": VARIANT,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "prompt_sha256": prompt_sha256,
                "ranking_fingerprint_sha256": fingerprint,
                "reused_existing_answer": reuse_answer,
                "baseline_top5_document_ids": old_document_ids,
                "qwen_4b_top5_document_ids": new_document_ids,
                "answer": answer,
                "sources": sources,
            }
        )
        atomic_jsonl(GENERATIONS, completed)
        print(f"{action} {len(completed)}/{SAMPLE_SIZE}: {question_id}", flush=True)

    gold = load_gold(set(sample_ids))
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
            "retriever": "PPLX fixed_600_0 + Qwen3-Reranker-4B Top-20",
            "variant": VARIANT,
            "baseline_passed": BASELINE_PASSED,
            "answers_reused": len(reused_ids),
            "answers_regenerated": len(regenerated_ids),
            "reused_question_ids": reused_ids,
            "regenerated_question_ids": regenerated_ids,
            "requests_completed": len(regenerated_ids),
            "generation_file": str(GENERATIONS.relative_to(PROJECT_ROOT)),
        },
    )
    print(f"generations={GENERATIONS}", flush=True)
    print(f"manifest={MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
