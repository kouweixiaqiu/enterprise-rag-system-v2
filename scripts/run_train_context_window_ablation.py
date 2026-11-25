#!/usr/bin/env python3
"""Generate a fixed 20-question TRAIN context-window ablation with SiliconFlow."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_train_q000_siliconflow import (  # noqa: E402
    MODEL,
    TEMPLATE,
    jsonl_rows,
    request_answer,
)


SEED = 42
SAMPLE_SIZE = 20
RANKING_DEPTH = 5
CHUNK_SIZE = 600
QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
RANKINGS = (
    PROJECT_ROOT
    / "experiments/dense/results/pplx_embed_v1_0.6b_fixed_600_0_train/"
    "dense_rankings_top100.jsonl"
)
RAW_TRAIN = (
    PROJECT_ROOT
    / "data/raw/techqa/TechQA/training_and_dev/training_Q_A.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/generation/results/"
    "siliconflow_train_context_window_seed42_20"
)
GENERATIONS = OUTPUT_DIR / "generations.jsonl"
MANIFEST = OUTPUT_DIR / "manifest.json"


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


def fixed_sample(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indices = sorted(random.Random(SEED).sample(range(len(queries)), SAMPLE_SIZE))
    return [queries[index] for index in indices]


def load_inputs() -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    queries = list(jsonl_rows(QUERIES))
    rankings = {row["question_id"]: row for row in jsonl_rows(RANKINGS)}
    if len(queries) != 450 or len(rankings) != 450:
        raise ValueError("expected 450 saved TRAIN queries and PPLX rankings")
    sample = fixed_sample(queries)
    needed_document_ids = {
        candidate["document_id"]
        for query in sample
        for candidate in rankings[query["question_id"]]["top_documents"][:RANKING_DEPTH]
    }
    documents = {
        row["document_id"]: row
        for row in jsonl_rows(CORPUS)
        if row["document_id"] in needed_document_ids
    }
    if set(documents) != needed_document_ids:
        raise ValueError("a sampled candidate document is missing from the corpus")
    return sample, rankings, documents


def source_for(
    source_number: int,
    candidate: dict[str, Any],
    document: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    chunk_id = candidate["representative_chunk_id"]
    if not chunk_id.startswith(candidate["document_id"] + "::"):
        raise ValueError(f"representative chunk/document mismatch: {chunk_id}")
    representative_index = int(chunk_id.rsplit("::", 1)[1])
    document_text = document["text"]
    chunk_count = (len(document_text) + CHUNK_SIZE - 1) // CHUNK_SIZE
    if variant == "representative_only":
        chunk_indices = [representative_index]
    elif variant == "representative_with_neighbors":
        chunk_indices = list(
            range(
                max(0, representative_index - 1),
                min(chunk_count, representative_index + 2),
            )
        )
    else:
        raise ValueError(f"unknown variant: {variant}")
    included_chunk_ids = [
        f"{candidate['document_id']}::{index:06d}" for index in chunk_indices
    ]
    text = "".join(
        document_text[index * CHUNK_SIZE : (index + 1) * CHUNK_SIZE]
        for index in chunk_indices
    )
    return {
        "number": source_number,
        "document_id": candidate["document_id"],
        "representative_chunk_id": chunk_id,
        "included_chunk_ids": included_chunk_ids,
        "title": document["title"],
        "text": text,
    }


def render_prompt(
    template: str,
    query: dict[str, Any],
    ranking: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    variant: str,
) -> tuple[str, list[dict[str, Any]]]:
    candidates = ranking["top_documents"][:RANKING_DEPTH]
    if len(candidates) != RANKING_DEPTH or len(
        {row["document_id"] for row in candidates}
    ) != RANKING_DEPTH:
        raise ValueError(f"invalid PPLX Top-5 for {query['question_id']}")
    sources = [
        source_for(number, candidate, documents[candidate["document_id"]], variant)
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


def load_gold_after_generation(question_ids: set[str]) -> dict[str, str]:
    records = json.loads(RAW_TRAIN.read_text(encoding="utf-8"))
    output = {
        row["QUESTION_ID"]: row["ANSWER"]
        for row in records
        if row.get("QUESTION_ID") in question_ids
        and isinstance(row.get("ANSWER"), str)
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
    sample, rankings, documents = load_inputs()
    prompt_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    sample_ids = {query["question_id"] for query in sample}
    variants = ("representative_only", "representative_with_neighbors")
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    if GENERATIONS.is_file():
        for row in jsonl_rows(GENERATIONS):
            key = (row.get("question_id"), row.get("variant"))
            if (
                key[0] not in sample_ids
                or key[1] not in variants
                or row.get("model") != MODEL
                or row.get("prompt_sha256") != prompt_sha256
                or key in existing
            ):
                raise ValueError("existing generation checkpoint is incompatible")
            existing[key] = row
    rows: list[dict[str, Any]] = []
    for query in sample:
        ranking = rankings[query["question_id"]]
        if ranking["reference_document_id"] != query["reference_document_id"]:
            raise ValueError(f"query/ranking mismatch for {query['question_id']}")
        for variant in variants:
            checkpoint_key = (query["question_id"], variant)
            if checkpoint_key in existing:
                rows.append(existing[checkpoint_key])
                print(
                    f"reused {len(rows)}/{SAMPLE_SIZE * len(variants)}: "
                    f"{query['question_id']} / {variant}",
                    flush=True,
                )
                continue
            prompt, sources = render_prompt(
                template, query, ranking, documents, variant
            )
            try:
                answer = request_answer(api_key, prompt)
            except RuntimeError as exc:
                print(
                    f"Stopped at {query['question_id']} / {variant}: {exc}",
                    file=sys.stderr,
                )
                return 1
            rows.append(
                {
                    "question_id": query["question_id"],
                    "variant": variant,
                    "model": MODEL,
                    "prompt_sha256": prompt_sha256,
                    "answer": answer,
                    "sources": sources,
                }
            )
            # Commit completed generations without logging the API key or prompt headers.
            atomic_jsonl(GENERATIONS, rows)
            print(
                f"completed {len(rows)}/{SAMPLE_SIZE * len(variants)}: "
                f"{query['question_id']} / {variant}",
                flush=True,
            )

    # Gold labels are loaded only after all generation requests have completed.
    gold = load_gold_after_generation({query["question_id"] for query in sample})
    for row in rows:
        row["gold_answer"] = gold[row["question_id"]]
    atomic_jsonl(GENERATIONS, rows)
    atomic_json(
        MANIFEST,
        {
            "status": "generated_pending_support_audit",
            "evaluation_split": "TRAIN only",
            "seed": SEED,
            "sample_size": SAMPLE_SIZE,
            "question_ids": [query["question_id"] for query in sample],
            "model": MODEL,
            "prompt_sha256": prompt_sha256,
            "prompt_template": str(TEMPLATE.relative_to(PROJECT_ROOT)),
            "retriever": "PPLX fixed_600_0 original Top-5",
            "variants": list(variants),
            "requests_completed": len(rows),
            "generation_file": str(GENERATIONS.relative_to(PROJECT_ROOT)),
        },
    )
    print(f"generations={GENERATIONS}", flush=True)
    print(f"manifest={MANIFEST}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
