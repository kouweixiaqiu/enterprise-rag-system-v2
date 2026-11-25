#!/usr/bin/env python3
"""Run prompt v2 with neighbor windows plus one query-similar extra chunk."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_train_q000_siliconflow import (  # noqa: E402
    MAX_TOKENS,
    MODEL,
    TEMPERATURE,
    request_answer,
)
from scripts.run_train_context_window_ablation import (  # noqa: E402
    SAMPLE_SIZE,
    load_gold_after_generation,
    load_inputs,
    render_prompt,
)


TEMPLATE = (
    PROJECT_ROOT
    / "configs/generation/answer_with_citations_direct_conditions.prompt"
)
BASELINE = (
    PROJECT_ROOT
    / "experiments/generation/results/"
    "siliconflow_train_context_window_prompt_v2_seed42_20/generations.jsonl"
)
VECTORS = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/"
    "pplx-embed-v1-0.6b_fixed_600_0/embeddings.npy"
)
METADATA = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/"
    "bge-small-en-v1.5_fixed_600_0/chunk_metadata.jsonl"
)
PPLX_MODEL = "perplexity-ai/pplx-embed-v1-0.6b"
PPLX_REVISION = "2c4d510dd4a732063c31a0f70193e35067b51fd8"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/generation/results/"
    "siliconflow_train_context_window_prompt_v2_pplx_extra_seed42_20"
)
PREPARED = OUTPUT_DIR / "prepared_sources.jsonl"
GENERATIONS = OUTPUT_DIR / "generations.jsonl"
MANIFEST = OUTPUT_DIR / "manifest.json"
VARIANT = "representative_with_neighbors_plus_pplx_extra"
SEPARATOR = "\n\n[Additional query-similar excerpt from the same document]\n\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return args


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
    atomic_text(
        path,
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
    )


def baseline_sources() -> dict[str, list[dict[str, Any]]]:
    rows = list(jsonl_rows(BASELINE))
    if len(rows) != SAMPLE_SIZE:
        raise ValueError("expected 20 prompt-v2 baseline rows")
    output = {row["question_id"]: row["sources"] for row in rows}
    if len(output) != SAMPLE_SIZE:
        raise ValueError("duplicate prompt-v2 baseline question")
    return output


def source_texts_from_corpus(
    documents: dict[str, dict[str, Any]],
) -> dict[str, str]:
    return {document_id: document["text"] for document_id, document in documents.items()}


def prepare_sources(device_choice: str, batch_size: int) -> list[dict[str, Any]]:
    sample, rankings, documents = load_inputs()
    baseline = baseline_sources()
    expected_template = TEMPLATE.read_text(encoding="utf-8")

    # Re-render and compare the old neighbor evidence before adding anything.
    for query in sample:
        question_id = query["question_id"]
        _, sources = render_prompt(
            expected_template,
            query,
            rankings[question_id],
            documents,
            "representative_with_neighbors",
        )
        if sources != baseline[question_id]:
            raise ValueError(f"baseline neighbor evidence changed for {question_id}")

    metadata = list(jsonl_rows(METADATA))
    vectors = np.load(VECTORS, mmap_mode="r")
    if vectors.shape != (len(metadata), 1024) or vectors.dtype != np.float32:
        raise ValueError(f"unexpected cached PPLX vectors: {vectors.shape} {vectors.dtype}")

    required_documents = {
        source["document_id"]
        for sources in baseline.values()
        for source in sources
    }
    rows_by_document: dict[str, list[int]] = {
        document_id: [] for document_id in required_documents
    }
    for row_index, row in enumerate(metadata):
        document_id = row["document_id"]
        if document_id in rows_by_document:
            rows_by_document[document_id].append(row_index)
    if any(not rows for rows in rows_by_document.values()):
        raise ValueError("a Top-5 document is missing cached PPLX chunks")

    import torch
    from sentence_transformers import SentenceTransformer

    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    print(f"encoding 20 saved TRAIN queries on {device}", flush=True)
    model = SentenceTransformer(
        PPLX_MODEL,
        revision=PPLX_REVISION,
        trust_remote_code=True,
        local_files_only=True,
        device=device,
    )
    encoded = model.encode(
        [query["query"] for query in sample],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    del model

    document_texts = source_texts_from_corpus(documents)
    prepared: list[dict[str, Any]] = []
    for query_index, query in enumerate(sample):
        question_id = query["question_id"]
        augmented_sources: list[dict[str, Any]] = []
        for original in baseline[question_id]:
            source = dict(original)
            excluded = set(source["included_chunk_ids"])
            candidate_rows = [
                row_index
                for row_index in rows_by_document[source["document_id"]]
                if metadata[row_index]["chunk_id"] not in excluded
            ]
            source["neighbor_chunk_ids"] = list(source["included_chunk_ids"])
            if candidate_rows:
                scores = (
                    np.asarray(vectors[candidate_rows], dtype=np.float32)
                    @ encoded[query_index]
                )
                best_position = int(np.argmax(scores))
                best_row = candidate_rows[best_position]
                extra_chunk_id = metadata[best_row]["chunk_id"]
                chunk_index = int(extra_chunk_id.rsplit("::", 1)[1])
                body = document_texts[source["document_id"]][
                    chunk_index * 600 : (chunk_index + 1) * 600
                ]
                if not body:
                    raise ValueError(f"empty selected extra chunk: {extra_chunk_id}")
                source["extra_chunk_id"] = extra_chunk_id
                source["extra_chunk_score"] = float(scores[best_position])
                source["included_chunk_ids"] = [*source["included_chunk_ids"], extra_chunk_id]
                source["text"] = source["text"] + SEPARATOR + body
            else:
                source["extra_chunk_id"] = None
                source["extra_chunk_score"] = None
            augmented_sources.append(source)
        prepared.append(
            {
                "question_id": question_id,
                "query": query["query"],
                "sources": augmented_sources,
            }
        )

    atomic_jsonl(PREPARED, prepared)
    print(f"prepared={PREPARED}", flush=True)
    return prepared


def load_prepared() -> list[dict[str, Any]]:
    rows = list(jsonl_rows(PREPARED))
    sample, _, _ = load_inputs()
    expected_ids = [row["question_id"] for row in sample]
    if [row.get("question_id") for row in rows] != expected_ids:
        raise ValueError("prepared sources do not match the fixed sample")
    if any(row.get("query") != query["query"] for row, query in zip(rows, sample, strict=True)):
        raise ValueError("prepared query text changed")
    return rows


def render_prepared(template: str, row: dict[str, Any]) -> str:
    evidence = "\n\n".join(
        f"[{source['number']}]\nTitle: {source['title']}\n\n{source['text']}"
        for source in row["sources"]
    )
    prompt = template.replace("{{question}}", row["query"]).replace(
        "{{evidence}}", evidence
    )
    if "{{question}}" in prompt or "{{evidence}}" in prompt:
        raise ValueError("unresolved prompt placeholder")
    return prompt


def generate(prepared: list[dict[str, Any]]) -> int:
    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        print("SILICONFLOW_API_KEY is not set or is empty.", file=sys.stderr)
        return 2
    template = TEMPLATE.read_text(encoding="utf-8")
    prompt_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    existing: dict[str, dict[str, Any]] = {}
    if GENERATIONS.is_file():
        for row in jsonl_rows(GENERATIONS):
            question_id = row.get("question_id")
            if (
                question_id in existing
                or row.get("variant") != VARIANT
                or row.get("model") != MODEL
                or row.get("temperature") != TEMPERATURE
                or row.get("max_tokens") != MAX_TOKENS
                or row.get("prompt_sha256") != prompt_sha256
            ):
                raise ValueError("existing checkpoint is incompatible")
            existing[question_id] = row

    completed: list[dict[str, Any]] = []
    for row in prepared:
        question_id = row["question_id"]
        if question_id in existing:
            completed.append(existing[question_id])
            print(f"reused {len(completed)}/{SAMPLE_SIZE}: {question_id}", flush=True)
            continue
        try:
            answer = request_answer(api_key, render_prepared(template, row))
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
                "sources": row["sources"],
            }
        )
        atomic_jsonl(GENERATIONS, completed)
        print(f"completed {len(completed)}/{SAMPLE_SIZE}: {question_id}", flush=True)

    question_ids = {row["question_id"] for row in prepared}
    gold = load_gold_after_generation(question_ids)
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
            "question_ids": [row["question_id"] for row in prepared],
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "prompt_sha256": prompt_sha256,
            "prompt_template": str(TEMPLATE.relative_to(PROJECT_ROOT)),
            "retriever": "PPLX fixed_600_0 original Top-5",
            "variant": VARIANT,
            "extra_chunk_rule": "highest PPLX query dot product outside neighbor window",
            "pplx_model_revision": PPLX_REVISION,
            "baseline_passed": 9,
            "requests_completed": len(completed),
            "prepared_sources": str(PREPARED.relative_to(PROJECT_ROOT)),
            "generation_file": str(GENERATIONS.relative_to(PROJECT_ROOT)),
        },
    )
    print(f"generations={GENERATIONS}", flush=True)
    print(f"manifest={MANIFEST}", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    prepared = (
        prepare_sources(args.device, args.batch_size)
        if not PREPARED.is_file()
        else load_prepared()
    )
    if args.prepare_only:
        print("prepare-only complete", flush=True)
        return 0
    return generate(prepared)


if __name__ == "__main__":
    raise SystemExit(main())
