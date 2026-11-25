#!/usr/bin/env python3
"""Generate TRAIN answers from saved retrieval rankings."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_train_q000_siliconflow import (  # noqa: E402
    MAX_TOKENS,
    MODEL,
    RetryableSiliconFlowError,
    TEMPERATURE,
    request_completion,
)
from experiments.chunk.run_bm25_chunk_experiments import chunks_for  # noqa: E402
from experiments.chunk.run_bm25_parameter_grid import CONFIGS  # noqa: E402


QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
RAW_TRAIN = (
    PROJECT_ROOT / "data/raw/techqa/TechQA/training_and_dev/training_Q_A.json"
)
SAMPLE_MANIFEST = (
    PROJECT_ROOT
    / "experiments/generation/results/"
    "siliconflow_train_context_window_prompt_v2_seed42_20/manifest.json"
)
PROMPT_TEMPLATE = (
    PROJECT_ROOT / "configs/generation/answer_with_citations_direct_conditions.prompt"
)
LEGACY_V2_NEIGHBORS = (
    PROJECT_ROOT
    / "experiments/generation/results/"
    "siliconflow_train_context_window_prompt_v2_seed42_20/generations.jsonl"
)
RANKING_DEPTH = 5
SAMPLE_SIZE = 20
EXPECTED_TRAIN_QUERIES = 450
TOTAL_CONTEXT_TOKEN_LIMIT = 32_768
INPUT_TOKEN_LIMIT = TOTAL_CONTEXT_TOKEN_LIMIT - MAX_TOKENS
VARIANTS = ("representative_only", "representative_with_neighbors")
DEFAULT_API_KEY_FILE = Path.home() / ".config/enterprise-rag/siliconflow.env"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--chunk-metadata", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--chunk-overlap", type=int, required=True)
    parser.add_argument("--chunk-config", required=True)
    parser.add_argument("--context-mode", choices=VARIANTS, required=True)
    parser.add_argument(
        "--question-scope",
        choices=("seed42_20", "all_train"),
        default="seed42_20",
        help="Generate the saved 20-case sample or all 450 TRAIN questions.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        help="Generate only this TRAIN question; repeat to select more than one.",
    )
    parser.add_argument(
        "--question-id-file",
        type=Path,
        help=(
            "JSON file containing a question_ids list. Cannot be combined with "
            "--question-id; the saved order is preserved."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-template",
        type=Path,
        default=PROMPT_TEMPLATE,
        help="Prompt template containing exactly one {{question}} and {{evidence}}.",
    )
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=DEFAULT_API_KEY_FILE,
        help=(
            "Local env file used only when SILICONFLOW_API_KEY is absent from the "
            "process environment."
        ),
    )
    parser.add_argument(
        "--no-legacy-reuse",
        action="store_true",
        help="Do not import a compatible saved 600/0 prompt-v2 neighbor answer.",
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=10.0,
        help="Initial wait after a transient API failure; retries have no limit.",
    )
    parser.add_argument(
        "--max-retry-delay-seconds",
        type=float,
        default=60.0,
        help="Maximum wait between unlimited transient-error retries.",
    )
    parser.add_argument(
        "--hard-request-timeout-seconds",
        type=float,
        default=300.0,
        help=(
            "Hard wall-clock limit for one API attempt. Exceeding it cancels only "
            "that attempt and retries without ending the generation job."
        ),
    )
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must satisfy 0 <= overlap < size")
    if args.retry_delay_seconds <= 0:
        parser.error("--retry-delay-seconds must be positive")
    if args.max_retry_delay_seconds < args.retry_delay_seconds:
        parser.error("--max-retry-delay-seconds must be >= --retry-delay-seconds")
    if args.hard_request_timeout_seconds <= 0:
        parser.error("--hard-request-timeout-seconds must be positive")
    if args.question_id and args.question_id_file:
        parser.error("--question-id and --question-id-file cannot be combined")
    return args


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_api_key(config_path: Path) -> str:
    value = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if value:
        return value
    if not config_path.is_file():
        return ""
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, separator, raw_value = line.partition("=")
        if separator and key.strip() == "SILICONFLOW_API_KEY":
            return raw_value.strip().strip("'\"").strip()
    return ""


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


@contextlib.contextmanager
def hard_request_deadline(seconds: float):
    """Interrupt one stuck request without terminating the surrounding job."""
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum, _frame):
        raise TimeoutError(f"hard request deadline exceeded after {seconds:g}s")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def load_sample_ids() -> list[str]:
    manifest = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    ids = manifest.get("question_ids")
    if not isinstance(ids, list) or len(ids) != SAMPLE_SIZE or len(set(ids)) != SAMPLE_SIZE:
        raise ValueError("saved seed-42 manifest does not contain 20 distinct IDs")
    return ids


def load_question_id_file(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    ids = value.get("question_ids") if isinstance(value, dict) else value
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(question_id, str) or not question_id for question_id in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError(f"{path} does not contain distinct question_ids")
    return ids


def select_question_ids(
    scope: str,
    ordered_queries: list[dict[str, Any]],
    explicit_ids: list[str] | None = None,
) -> list[str]:
    if explicit_ids:
        known = {row.get("question_id") for row in ordered_queries}
        if len(set(explicit_ids)) != len(explicit_ids) or not set(explicit_ids) <= known:
            raise ValueError("--question-id contains an unknown or duplicate TRAIN ID")
        return explicit_ids
    if scope == "seed42_20":
        return load_sample_ids()
    if len(ordered_queries) != EXPECTED_TRAIN_QUERIES:
        raise ValueError(
            f"expected {EXPECTED_TRAIN_QUERIES} TRAIN queries, "
            f"got {len(ordered_queries)}"
        )
    ids = [row.get("question_id") for row in ordered_queries]
    if any(not isinstance(question_id, str) for question_id in ids):
        raise ValueError("a TRAIN query has an invalid question_id")
    if len(set(ids)) != EXPECTED_TRAIN_QUERIES:
        raise ValueError("TRAIN question IDs are not unique")
    return ids


def load_chunk_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in jsonl_rows(path):
        chunk_id = row.get("chunk_id")
        if not isinstance(chunk_id, str) or chunk_id in rows:
            raise ValueError("invalid or duplicate chunk metadata")
        rows[chunk_id] = row
    if not rows:
        raise ValueError("chunk metadata is empty")
    return rows


def selected_indices(
    representative_index: int,
    document_text_length: int,
    size: int,
    overlap: int,
    mode: str,
) -> list[int]:
    step = size - overlap
    chunk_count = 0
    for start in range(0, document_text_length, step):
        chunk_count += 1
        if start + size >= document_text_length:
            break
    if representative_index < 0 or representative_index >= chunk_count:
        raise ValueError(f"representative chunk index out of range: {representative_index}")
    if mode == "representative_only":
        return [representative_index]
    return list(
        range(max(0, representative_index - 1), min(chunk_count, representative_index + 2))
    )


def source_for(
    number: int,
    candidate: dict[str, Any],
    document: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    size: int,
    overlap: int,
    mode: str,
    chunk_config: str,
) -> dict[str, Any]:
    document_id = candidate["document_id"]
    chunk_id = candidate["representative_chunk_id"]
    if not chunk_id.startswith(document_id + "::") or chunk_id not in metadata:
        raise ValueError(f"unknown representative chunk: {chunk_id}")
    representative_index = int(chunk_id.rsplit("::", 1)[1])
    document_text = document["text"]
    if chunk_config not in CONFIGS:
        raise ValueError(f"unknown chunk config: {chunk_config}")
    actual_chunks = list(chunks_for(document_text, CONFIGS[chunk_config]))
    if representative_index < 0 or representative_index >= len(actual_chunks):
        raise ValueError(f"representative chunk index out of range: {representative_index}")
    if metadata[chunk_id].get("body_length") != len(actual_chunks[representative_index]):
        raise ValueError(f"reconstructed chunk differs from metadata: {chunk_id}")
    if mode == "representative_only":
        return {
            "number": number,
            "document_id": document_id,
            "document_rank": candidate["rank"],
            "document_score": candidate["score"],
            "representative_chunk_id": chunk_id,
            "included_chunk_ids": [chunk_id],
            "source_start": None,
            "source_end": None,
            "title": document["title"],
            "text": actual_chunks[representative_index],
        }
    config = CONFIGS[chunk_config]
    if config.strategy == "paragraph_merge":
        indices = list(
            range(
                max(0, representative_index - 1),
                min(len(actual_chunks), representative_index + 2),
            )
        )
    else:
        indices = selected_indices(
            representative_index, len(document_text), size, overlap, mode
        )
    step = size - overlap
    included_chunk_ids = [f"{document_id}::{index:06d}" for index in indices]
    for expected_id in included_chunk_ids:
        if expected_id not in metadata:
            raise ValueError(f"adjacent chunk missing from metadata: {expected_id}")

    if config.strategy == "paragraph_merge":
        for index, expected_id in zip(indices, included_chunk_ids):
            if metadata[expected_id].get("body_length") != len(actual_chunks[index]):
                raise ValueError(f"reconstructed chunk differs from metadata: {expected_id}")
        text = "\n\n".join(actual_chunks[index] for index in indices)
        if not text:
            raise ValueError(f"empty source text for {chunk_id}")
        return {
            "number": number,
            "document_id": document_id,
            "document_rank": candidate["rank"],
            "document_score": candidate["score"],
            "representative_chunk_id": chunk_id,
            "included_chunk_ids": included_chunk_ids,
            "source_start": None,
            "source_end": None,
            "title": document["title"],
            "text": text,
        }

    # Read one contiguous original-text interval. This preserves source order while
    # removing the repeated overlap between fixed chunks.
    start = indices[0] * step
    end = min(len(document_text), indices[-1] * step + size)
    text = document_text[start:end]
    if not text:
        raise ValueError(f"empty source text for {chunk_id}")
    return {
        "number": number,
        "document_id": document_id,
        "document_rank": candidate["rank"],
        "document_score": candidate["score"],
        "representative_chunk_id": chunk_id,
        "included_chunk_ids": included_chunk_ids,
        "source_start": start,
        "source_end": end,
        "title": document["title"],
        "text": text,
    }


def normalize_ranking_row(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("top_documents"), list):
        return row
    normalized: list[dict[str, Any]] = []
    if isinstance(row.get("fused_documents"), list):
        for candidate in row["fused_documents"]:
            chunk_id = candidate.get("pplx_representative_chunk_id") or candidate.get(
                "bm25_representative_chunk_id"
            )
            normalized.append(
                {
                    "document_id": candidate.get("document_id"),
                    "rank": candidate.get("rank"),
                    "score": candidate.get("rrf_score"),
                    "representative_chunk_id": chunk_id,
                }
            )
    elif isinstance(row.get("reranked_documents"), list):
        for candidate in row["reranked_documents"]:
            normalized.append(
                {
                    "document_id": candidate.get("document_id"),
                    "rank": candidate.get("rank"),
                    "score": candidate.get("rerank_score"),
                    "representative_chunk_id": candidate.get("scored_chunk_id"),
                }
            )
    else:
        raise ValueError(f"unsupported ranking format for {row.get('question_id')}")
    if not normalized or any(
        not isinstance(candidate.get("document_id"), str)
        or not isinstance(candidate.get("representative_chunk_id"), str)
        or not isinstance(candidate.get("rank"), int)
        for candidate in normalized
    ):
        raise ValueError(f"invalid normalized ranking for {row.get('question_id')}")
    return {**row, "top_documents": normalized}


def render_prompt(template: str, question: str, sources: list[dict[str, Any]]) -> str:
    evidence = "\n\n".join(
        f"[{source['number']}]\nTitle: {source['title']}\n\n{source['text']}"
        for source in sources
    )
    prompt = template.replace("{{question}}", question).replace("{{evidence}}", evidence)
    if "{{question}}" in prompt or "{{evidence}}" in prompt:
        raise ValueError("rendered prompt contains unresolved placeholders")
    return prompt


def citation_map(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numbers: list[int] = []
    for match in re.findall(r"\[(\d+)\]", answer):
        number = int(match)
        if 1 <= number <= len(sources) and number not in numbers:
            numbers.append(number)
    by_number = {source["number"]: source for source in sources}
    return [
        {
            "number": number,
            "document_id": by_number[number]["document_id"],
            "representative_chunk_id": by_number[number]["representative_chunk_id"],
            "included_chunk_ids": by_number[number]["included_chunk_ids"],
        }
        for number in numbers
    ]


def legacy_rows() -> dict[str, dict[str, Any]]:
    if not LEGACY_V2_NEIGHBORS.is_file():
        return {}
    output: dict[str, dict[str, Any]] = {}
    for row in jsonl_rows(LEGACY_V2_NEIGHBORS):
        question_id = row.get("question_id")
        if question_id in output:
            raise ValueError("duplicate legacy prompt-v2 checkpoint")
        output[question_id] = row
    return output


def row_is_compatible(
    row: dict[str, Any],
    question_id: str,
    mode: str,
    prompt_sha256: str,
    sources: list[dict[str, Any]],
) -> bool:
    comparable_keys = (
        "number",
        "document_id",
        "representative_chunk_id",
        "included_chunk_ids",
        "title",
        "text",
    )
    comparable_sources = [
        {
            key: source[key] for key in comparable_keys
        }
        for source in sources
    ]
    saved_sources = row.get("sources")
    if not isinstance(saved_sources, list):
        return False
    try:
        comparable_saved_sources = [
            {key: source[key] for key in comparable_keys} for source in saved_sources
        ]
    except (KeyError, TypeError):
        return False
    return all(
        (
            row.get("question_id") == question_id,
            row.get("variant") == mode,
            row.get("model") == MODEL,
            row.get("temperature", TEMPERATURE) == TEMPERATURE,
            row.get("max_tokens", MAX_TOKENS) == MAX_TOKENS,
            row.get("prompt_sha256") == prompt_sha256,
            comparable_saved_sources == comparable_sources,
            isinstance(row.get("answer"), str) and bool(row["answer"].strip()),
        )
    )


def main() -> int:
    args = parse_args()
    rankings_path = resolve(args.rankings)
    metadata_path = resolve(args.chunk_metadata)
    prompt_template_path = resolve(args.prompt_template)
    output_dir = resolve(args.output_dir)
    generations_path = output_dir / "generations.jsonl"
    contexts_path = output_dir / "contexts.jsonl"
    manifest_path = output_dir / "manifest.json"

    template = prompt_template_path.read_text(encoding="utf-8")
    if template.count("{{question}}") != 1 or template.count("{{evidence}}") != 1:
        raise ValueError("prompt-v2 template must contain exactly two placeholders")
    prompt_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()
    ordered_queries = list(jsonl_rows(QUERIES))
    question_ids = select_question_ids(
        args.question_scope,
        ordered_queries,
        (
            load_question_id_file(resolve(args.question_id_file))
            if args.question_id_file
            else args.question_id
        ),
    )
    question_set = set(question_ids)
    expected_count = len(question_ids)
    queries = {
        row["question_id"]: row
        for row in ordered_queries
        if row["question_id"] in question_set
    }
    rankings = {
        row["question_id"]: normalize_ranking_row(row)
        for row in jsonl_rows(rankings_path)
        if row["question_id"] in question_set
    }
    if set(queries) != question_set or set(rankings) != question_set:
        raise ValueError("queries/rankings do not cover the selected TRAIN scope")
    metadata = load_chunk_metadata(metadata_path)
    needed_document_ids = {
        candidate["document_id"]
        for question_id in question_ids
        for candidate in rankings[question_id]["top_documents"][:RANKING_DEPTH]
    }
    documents = {
        row["document_id"]: row
        for row in jsonl_rows(CORPUS)
        if row["document_id"] in needed_document_ids
    }
    if set(documents) != needed_document_ids:
        raise ValueError("a Top-5 document is absent from the shared corpus")

    existing: dict[str, dict[str, Any]] = {}
    if generations_path.is_file():
        for row in jsonl_rows(generations_path):
            question_id = row.get("question_id")
            if question_id not in question_set or question_id in existing:
                raise ValueError("invalid generation checkpoint")
            existing[question_id] = row
    legacy = (
        legacy_rows()
        if args.chunk_config == "fixed_600_0"
        and args.context_mode == "representative_with_neighbors"
        and not args.no_legacy_reuse
        else {}
    )

    prepared: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    context_rows: list[dict[str, Any]] = []
    for question_id in question_ids:
        query = queries[question_id]
        ranking = rankings[question_id]
        candidates = ranking["top_documents"][:RANKING_DEPTH]
        if len(candidates) != RANKING_DEPTH or len(
            {candidate["document_id"] for candidate in candidates}
        ) != RANKING_DEPTH:
            raise ValueError(f"invalid Top-5 for {question_id}")
        sources = [
            source_for(
                number,
                candidate,
                documents[candidate["document_id"]],
                metadata,
                args.chunk_size,
                args.chunk_overlap,
                args.context_mode,
                args.chunk_config,
            )
            for number, candidate in enumerate(candidates, 1)
        ]
        prompt = render_prompt(template, query["query"], sources)
        prompt_utf8_bytes = len(prompt.encode("utf-8"))
        # A UTF-8 byte is a conservative upper bound on BPE token count. The API's
        # exact prompt_tokens value is recorded after each newly generated answer.
        if prompt_utf8_bytes > INPUT_TOKEN_LIMIT:
            raise ValueError(
                f"{question_id} exceeds the shared input limit: "
                f"utf8_bytes={prompt_utf8_bytes} > {INPUT_TOKEN_LIMIT}"
            )
        prepared[question_id] = (prompt, sources)
        context_rows.append(
            {
                "question_id": question_id,
                "chunk_config": args.chunk_config,
                "context_mode": args.context_mode,
                "question": query["query"],
                "prompt": prompt,
                "prompt_character_count": len(prompt),
                "prompt_utf8_bytes": prompt_utf8_bytes,
                "evidence_character_count": sum(len(source["text"]) for source in sources),
                "sources": sources,
            }
        )
    atomic_jsonl(contexts_path, context_rows)

    api_key = load_api_key(args.api_key_file)
    rows: list[dict[str, Any]] = []
    for question_id in question_ids:
        prompt, sources = prepared[question_id]
        checkpoint = existing.get(question_id)
        if checkpoint is not None:
            if not row_is_compatible(
                checkpoint, question_id, args.context_mode, prompt_sha256, sources
            ):
                raise ValueError(f"incompatible checkpoint for {question_id}")
            rows.append(checkpoint)
            print(
                f"reused checkpoint {len(rows)}/{expected_count}: {question_id}",
                flush=True,
            )
            continue
        saved = legacy.get(question_id)
        if saved is not None and row_is_compatible(
            saved, question_id, args.context_mode, prompt_sha256, sources
        ):
            answer = saved["answer"]
            row = {
                "question_id": question_id,
                "chunk_config": args.chunk_config,
                "variant": args.context_mode,
                "model": MODEL,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "total_context_token_limit": TOTAL_CONTEXT_TOKEN_LIMIT,
                "prompt_sha256": prompt_sha256,
                "prompt": prompt,
                "answer": answer,
                "sources": sources,
                "citation_map": citation_map(answer, sources),
                "usage": {"prompt_tokens": None, "note": "legacy API usage was not saved"},
                "reused_from": str(LEGACY_V2_NEIGHBORS.relative_to(PROJECT_ROOT)),
            }
            rows.append(row)
            atomic_jsonl(generations_path, rows)
            print(
                f"imported legacy {len(rows)}/{expected_count}: {question_id}",
                flush=True,
            )
            continue
        if not api_key:
            atomic_jsonl(generations_path, rows)
            print(
                "SILICONFLOW_API_KEY is required for missing case "
                f"{question_id}; environment and {args.api_key_file} were empty.",
                file=sys.stderr,
            )
            return 2
        retry_count = 0
        while True:
            try:
                with hard_request_deadline(args.hard_request_timeout_seconds):
                    answer, usage = request_completion(
                        api_key,
                        prompt,
                        timeout=max(1, int(args.hard_request_timeout_seconds)),
                    )
                break
            except RetryableSiliconFlowError as exc:
                retry_count += 1
                delay = min(
                    args.max_retry_delay_seconds,
                    args.retry_delay_seconds * (2 ** min(retry_count - 1, 6)),
                )
                print(
                    f"Transient failure at {question_id}; retry={retry_count} "
                    f"in {delay:g}s: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
            except RuntimeError as exc:
                retry_count += 1
                delay = min(
                    args.max_retry_delay_seconds,
                    args.retry_delay_seconds * (2 ** min(retry_count - 1, 6)),
                )
                print(
                    f"API failure at {question_id}; retry={retry_count} "
                    f"in {delay:g}s (Ctrl+C to stop): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        row = {
            "question_id": question_id,
            "chunk_config": args.chunk_config,
            "variant": args.context_mode,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "total_context_token_limit": TOTAL_CONTEXT_TOKEN_LIMIT,
            "prompt_sha256": prompt_sha256,
            "prompt": prompt,
            "answer": answer,
            "sources": sources,
            "citation_map": citation_map(answer, sources),
            "usage": usage,
            "reused_from": None,
        }
        rows.append(row)
        atomic_jsonl(generations_path, rows)
        print(f"generated {len(rows)}/{expected_count}: {question_id}", flush=True)

    gold_records = json.loads(RAW_TRAIN.read_text(encoding="utf-8"))
    gold = {
        row["QUESTION_ID"]: row["ANSWER"]
        for row in gold_records
        if row.get("QUESTION_ID") in question_set
    }
    if set(gold) != question_set:
        raise ValueError("TRAIN labels are incomplete")
    for row in rows:
        row["gold_answer"] = gold[row["question_id"]]
    atomic_jsonl(generations_path, rows)
    exact_prompt_tokens = [
        row.get("usage", {}).get("prompt_tokens")
        for row in rows
        if isinstance(row.get("usage", {}).get("prompt_tokens"), int)
    ]
    atomic_json(
        manifest_path,
        {
            "status": "complete_pending_review",
            "evaluation_split": "TRAIN only",
            "question_scope": args.question_scope,
            "seed": 42 if args.question_scope == "seed42_20" else None,
            "question_count": expected_count,
            "question_ids": question_ids,
            "chunk_config": args.chunk_config,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "context_mode": args.context_mode,
            "ranking_file": str(rankings_path),
            "chunk_metadata": str(metadata_path),
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "total_context_token_limit": TOTAL_CONTEXT_TOKEN_LIMIT,
            "input_token_limit": INPUT_TOKEN_LIMIT,
            "input_limit_preflight": "UTF-8 bytes <= input token limit (conservative)",
            "hard_request_timeout_seconds": args.hard_request_timeout_seconds,
            "api_retry_limit": None,
            "prompt_template": str(prompt_template_path),
            "prompt_sha256": prompt_sha256,
            "completed": len(rows),
            "exact_api_prompt_tokens_available": len(exact_prompt_tokens),
            "max_exact_api_prompt_tokens": max(exact_prompt_tokens, default=None),
            "generations": str(generations_path),
            "contexts": str(contexts_path),
        },
    )
    print(f"generations={generations_path}", flush=True)
    print(f"contexts={contexts_path}", flush=True)
    print(f"manifest={manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
