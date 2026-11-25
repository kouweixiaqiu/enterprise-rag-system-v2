#!/usr/bin/env python3
"""Generate refusal decisions for all 150 unanswerable TechQA TRAIN questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import jsonl_rows  # noqa: E402
from scripts.generate_train_q000_siliconflow import (  # noqa: E402
    MAX_TOKENS,
    MODEL,
    RetryableSiliconFlowError,
    TEMPERATURE,
    request_completion,
)
from scripts.run_train_generation_from_rankings import (  # noqa: E402
    INPUT_TOKEN_LIMIT,
    TOTAL_CONTEXT_TOKEN_LIMIT,
    citation_map,
    hard_request_deadline,
    load_api_key,
    load_chunk_metadata,
    render_prompt,
    source_for,
)


RAW_TRAIN = (
    PROJECT_ROOT / "data/raw/techqa/TechQA/training_and_dev/training_Q_A.json"
)
CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
RANKINGS = (
    PROJECT_ROOT
    / "experiments/dense/results/"
    "pplx_embed_v1_0.6b_fixed_600_0_train_unanswerable/"
    "dense_rankings_top100.jsonl"
)
CHUNK_METADATA = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/"
    "bge-small-en-v1.5_fixed_600_0/chunk_metadata.jsonl"
)
PROMPT_TEMPLATE = (
    PROJECT_ROOT / "configs/generation/answer_with_citations_refusal.prompt"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/generation/results/"
    "siliconflow_train150_unanswerable_pplx_fixed_600_0_top5_neighbors_refusal"
)
DEFAULT_API_KEY_FILE = Path.home() / ".config/enterprise-rag/siliconflow.env"
EXPECTED_QUESTIONS = 150
RANKING_DEPTH = 5
CHUNK_CONFIG = "fixed_600_0"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 0
CONTEXT_MODE = "representative_with_neighbors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    parser.add_argument("--max-retry-delay-seconds", type=float, default=60.0)
    parser.add_argument("--hard-request-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and validate every context without calling SiliconFlow.",
    )
    args = parser.parse_args()
    if args.retry_delay_seconds <= 0:
        parser.error("--retry-delay-seconds must be positive")
    if args.max_retry_delay_seconds < args.retry_delay_seconds:
        parser.error("--max-retry-delay-seconds must be >= --retry-delay-seconds")
    if args.hard_request_timeout_seconds <= 0:
        parser.error("--hard-request-timeout-seconds must be positive")
    return args


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


class RunLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.stream.write(line + "\n")

    def close(self) -> None:
        self.stream.close()


def build_query(record: dict[str, Any], index: int) -> str:
    title = record.get("QUESTION_TITLE")
    body = record.get("QUESTION_TEXT")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"unanswerable TRAIN record {index} has no title")
    if body is None:
        body = ""
    if not isinstance(body, str):
        raise ValueError(f"unanswerable TRAIN record {index} has invalid body")
    title, body = title.strip(), body.strip()
    return f"{title}\n\n{body}" if body else title


def load_questions() -> list[dict[str, str]]:
    records = json.loads(RAW_TRAIN.read_text(encoding="utf-8"))
    questions: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or record.get("ANSWERABLE") != "N":
            continue
        question_id = record.get("QUESTION_ID")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"unanswerable TRAIN record {index} has no question ID")
        if record.get("DOCUMENT") != "-" or record.get("ANSWER") != "-":
            raise ValueError(f"{question_id} has inconsistent unanswerable labels")
        questions.append(
            {"question_id": question_id, "question": build_query(record, index)}
        )
    ids = [row["question_id"] for row in questions]
    if len(questions) != EXPECTED_QUESTIONS or len(set(ids)) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"expected {EXPECTED_QUESTIONS} distinct unanswerable TRAIN questions, "
            f"got {len(questions)} rows and {len(set(ids))} IDs"
        )
    return questions


def unique_rankings() -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in jsonl_rows(RANKINGS):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or question_id in output:
            raise ValueError(f"invalid or duplicate ranking question ID: {question_id!r}")
        candidates = row.get("top_documents")
        if not isinstance(candidates, list) or len(candidates) < RANKING_DEPTH:
            raise ValueError(f"ranking has fewer than Top-5 documents: {question_id}")
        output[question_id] = row
    return output


def parse_decision(answer: str) -> str:
    first_line = next((line.strip() for line in answer.splitlines() if line.strip()), "")
    match = re.fullmatch(r"DECISION:\s*(ANSWER|INSUFFICIENT_CONTEXT)", first_line)
    return match.group(1) if match else "INVALID_FORMAT"


def checkpoint_compatible(
    row: dict[str, Any], prompt_sha256: str, sources: list[dict[str, Any]]
) -> bool:
    return all(
        (
            row.get("model") == MODEL,
            row.get("temperature") == TEMPERATURE,
            row.get("max_tokens") == MAX_TOKENS,
            row.get("prompt_sha256") == prompt_sha256,
            row.get("sources") == sources,
            isinstance(row.get("answer"), str) and bool(row["answer"].strip()),
        )
    )


def write_manifest(
    path: Path,
    *,
    status: str,
    question_ids: list[str],
    prompt_sha256: str,
    completed: int,
    decision_counts: dict[str, int],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    atomic_json(
        path,
        {
            "status": status,
            "evaluation_split": "TRAIN ANSWERABLE=N",
            "expected_decision": "INSUFFICIENT_CONTEXT",
            "question_count": len(question_ids),
            "question_ids": question_ids,
            "retriever": "PPLX fixed_600_0 original Top-5",
            "context_mode": CONTEXT_MODE,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "prompt_template": str(PROMPT_TEMPLATE.relative_to(PROJECT_ROOT)),
            "prompt_sha256": prompt_sha256,
            "completed": completed,
            "decision_counts": decision_counts,
            "api_retry_limit": None,
            "hard_request_timeout_seconds": args.hard_request_timeout_seconds,
            "rankings": str(RANKINGS.relative_to(PROJECT_ROOT)),
            "contexts": str((output_dir / "contexts.jsonl").relative_to(PROJECT_ROOT)),
            "generations": str(
                (output_dir / "generations.jsonl").relative_to(PROJECT_ROOT)
            ),
        },
    )


def run(args: argparse.Namespace, logger: RunLogger) -> int:
    output_dir = args.output_dir.resolve()
    contexts_path = output_dir / "contexts.jsonl"
    generations_path = output_dir / "generations.jsonl"
    manifest_path = output_dir / "manifest.json"
    template = PROMPT_TEMPLATE.read_text(encoding="utf-8")
    if template.count("{{question}}") != 1 or template.count("{{evidence}}") != 1:
        raise ValueError("refusal prompt must contain one question and one evidence placeholder")
    prompt_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest()

    questions = load_questions()
    question_ids = [row["question_id"] for row in questions]
    rankings = unique_rankings()
    if set(rankings) != set(question_ids):
        raise ValueError(
            f"question/ranking IDs differ: missing={sorted(set(question_ids)-set(rankings))}, "
            f"extra={sorted(set(rankings)-set(question_ids))}"
        )
    metadata = load_chunk_metadata(CHUNK_METADATA)
    needed_document_ids = {
        candidate["document_id"]
        for question_id in question_ids
        for candidate in rankings[question_id]["top_documents"][:RANKING_DEPTH]
    }
    documents = {
        row["document_id"]: row
        for row in jsonl_rows(CORPUS)
        if row.get("document_id") in needed_document_ids
    }
    if set(documents) != needed_document_ids:
        raise ValueError(f"candidate documents missing: {sorted(needed_document_ids-set(documents))}")

    prepared: dict[str, tuple[str, list[dict[str, Any]]]] = {}
    contexts: list[dict[str, Any]] = []
    for question in questions:
        question_id = question["question_id"]
        candidates = rankings[question_id]["top_documents"][:RANKING_DEPTH]
        if len({candidate.get("document_id") for candidate in candidates}) != RANKING_DEPTH:
            raise ValueError(f"Top-5 documents are not distinct for {question_id}")
        sources = [
            source_for(
                number,
                candidate,
                documents[candidate["document_id"]],
                metadata,
                CHUNK_SIZE,
                CHUNK_OVERLAP,
                CONTEXT_MODE,
                CHUNK_CONFIG,
            )
            for number, candidate in enumerate(candidates, 1)
        ]
        prompt = render_prompt(template, question["question"], sources)
        prompt_bytes = len(prompt.encode("utf-8"))
        if prompt_bytes > INPUT_TOKEN_LIMIT:
            raise ValueError(
                f"{question_id} exceeds input limit: {prompt_bytes} > {INPUT_TOKEN_LIMIT}"
            )
        prepared[question_id] = (prompt, sources)
        contexts.append(
            {
                "question_id": question_id,
                "answerable_label": "N",
                "question": question["question"],
                "context_mode": CONTEXT_MODE,
                "top_documents": candidates,
                "sources": sources,
                "prompt": prompt,
                "prompt_character_count": len(prompt),
                "prompt_utf8_bytes": prompt_bytes,
                "evidence_character_count": sum(len(source["text"]) for source in sources),
            }
        )
    atomic_jsonl(contexts_path, contexts)
    logger.log(f"validated and saved {len(contexts)}/{EXPECTED_QUESTIONS} contexts")
    if args.prepare_only:
        write_manifest(
            manifest_path,
            status="contexts_prepared_no_api_calls",
            question_ids=question_ids,
            prompt_sha256=prompt_sha256,
            completed=0,
            decision_counts={},
            output_dir=output_dir,
            args=args,
        )
        return 0

    existing: dict[str, dict[str, Any]] = {}
    if generations_path.is_file():
        for row in jsonl_rows(generations_path):
            question_id = row.get("question_id")
            if question_id not in prepared or question_id in existing:
                raise ValueError(f"invalid generation checkpoint: {question_id!r}")
            if not checkpoint_compatible(row, prompt_sha256, prepared[question_id][1]):
                raise ValueError(f"incompatible generation checkpoint: {question_id}")
            existing[question_id] = row

    api_key = load_api_key(args.api_key_file)
    if not api_key and len(existing) != EXPECTED_QUESTIONS:
        raise RuntimeError(
            f"SILICONFLOW_API_KEY missing from environment and {args.api_key_file}"
        )
    rows = dict(existing)
    logger.log(
        f"starting refusal generation: total={EXPECTED_QUESTIONS} "
        f"reused={len(existing)} remaining={EXPECTED_QUESTIONS-len(existing)}"
    )
    for index, question_id in enumerate(question_ids, 1):
        if question_id in rows:
            logger.log(f"reused {index}/{EXPECTED_QUESTIONS}: {question_id}")
            continue
        prompt, sources = prepared[question_id]
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
                logger.log(
                    f"transient failure at {question_id}; retry={retry_count} "
                    f"in {delay:g}s: {exc}"
                )
                time.sleep(delay)
            except TimeoutError as exc:
                retry_count += 1
                delay = min(
                    args.max_retry_delay_seconds,
                    args.retry_delay_seconds * (2 ** min(retry_count - 1, 6)),
                )
                logger.log(
                    f"hard timeout at {question_id}; retry={retry_count} "
                    f"in {delay:g}s: {exc}"
                )
                time.sleep(delay)
        decision = parse_decision(answer)
        rows[question_id] = {
            "question_id": question_id,
            "answerable_label": "N",
            "expected_decision": "INSUFFICIENT_CONTEXT",
            "decision": decision,
            "chunk_config": CHUNK_CONFIG,
            "context_mode": CONTEXT_MODE,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "total_context_token_limit": TOTAL_CONTEXT_TOKEN_LIMIT,
            "prompt_sha256": prompt_sha256,
            "answer": answer,
            "sources": sources,
            "citation_map": citation_map(answer, sources),
            "usage": usage,
            "retry_count": retry_count,
        }
        ordered_rows = [rows[qid] for qid in question_ids if qid in rows]
        atomic_jsonl(generations_path, ordered_rows)
        counts = {
            name: sum(row["decision"] == name for row in ordered_rows)
            for name in ("ANSWER", "INSUFFICIENT_CONTEXT", "INVALID_FORMAT")
        }
        write_manifest(
            manifest_path,
            status="running",
            question_ids=question_ids,
            prompt_sha256=prompt_sha256,
            completed=len(ordered_rows),
            decision_counts=counts,
            output_dir=output_dir,
            args=args,
        )
        logger.log(
            f"generated {len(ordered_rows)}/{EXPECTED_QUESTIONS}: "
            f"{question_id} decision={decision}"
        )

    ordered_rows = [rows[qid] for qid in question_ids]
    counts = {
        name: sum(row["decision"] == name for row in ordered_rows)
        for name in ("ANSWER", "INSUFFICIENT_CONTEXT", "INVALID_FORMAT")
    }
    summary = {
        "status": "complete_pending_review",
        "question_count": EXPECTED_QUESTIONS,
        "expected_decision": "INSUFFICIENT_CONTEXT",
        "decision_counts": counts,
        "automatic_refusal_rate": counts["INSUFFICIENT_CONTEXT"] / EXPECTED_QUESTIONS,
        "automatic_answer_rate": counts["ANSWER"] / EXPECTED_QUESTIONS,
        "note": (
            "This is an output-contract count. Final refusal quality still requires "
            "checking that answers do not contain unsupported operational guidance."
        ),
    }
    atomic_json(output_dir / "summary.json", summary)
    write_manifest(
        manifest_path,
        status="complete_pending_review",
        question_ids=question_ids,
        prompt_sha256=prompt_sha256,
        completed=EXPECTED_QUESTIONS,
        decision_counts=counts,
        output_dir=output_dir,
        args=args,
    )
    logger.log(f"complete: decisions={counts} output={output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    args.output_dir = (
        args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    )
    logger = RunLogger(args.output_dir / "run.log")
    exit_code_path = args.output_dir / "exit_code.txt"
    try:
        code = run(args, logger)
    except KeyboardInterrupt:
        logger.log("interrupted by user")
        code = 130
    except Exception:
        logger.log("FAILED\n" + traceback.format_exc().rstrip())
        code = 1
    atomic_text(exit_code_path, f"{code}\n")
    logger.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
