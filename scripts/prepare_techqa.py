#!/usr/bin/env python3
"""Prepare deterministic TechQA retrieval inputs from the official raw JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data/raw/techqa/TechQA/training_and_dev"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/processed/techqa"

TRAIN_SOURCE = "training_Q_A.json"
DEV_SOURCE = "dev_Q_A.json"
CORPUS_SOURCE = "training_dev_technotes.json"

TRAIN_OUTPUT = "train_queries.jsonl"
DEV_OUTPUT = "dev_queries.jsonl"
CORPUS_OUTPUT = "corpus.jsonl"
EXCLUDED_DEV_OUTPUT = "excluded_dev.jsonl"
SUMMARY_OUTPUT = "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare answerable TechQA queries and a shared Technote corpus."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing the three official JSON files (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated JSONL files (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


def load_json(path: Path, expected_type: type) -> Any:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, expected_type):
        raise ValueError(
            f"Expected {path} to contain {expected_type.__name__}, "
            f"found {type(value).__name__}"
        )
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_string(record: dict[str, Any], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} has missing or empty {field}")
    return value


def duplicate_key(record: dict[str, Any]) -> tuple[Any, Any]:
    """Return the untouched title/body pair used for cross-split comparison."""
    return record.get("QUESTION_TITLE"), record.get("QUESTION_TEXT")


def build_query(record: dict[str, Any], context: str) -> str:
    title = required_string(record, "QUESTION_TITLE", context).strip()
    body_value = record.get("QUESTION_TEXT")
    if body_value is None:
        body = ""
    elif isinstance(body_value, str):
        body = body_value.strip()
    else:
        raise ValueError(f"{context} has non-string QUESTION_TEXT")
    return f"{title}\n\n{body}" if body else title


def query_output(record: dict[str, Any], context: str) -> dict[str, str]:
    return {
        "question_id": required_string(record, "QUESTION_ID", context),
        "query": build_query(record, context),
        "reference_document_id": required_string(record, "DOCUMENT", context),
    }


def answerable_records(records: list[Any], split: str) -> list[dict[str, Any]]:
    answerable: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{split} record {index} is not an object")
        if record.get("ANSWERABLE") == "Y":
            answerable.append(record)
    return answerable


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for record in records:
                output.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                )
                output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()

    train_path = raw_dir / TRAIN_SOURCE
    dev_path = raw_dir / DEV_SOURCE
    corpus_path = raw_dir / CORPUS_SOURCE

    train_raw: list[Any] = load_json(train_path, list)
    dev_raw: list[Any] = load_json(dev_path, list)
    documents: dict[str, Any] = load_json(corpus_path, dict)

    train_answerable = answerable_records(train_raw, "TRAIN")
    dev_answerable = answerable_records(dev_raw, "DEV")

    train_keys: dict[tuple[Any, Any], list[str]] = {}
    for index, record in enumerate(train_answerable):
        context = f"TRAIN answerable record {index}"
        question_id = required_string(record, "QUESTION_ID", context)
        train_keys.setdefault(duplicate_key(record), []).append(question_id)

    dev_kept: list[dict[str, Any]] = []
    excluded_dev: list[dict[str, Any]] = []
    for index, record in enumerate(dev_answerable):
        context = f"DEV answerable record {index}"
        question_id = required_string(record, "QUESTION_ID", context)
        matching_train_ids = train_keys.get(duplicate_key(record))
        if matching_train_ids:
            excluded_dev.append(
                {
                    "question_id": question_id,
                    "reason": "exact_title_and_text_duplicate_of_answerable_train",
                    "duplicate_train_question_ids": matching_train_ids,
                }
            )
        else:
            dev_kept.append(record)

    train_queries = [
        query_output(record, f"TRAIN answerable record {index}")
        for index, record in enumerate(train_answerable)
    ]
    dev_queries = [
        query_output(record, f"kept DEV record {index}")
        for index, record in enumerate(dev_kept)
    ]

    corpus_records: list[dict[str, str]] = []
    for document_id in sorted(documents):
        document = documents[document_id]
        if not isinstance(document, dict):
            raise ValueError(f"Document {document_id!r} is not an object")
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        title = document.get("title")
        if title is None:
            title = ""
        if not isinstance(title, str):
            raise ValueError(f"Document {document_id!r} has non-string title")
        corpus_records.append(
            {
                "document_id": document_id,
                "title": title,
                "text": text,
            }
        )

    corpus_ids = {record["document_id"] for record in corpus_records}
    train_missing = sorted(
        record["reference_document_id"]
        for record in train_queries
        if record["reference_document_id"] not in corpus_ids
    )
    dev_missing = sorted(
        record["reference_document_id"]
        for record in dev_queries
        if record["reference_document_id"] not in corpus_ids
    )
    if train_missing or dev_missing:
        raise ValueError(
            "Reference documents missing from the non-empty corpus: "
            f"TRAIN={train_missing}, DEV={dev_missing}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_output_path = output_dir / TRAIN_OUTPUT
    dev_output_path = output_dir / DEV_OUTPUT
    corpus_output_path = output_dir / CORPUS_OUTPUT
    excluded_output_path = output_dir / EXCLUDED_DEV_OUTPUT

    write_jsonl(train_output_path, train_queries)
    write_jsonl(dev_output_path, dev_queries)
    write_jsonl(corpus_output_path, corpus_records)
    write_jsonl(excluded_output_path, excluded_dev)

    summary = {
        "rules": {
            "qa_filter": "ANSWERABLE == 'Y'",
            "query_construction": (
                "strip title and body; join non-empty body to title with two newlines; "
                "use title alone when body is empty"
            ),
            "dev_deduplication": (
                "exclude DEV when the untouched QUESTION_TITLE and QUESTION_TEXT pair "
                "exactly matches an answerable TRAIN record"
            ),
            "corpus_filter": "keep every document whose text field is non-empty",
        },
        "sources": {
            TRAIN_SOURCE: {"sha256": sha256_file(train_path)},
            DEV_SOURCE: {"sha256": sha256_file(dev_path)},
            CORPUS_SOURCE: {"sha256": sha256_file(corpus_path)},
        },
        "outputs": {
            TRAIN_OUTPUT: {
                "records": len(train_queries),
                "matched_reference_documents": len(train_queries),
                "missing_reference_documents": 0,
                "sha256": sha256_file(train_output_path),
            },
            DEV_OUTPUT: {
                "records": len(dev_queries),
                "matched_reference_documents": len(dev_queries),
                "missing_reference_documents": 0,
                "sha256": sha256_file(dev_output_path),
            },
            CORPUS_OUTPUT: {
                "records": len(corpus_records),
                "sha256": sha256_file(corpus_output_path),
            },
            EXCLUDED_DEV_OUTPUT: {
                "records": len(excluded_dev),
                "sha256": sha256_file(excluded_output_path),
            },
        },
    }
    summary_path = output_dir / SUMMARY_OUTPUT
    write_json(summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
