#!/usr/bin/env python3
"""Generate one cited answer for TRAIN_Q000 with SiliconFlow."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
QUESTION_ID = "TRAIN_Q000"
TEMPERATURE = 0
MAX_TOKENS = 512
TEMPLATE = PROJECT_ROOT / "configs/generation/answer_with_citations.prompt"
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


class RetryableSiliconFlowError(RuntimeError):
    """A transient API/network failure for which the request may be retried."""


def jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def assemble_case() -> tuple[str, list[dict[str, Any]]]:
    template = TEMPLATE.read_text(encoding="utf-8")
    if template.count("{{question}}") != 1 or template.count("{{evidence}}") != 1:
        raise ValueError("generation template must contain exactly two placeholders")
    query = next(
        (row for row in jsonl_rows(QUERIES) if row["question_id"] == QUESTION_ID),
        None,
    )
    ranking = next(
        (row for row in jsonl_rows(RANKINGS) if row["question_id"] == QUESTION_ID),
        None,
    )
    if query is None or ranking is None:
        raise ValueError(f"missing saved inputs for {QUESTION_ID}")
    candidates = ranking["top_documents"][:5]
    if len(candidates) != 5 or len(
        {row["document_id"] for row in candidates}
    ) != 5:
        raise ValueError("saved PPLX ranking does not contain five distinct documents")

    needed_ids = {row["document_id"] for row in candidates}
    documents = {
        row["document_id"]: row
        for row in jsonl_rows(CORPUS)
        if row["document_id"] in needed_ids
    }
    if set(documents) != needed_ids:
        raise ValueError("a saved candidate document is missing from the corpus")

    sources: list[dict[str, Any]] = []
    evidence_parts: list[str] = []
    for source_number, candidate in enumerate(candidates, 1):
        document = documents[candidate["document_id"]]
        chunk_id = candidate["representative_chunk_id"]
        if not chunk_id.startswith(candidate["document_id"] + "::"):
            raise ValueError(f"representative chunk/document mismatch: {chunk_id}")
        chunk_index = int(chunk_id.rsplit("::", 1)[1])
        body = document["text"][chunk_index * 600 : (chunk_index + 1) * 600]
        if not body:
            raise ValueError(f"empty representative chunk: {chunk_id}")
        source = {
            "number": source_number,
            "document_id": candidate["document_id"],
            "chunk_id": chunk_id,
            "title": document["title"],
            "text": body,
        }
        sources.append(source)
        evidence_parts.append(
            f"[{source_number}]\nTitle: {source['title']}\n\n{source['text']}"
        )

    prompt = template.replace("{{question}}", query["query"]).replace(
        "{{evidence}}", "\n\n".join(evidence_parts)
    )
    if "{{question}}" in prompt or "{{evidence}}" in prompt:
        raise ValueError("generation prompt contains an unresolved placeholder")
    return prompt, sources


def error_reason(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        return "the server returned no error details"
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(value, dict):
        parts = [
            str(value[key])
            for key in ("code", "message", "data")
            if value.get(key) not in (None, "")
        ]
        return ": ".join(parts) if parts else text
    return str(value)


def request_completion(
    api_key: str, prompt: str, timeout: int = 180
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        reason = error_reason(exc.read())
        if exc.code in (401, 403):
            raise RuntimeError(f"authorization error (HTTP {exc.code}): {reason}") from None
        if exc.code == 429:
            raise RetryableSiliconFlowError(
                f"rate limit error (HTTP 429): {reason}"
            ) from None
        if exc.code in (408, 409, 425) or 500 <= exc.code <= 599:
            raise RetryableSiliconFlowError(
                f"temporary SiliconFlow HTTP {exc.code}: {reason}"
            ) from None
        raise RuntimeError(f"SiliconFlow HTTP {exc.code}: {reason}") from None
    except urllib.error.URLError as exc:
        raise RetryableSiliconFlowError(
            f"SiliconFlow request failed: {exc.reason}"
        ) from None
    except TimeoutError:
        raise RetryableSiliconFlowError("SiliconFlow request timed out") from None

    try:
        answer = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RetryableSiliconFlowError(
            "SiliconFlow returned a malformed completion response"
        ) from None
    if not isinstance(answer, str) or not answer.strip():
        raise RetryableSiliconFlowError("SiliconFlow returned an empty answer")
    usage = result.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    return answer.strip(), usage


def request_answer(api_key: str, prompt: str, timeout: int = 180) -> str:
    """Compatibility wrapper for existing generation scripts."""
    answer, _ = request_completion(api_key, prompt, timeout=timeout)
    return answer


def cited_numbers(answer: str, source_count: int) -> list[int]:
    found: list[int] = []
    for raw_number in re.findall(r"\[(\d+)\]", answer):
        number = int(raw_number)
        if 1 <= number <= source_count and number not in found:
            found.append(number)
    return found


def gold_answer() -> str:
    # Read only after generation; this label is never included in the prompt.
    records = json.loads(RAW_TRAIN.read_text(encoding="utf-8"))
    record = next(
        (row for row in records if row.get("QUESTION_ID") == QUESTION_ID), None
    )
    if record is None or not isinstance(record.get("ANSWER"), str):
        raise ValueError(f"missing gold answer for {QUESTION_ID}")
    return record["ANSWER"]


def main() -> int:
    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if not api_key:
        print("SILICONFLOW_API_KEY is not set or is empty.", file=sys.stderr)
        return 2
    prompt, sources = assemble_case()
    try:
        answer = request_answer(api_key, prompt)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("=== Generated answer ===")
    print(answer)
    print()
    print("=== Actually cited sources ===")
    numbers = cited_numbers(answer, len(sources))
    if not numbers:
        print("No valid [1]-[5] source citation was found in the answer.")
    for number in numbers:
        source = sources[number - 1]
        print()
        print(f"[{number}] document_id: {source['document_id']}")
        print(f"[{number}] chunk_id: {source['chunk_id']}")
        print(f"[{number}] title: {source['title']}")
        print(f"[{number}] original text:")
        print(source["text"])

    print()
    print("=== Gold answer (post-generation comparison only) ===")
    print(gold_answer())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
