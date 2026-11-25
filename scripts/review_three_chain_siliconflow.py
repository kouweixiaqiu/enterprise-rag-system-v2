#!/usr/bin/env python3
"""Anonymously judge the saved A/B/C TRAIN generations with SiliconFlow."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "experiments/generation/results/siliconflow_train450_three_chain_top5"
OUTPUT_DIR = SOURCE_ROOT / "siliconflow_initial_review"
ENDPOINT = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
API_KEY_FILE = Path.home() / ".config/enterprise-rag/siliconflow.env"
CHAINS = {
    "A": "A_fixed_1200_0_rrf_original",
    "B": "B_fixed_1200_0_rrf_qwen3_4b_bf16_reranked",
    "C": "C_paragraph_merge_1200_0_rrf_original",
}
PROMPT = """You are grading one enterprise technical-support answer.

Use the reference answer only to identify the core answer, necessary procedures,
prerequisites, versions, environments, and applicability conditions. Then verify
the candidate answer against the evidence that was actually supplied to the
answering model. Wording may differ from the reference answer.

Return complete_pass only when ALL of these hold:
- the core answer is correct;
- necessary steps, prerequisites, and applicability conditions are present;
- every key conclusion or operation is supported by the supplied evidence;
- facts from different products, versions, environments, or vulnerabilities are
  not incorrectly combined.

Otherwise return fail. Be strict. A relevant answer that omits a necessary point
is fail. A correct-looking claim unsupported by the supplied evidence is fail.
Separately assess whether the answer's actual [n] citations point to evidence
that supports the associated claims. Missing citations do not by themselves
change complete_pass to fail when the claims are nevertheless supported somewhere
in the supplied evidence; record the citation problem separately.

Return exactly one JSON object with this schema:
{{
  "verdict": "complete_pass" or "fail",
  "reason": "one concise, specific reason",
  "evidence_positions": [
    {{"source_number": 1, "chunk_id": "...", "support": "what this source supports"}}
  ],
  "citation_assessment": "supported" or "partially_supported" or "unsupported" or "no_citations",
  "citation_reason": "one concise reason"
}}

Every field is required. The reason, support, and citation_reason strings must
be non-empty and specific to this case.

Question:
{question}

Reference answer:
{gold}

Candidate answer:
{answer}

Evidence actually supplied to the answering model:
{evidence}
"""


class RetryableError(RuntimeError):
    pass


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_key() -> str:
    value = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if value:
        return value
    if API_KEY_FILE.is_file():
        for raw in API_KEY_FILE.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            key, separator, value = line.partition("=")
            if separator and key.strip() == "SILICONFLOW_API_KEY":
                return value.strip().strip("'\"")
    return ""


def atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_judgment(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise RetryableError("judge returned no JSON object")
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise RetryableError(f"judge returned malformed JSON: {exc}") from None
    if value.get("verdict") not in {"complete_pass", "fail"}:
        raise RetryableError("judge returned an invalid verdict")
    if value.get("citation_assessment") not in {
        "supported", "partially_supported", "unsupported", "no_citations"
    }:
        raise RetryableError("judge returned an invalid citation assessment")
    if not isinstance(value.get("reason"), str) or not value["reason"].strip() or not isinstance(
        value.get("citation_reason"), str
    ) or not value["citation_reason"].strip() or not isinstance(
        value.get("evidence_positions"), list
    ):
        raise RetryableError("judge omitted required fields")
    return value


def request(api_key: str, prompt: str, timeout: int) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 768,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
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
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code in {401, 403}:
            raise RuntimeError(f"authorization error HTTP {exc.code}: {body}") from None
        if exc.code == 429 or exc.code in {408, 409, 425} or 500 <= exc.code <= 599:
            raise RetryableError(f"temporary HTTP {exc.code}: {body}") from None
        raise RuntimeError(f"SiliconFlow HTTP {exc.code}: {body}") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RetryableError(f"request failed: {exc}") from None
    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RetryableError("judge returned a malformed completion") from None
    return parse_judgment(content), result.get("usage") or {}


def evidence_text(sources: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{source['number']}] chunk_id={source['representative_chunk_id']}\n"
        f"Title: {source['title']}\n{source['text']}"
        for source in sources
    )


def build_items() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_chain: dict[str, dict[str, dict[str, Any]]] = {}
    for chain, directory in CHAINS.items():
        path = SOURCE_ROOT / directory / "generations.jsonl"
        data = {row["question_id"]: row for row in rows(path)}
        if len(data) != 450:
            raise ValueError(f"{chain} has {len(data)} questions, expected 450: {path}")
        by_chain[chain] = data
    if len({frozenset(data) for data in by_chain.values()}) != 1:
        raise ValueError("A/B/C question sets differ")
    items = []
    for chain, data in by_chain.items():
        for question_id, row in data.items():
            items.append({
                "chain": chain,
                "question_id": question_id,
                "question": row["prompt"].split("Question:\n\n", 1)[1].split(
                    "\n\nEvidence:\n", 1
                )[0],
                "gold": row["gold_answer"],
                "answer": row["answer"],
                "sources": row["sources"],
            })
    random.Random(20261124).shuffle(items)
    for index, item in enumerate(items, 1):
        item["anonymous_id"] = f"J{index:04d}"
    return items, {item["anonymous_id"]: item for item in items}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--retry-delay", type=float, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    api_key = load_key()
    if not api_key:
        raise SystemExit(f"SILICONFLOW_API_KEY missing from environment and {API_KEY_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "initial_reviews.jsonl"
    mapping_path = OUTPUT_DIR / "anonymous_mapping.jsonl"
    items, by_id = build_items()
    atomic_jsonl(mapping_path, [
        {"anonymous_id": item["anonymous_id"], "question_id": item["question_id"], "chain": item["chain"]}
        for item in items
    ])
    completed = {row["anonymous_id"]: row for row in rows(output_path)} if output_path.is_file() else {}
    pending = [item for item in items if item["anonymous_id"] not in completed]
    # Keep the judge blind to chain names while placing the three variants of a
    # question next to one another. SiliconFlow can then reuse the identical
    # rubric/question/reference prefix without changing any evidence supplied.
    pending.sort(key=lambda item: (item["question_id"], item["anonymous_id"]))
    if args.limit is not None:
        pending = pending[: args.limit]
    lock = threading.Lock()

    def judge(item: dict[str, Any]) -> dict[str, Any]:
        base_prompt = PROMPT.format(
            question=item["question"], gold=item["gold"], answer=item["answer"],
            evidence=evidence_text(item["sources"]),
        )
        retry = 0
        while True:
            try:
                retry_note = "" if retry == 0 else (
                    f"\nThis is validation attempt {retry + 1}. Ensure every required "
                    "string is non-empty and return valid JSON only."
                )
                judgment, usage = request(api_key, base_prompt + retry_note, args.timeout)
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
                    print(f"retry {item['anonymous_id']} #{retry} in {delay:g}s: {exc}", flush=True)
                time.sleep(delay)

    print(f"loaded=1350 reused={len(completed)} pending={len(pending)} workers={args.workers}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(judge, item): item for item in pending}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed[row["anonymous_id"]] = row
            ordered = [completed[item["anonymous_id"]] for item in items if item["anonymous_id"] in completed]
            atomic_jsonl(output_path, ordered)
            done = len(completed)
            if done % 25 == 0 or done == len(items):
                print(f"reviewed {done}/1350", flush=True)
    print(f"initial_reviews={output_path}", flush=True)
    print(f"anonymous_mapping={mapping_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
