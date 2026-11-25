#!/usr/bin/env python3
"""Score the fixed four-query TRAIN smoke set through local llama-server."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.dense.run_qwen3_reranker_smoke import (
    CANDIDATE_COUNT,
    QUERY_COUNT,
    load_inputs,
    load_passages,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "http://127.0.0.1:18080/v1/rerank"
MODEL = "Qwen3-Reranker-4B-Q4_K_M.gguf"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments/dense/results/qwen3_reranker_4b_gguf_smoke_top20_train"
)
RANKINGS = OUTPUT_DIR / "rankings.jsonl"
SUMMARY = OUTPUT_DIR / "summary.json"


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


def rerank(query: str, documents: list[str]) -> list[float]:
    body = json.dumps(
        {"model": MODEL, "query": query, "documents": documents},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server HTTP {exc.code}: {detail}") from None
    rows = result.get("results")
    if not isinstance(rows, list) or len(rows) != len(documents):
        raise ValueError(f"invalid rerank result: {result}")
    scores = [None] * len(documents)
    for row in rows:
        index = row.get("index")
        score = row.get("relevance_score")
        if (
            not isinstance(index, int)
            or not 0 <= index < len(documents)
            or not isinstance(score, (int, float))
            or not np.isfinite(score)
        ):
            raise ValueError(f"invalid rerank row: {row}")
        scores[index] = float(score)
    if any(score is None for score in scores):
        raise ValueError("rerank response omitted a document")
    return [float(score) for score in scores]


class GpuMemoryMonitor:
    def __init__(self) -> None:
        self.samples_mib: list[int] = []
        self.process: subprocess.Popen[str] | None = None
        self.thread: threading.Thread | None = None

    @staticmethod
    def read_used_mib() -> int:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return int(output.strip().splitlines()[0])

    def _sample(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            try:
                self.samples_mib.append(int(line.strip()))
            except ValueError:
                continue

    def __enter__(self):
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--loop-ms=500",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        assert self.process is not None and self.thread is not None
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.thread.join(timeout=5)


def main() -> int:
    prior_rows, queries = load_inputs()
    passages = load_passages(prior_rows)
    baseline_gpu_mib = GpuMemoryMonitor.read_used_mib()
    output_rows: list[dict[str, Any]] = []

    scoring_started = time.perf_counter()
    with GpuMemoryMonitor() as monitor:
        for prior in prior_rows:
            query = queries[prior["question_id"]]
            candidates = prior["baseline"]["ranking"][:CANDIDATE_COUNT]
            documents = [passages[row["representative_chunk_id"]] for row in candidates]
            query_started = time.perf_counter()
            scores = rerank(query["query"], documents)
            query_seconds = time.perf_counter() - query_started

            scored = [
                {
                    "document_id": candidate["document_id"],
                    "original_rank": original_rank,
                    "representative_chunk_id": candidate["representative_chunk_id"],
                    "qwen_relevance_score": score,
                }
                for original_rank, (candidate, score) in enumerate(
                    zip(candidates, scores, strict=True), 1
                )
            ]
            reranked = sorted(
                scored,
                key=lambda row: (-row["qwen_relevance_score"], row["original_rank"]),
            )
            for reranked_rank, row in enumerate(reranked, 1):
                row["reranked_rank"] = reranked_rank
            if {row["document_id"] for row in scored} != {
                row["document_id"] for row in reranked
            }:
                raise ValueError(f"candidate set changed for {query['question_id']}")

            reference_id = query["reference_document_id"]
            before = prior["baseline"]["reference_document_rank"]
            after = next(
                (
                    row["reranked_rank"]
                    for row in reranked
                    if row["document_id"] == reference_id
                ),
                before,
            )
            output_rows.append(
                {
                    "question_id": query["question_id"],
                    "reference_document_id": reference_id,
                    "reference_rank_before": before,
                    "reference_rank_after": after,
                    "scoring_seconds": query_seconds,
                    "top_20_document_set_unchanged": True,
                    "scores_valid": True,
                    "reranked_top_20": reranked,
                }
            )
            print(
                f"{query['question_id']}: reference {before} -> {after}, "
                f"seconds={query_seconds:.3f}",
                flush=True,
            )
    scoring_seconds = time.perf_counter() - scoring_started
    peak_gpu_mib = max(monitor.samples_mib, default=baseline_gpu_mib)
    pair_count = QUERY_COUNT * CANDIDATE_COUNT
    summary = {
        "status": "passed",
        "mode": "local_llama_server_smoke_test",
        "evaluation_split": "TRAIN only",
        "model": MODEL,
        "model_path": "/home/fuxuanyi/models/qwen3-reranker-4b/Qwen3-Reranker-4B-Q4_K_M.gguf",
        "llama_server_endpoint": ENDPOINT,
        "query_count": QUERY_COUNT,
        "candidate_documents_per_query": CANDIDATE_COUNT,
        "pair_count": pair_count,
        "request_count": QUERY_COUNT,
        "scoring_seconds": scoring_seconds,
        "seconds_per_pair": scoring_seconds / pair_count,
        "estimated_9000_pair_seconds": scoring_seconds * 9000 / pair_count,
        "gpu_memory_used_before_scoring_mib": baseline_gpu_mib,
        "peak_total_gpu_memory_used_mib": peak_gpu_mib,
        "all_top_20_document_sets_unchanged": True,
        "all_scores_valid": True,
        "reference_rank_changes": [
            {
                "question_id": row["question_id"],
                "before": row["reference_rank_before"],
                "after": row["reference_rank_after"],
            }
            for row in output_rows
        ],
    }
    atomic_jsonl(RANKINGS, output_rows)
    atomic_json(SUMMARY, summary)
    print("QWEN4B_GGUF_SMOKE_RESULT=" + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
