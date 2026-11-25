#!/usr/bin/env python3
"""Rerank saved PPLX document candidates with bge-reranker-base on TRAIN."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import (  # noqa: E402
    fixed_chunks,
    jsonl_rows,
)


RANKINGS = (
    PROJECT_ROOT
    / "experiments/dense/results/pplx_embed_v1_0.6b_fixed_600_0_train/"
    "dense_rankings_top100.jsonl"
)
QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
CHUNK_METADATA = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/bge-small-en-v1.5_fixed_600_0/"
    "chunk_metadata.jsonl"
)
PPLX_VECTORS = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/pplx-embed-v1-0.6b_fixed_600_0/"
    "embeddings.npy"
)
PPLX_MODEL = "perplexity-ai/pplx-embed-v1-0.6b"
PPLX_REVISION = "2c4d510dd4a732063c31a0f70193e35067b51fd8"
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_REVISION = "75bc78764e5eeb49e6a4a085e65eb2985ce82320"
RERANKER_SNAPSHOT = Path(
    "/home/fuxuanyi/AI_projects/enterprise-rag-system/artifacts/model_cache/"
    "huggingface/models--BAAI--bge-reranker-base/snapshots/"
    + RERANKER_REVISION
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "experiments/dense/results/pplx_bge_reranker_fixed_600_0_train"
)
EXPECTED_QUERIES = 450
CANDIDATES = 50
TOP20 = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reranker-snapshot", type=Path, default=RERANKER_SNAPSHOT)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        parser.error("batch size and max length must be positive")
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be positive")
    return args


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


def load_inputs(limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ranking_rows = list(jsonl_rows(RANKINGS))
    query_rows = list(jsonl_rows(QUERIES))
    if len(ranking_rows) != EXPECTED_QUERIES or len(query_rows) != EXPECTED_QUERIES:
        raise ValueError("expected exactly 450 saved PPLX rankings and TRAIN queries")
    for index, (ranking, query) in enumerate(zip(ranking_rows, query_rows, strict=True)):
        if (
            ranking["question_id"] != query["question_id"]
            or ranking["reference_document_id"] != query["reference_document_id"]
        ):
            raise ValueError(f"ranking/query mismatch at row {index}")
        candidates = ranking.get("top_documents", [])[:CANDIDATES]
        document_ids = [row["document_id"] for row in candidates]
        if len(candidates) != CANDIDATES or len(set(document_ids)) != CANDIDATES:
            raise ValueError(f"question {query['question_id']} lacks 50 distinct documents")
    if limit is not None:
        ranking_rows = ranking_rows[:limit]
        query_rows = query_rows[:limit]
    return ranking_rows, query_rows


def choose_fallback_representatives(
    ranking_rows: Sequence[dict[str, Any]],
    query_rows: Sequence[dict[str, Any]],
) -> int:
    """Fill missing representative IDs using cached PPLX vectors within each document."""
    missing = [
        (query_index, candidate)
        for query_index, ranking in enumerate(ranking_rows)
        for candidate in ranking["top_documents"][:CANDIDATES]
        if not candidate.get("representative_chunk_id")
    ]
    if not missing:
        return 0

    metadata = list(jsonl_rows(CHUNK_METADATA))
    vectors = np.load(PPLX_VECTORS, mmap_mode="r")
    if vectors.shape[0] != len(metadata):
        raise ValueError("PPLX vector and chunk metadata row counts differ")
    rows_by_document: dict[str, list[int]] = {}
    for row_index, row in enumerate(metadata):
        rows_by_document.setdefault(row["document_id"], []).append(row_index)

    from sentence_transformers import SentenceTransformer

    print(f"representative fallback required for {len(missing)} candidates", flush=True)
    model = SentenceTransformer(
        PPLX_MODEL,
        revision=PPLX_REVISION,
        trust_remote_code=True,
        local_files_only=True,
        device="cuda",
    )
    needed_queries = sorted({query_index for query_index, _ in missing})
    encoded = model.encode(
        [query_rows[index]["query"] for index in needed_queries],
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    query_vectors = dict(zip(needed_queries, encoded, strict=True))
    for query_index, candidate in missing:
        document_rows = rows_by_document.get(candidate["document_id"])
        if not document_rows:
            raise ValueError(f"no chunks for candidate {candidate['document_id']}")
        scores = np.asarray(vectors[document_rows]) @ query_vectors[query_index]
        best_row = document_rows[int(np.argmax(scores))]
        candidate["representative_chunk_id"] = metadata[best_row]["chunk_id"]
        candidate["representative_source"] = "pplx_vector_fallback"
    del model
    torch.cuda.empty_cache()
    return len(missing)


def load_representative_passages(
    ranking_rows: Sequence[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    required: dict[str, str] = {}
    for ranking in ranking_rows:
        for candidate in ranking["top_documents"][:CANDIDATES]:
            chunk_id = candidate["representative_chunk_id"]
            document_id = candidate["document_id"]
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError(f"missing representative chunk for {document_id}")
            if not chunk_id.startswith(document_id + "::"):
                raise ValueError(f"representative {chunk_id} is not from {document_id}")
            required[chunk_id] = document_id

    passages: dict[str, dict[str, str]] = {}
    required_documents = set(required.values())
    for document in jsonl_rows(CORPUS):
        document_id = document["document_id"]
        if document_id not in required_documents:
            continue
        for chunk_index, body in enumerate(fixed_chunks(document["text"], 600, 0)):
            chunk_id = f"{document_id}::{chunk_index:06d}"
            if chunk_id in required:
                passages[chunk_id] = {
                    "document_id": document_id,
                    "title": document["title"],
                    "body": body,
                    "text": f"{document['title']}\n\n{body}",
                }
    missing = sorted(set(required) - set(passages))
    if missing:
        raise ValueError(f"could not reconstruct representative chunks: {missing[:5]}")
    return passages


class Reranker:
    def __init__(self, snapshot: Path, batch_size: int, max_length: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        if not snapshot.is_dir():
            raise FileNotFoundError(f"cached reranker snapshot not found: {snapshot}")
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
        self.tokenizer.truncation_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
        ).to("cuda")
        self.model.eval()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        scores: list[float] = []
        with torch.inference_mode():
            for start in range(0, len(passages), self.batch_size):
                batch = list(passages[start : start + self.batch_size])
                encoded = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding="longest",
                    truncation="longest_first",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
                logits = self.model(**encoded).logits.reshape(-1)
                scores.extend(logits.detach().to(torch.float32).cpu().tolist())
        if len(scores) != len(passages) or not np.isfinite(scores).all():
            raise ValueError("reranker returned invalid scores")
        return scores


def ranked_candidates(
    candidates: Sequence[dict[str, Any]], scores: Sequence[float], depth: int
) -> list[dict[str, Any]]:
    enriched = [
        {
            "document_id": candidate["document_id"],
            "original_rank": original_rank,
            "pplx_score": float(candidate["score"]),
            "representative_chunk_id": candidate["representative_chunk_id"],
            "rerank_score": float(score),
        }
        for original_rank, (candidate, score) in enumerate(
            zip(candidates, scores, strict=True), 1
        )
    ]
    head = sorted(enriched[:depth], key=lambda row: (-row["rerank_score"], row["original_rank"]))
    output = head + enriched[depth:]
    for rank, row in enumerate(output, 1):
        row["rank"] = rank
    return output


def baseline_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank,
            "document_id": candidate["document_id"],
            "original_rank": rank,
            "pplx_score": float(candidate["score"]),
            "representative_chunk_id": candidate["representative_chunk_id"],
        }
        for rank, candidate in enumerate(candidates, 1)
    ]


def reference_rank(reference_id: str, rows: Sequence[dict[str, Any]]) -> int | None:
    return next((row["rank"] for row in rows if row["document_id"] == reference_id), None)


def metric_summary(
    rows: Sequence[dict[str, Any]], variant: str, baseline_variant: str = "baseline"
) -> dict[str, Any]:
    ranks = [row[variant]["reference_document_rank"] for row in rows]
    baseline_ranks = [row[baseline_variant]["reference_document_rank"] for row in rows]
    hits = sum(rank is not None and rank <= 5 for rank in ranks)
    return {
        "query_count": len(rows),
        "hits_at_5": hits,
        "recall_at_5": hits / len(rows),
        "mrr_at_10": sum(
            1.0 / rank if rank is not None and rank <= 10 else 0.0 for rank in ranks
        )
        / len(rows),
        "new_into_top_5": sum(
            (base is None or base > 5) and rank is not None and rank <= 5
            for base, rank in zip(baseline_ranks, ranks, strict=True)
        ),
        "dropped_out_of_top_5": sum(
            base is not None and base <= 5 and (rank is None or rank > 5)
            for base, rank in zip(baseline_ranks, ranks, strict=True)
        ),
    }


def timing_summary(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, int(np.ceil(0.95 * len(ordered))) - 1)
    return {
        "total_seconds": sum(values),
        "mean_seconds_per_query": statistics.fmean(values),
        "median_seconds_per_query": statistics.median(values),
        "p95_seconds_per_query": ordered[p95_index],
    }


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    ranking_rows, query_rows = load_inputs(args.max_queries)
    fallback_count = choose_fallback_representatives(ranking_rows, query_rows)
    passages = load_representative_passages(ranking_rows)
    print(
        f"validated queries={len(query_rows)} candidates_per_query={CANDIDATES} "
        f"representative_chunks={len(passages)} fallback={fallback_count}",
        flush=True,
    )
    print(
        f"loading cached {RERANKER_MODEL}@{RERANKER_REVISION} "
        f"batch_size={args.batch_size} max_length={args.max_length}",
        flush=True,
    )
    reranker = Reranker(args.reranker_snapshot, args.batch_size, args.max_length)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    output_rows: list[dict[str, Any]] = []
    top20_times: list[float] = []
    extra30_times: list[float] = []
    for query_index, (ranking, query) in enumerate(
        zip(ranking_rows, query_rows, strict=True), 1
    ):
        candidates = ranking["top_documents"][:CANDIDATES]
        candidate_passages = [
            passages[candidate["representative_chunk_id"]]["text"]
            for candidate in candidates
        ]

        torch.cuda.synchronize()
        segment_started = time.perf_counter()
        first_scores = reranker.score(query["query"], candidate_passages[:TOP20])
        torch.cuda.synchronize()
        top20_seconds = time.perf_counter() - segment_started

        torch.cuda.synchronize()
        segment_started = time.perf_counter()
        additional_scores = reranker.score(query["query"], candidate_passages[TOP20:])
        torch.cuda.synchronize()
        extra30_seconds = time.perf_counter() - segment_started
        scores = first_scores + additional_scores

        baseline = baseline_candidates(candidates)
        rerank20 = ranked_candidates(candidates, scores, TOP20)
        rerank50 = ranked_candidates(candidates, scores, CANDIDATES)
        candidate_set = {row["document_id"] for row in baseline}
        if (
            candidate_set != {row["document_id"] for row in rerank20}
            or candidate_set != {row["document_id"] for row in rerank50}
        ):
            raise ValueError(f"candidate set changed for {query['question_id']}")
        reference_id = query["reference_document_id"]
        output_rows.append(
            {
                "question_id": query["question_id"],
                "reference_document_id": reference_id,
                "scoring_seconds": {
                    "top_20": top20_seconds,
                    "additional_30": extra30_seconds,
                    "top_50": top20_seconds + extra30_seconds,
                },
                "baseline": {
                    "reference_document_rank": reference_rank(reference_id, baseline),
                    "ranking": baseline,
                },
                "rerank_top_20": {
                    "reference_document_rank": reference_rank(reference_id, rerank20),
                    "ranking": rerank20,
                },
                "rerank_top_50": {
                    "reference_document_rank": reference_rank(reference_id, rerank50),
                    "ranking": rerank50,
                },
            }
        )
        top20_times.append(top20_seconds)
        extra30_times.append(extra30_seconds)
        if query_index % 25 == 0 or query_index == len(query_rows):
            print(f"scored {query_index}/{len(query_rows)} queries", flush=True)

    top50_times = [a + b for a, b in zip(top20_times, extra30_times, strict=True)]
    metrics = {
        "baseline": metric_summary(output_rows, "baseline"),
        "rerank_top_20": metric_summary(output_rows, "rerank_top_20"),
        "rerank_top_50": metric_summary(output_rows, "rerank_top_50"),
    }
    if args.max_queries is None and metrics["baseline"]["hits_at_5"] != 335:
        raise ValueError(
            f"baseline R@5 count changed: {metrics['baseline']['hits_at_5']} != 335"
        )
    summary = {
        "status": "complete",
        "evaluation_split": "TRAIN only",
        "query_count": len(output_rows),
        "candidate_documents_per_query": CANDIDATES,
        "same_candidate_set_confirmed": True,
        "baseline_335_of_450_confirmed": (
            args.max_queries is None and metrics["baseline"]["hits_at_5"] == 335
        ),
        "source_rankings": str(RANKINGS.relative_to(PROJECT_ROOT)),
        "representative_chunk_fallback_count": fallback_count,
        "passage_format": "document title + two newlines + representative fixed-600/0 body",
        "model": {
            "name": RERANKER_MODEL,
            "revision": RERANKER_REVISION,
            "snapshot": str(args.reranker_snapshot),
            "device": torch.cuda.get_device_name(0),
            "dtype": "float32",
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "score": "raw model logit",
        },
        "metrics": metrics,
        "timing": {
            "top_20": timing_summary(top20_times),
            "additional_30": timing_summary(extra30_times),
            "top_50": timing_summary(top50_times),
            "experiment_wall_seconds": time.perf_counter() - started,
        },
        "peak_gpu_memory_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "peak_gpu_memory_reserved_gib": torch.cuda.max_memory_reserved() / (1024**3),
    }
    rankings_path = args.output_dir / "rankings.jsonl"
    summary_path = args.output_dir / "summary.json"
    atomic_jsonl(rankings_path, output_rows)
    atomic_json(summary_path, summary)
    print("RERANK_RESULT=" + json.dumps(summary, ensure_ascii=False), flush=True)
    print(f"rankings={rankings_path}", flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
