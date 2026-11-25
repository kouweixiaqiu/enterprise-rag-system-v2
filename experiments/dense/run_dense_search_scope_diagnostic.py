#!/usr/bin/env python3
"""Compare full-corpus and old-project-prefiltered Dense search on TRAIN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import jsonl_rows  # noqa: E402
from experiments.dense.run_dense_fixed_600_0 import (  # noqa: E402
    count_truncated,
    document_layout,
    validate_embeddings,
)


DEFAULT_OLD_DOCUMENTS = (
    PROJECT_ROOT.parent
    / "enterprise-rag-system/data/processed/techqa-rag-eval/documents.jsonl"
)
DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
DEFAULT_QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/bge-small-en-v1.5_fixed_600_0"
)
DEFAULT_EXISTING_METRICS = (
    PROJECT_ROOT
    / "experiments/matrix/results/train/fixed_600_0/dense/metrics.json"
)
DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT / "experiments/dense/results/fixed_600_0_search_scope_diagnostic"
)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EXPECTED_QUERIES = 450
EXPECTED_FULL_DOCUMENTS = 28_482
EXPECTED_FULL_CHUNKS = 182_835
EXPECTED_OLD_DOCUMENTS = 496
EMBEDDING_DIMENSION = 384
CUTOFFS = (1, 5, 10, 20, 50)
TOP_K = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-documents", type=Path, default=DEFAULT_OLD_DOCUMENTS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--existing-metrics", type=Path, default=DEFAULT_EXISTING_METRICS
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--query-batch-size", type=int, default=64)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
    os.replace(temporary, path)


def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return requested


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def load_model(device: str) -> SentenceTransformer:
    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=device,
        local_files_only=True,
    )
    model.default_prompt_name = None
    model.eval()
    resolved = getattr(model._first_module().auto_model.config, "_commit_hash", None)
    if resolved is not None and resolved != MODEL_REVISION:
        raise RuntimeError(
            f"model revision mismatch: expected {MODEL_REVISION}, got {resolved}"
        )
    if model.get_embedding_dimension() != EMBEDDING_DIMENSION:
        raise RuntimeError("model embedding dimension mismatch")
    return model


def encode_queries(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, float]:
    synchronize(device)
    started = time.perf_counter()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)
    synchronize(device)
    elapsed = time.perf_counter() - started
    validate_embeddings(vectors)
    return vectors, elapsed


def extract_old_source_document_ids(path: Path) -> list[str]:
    identifiers: set[str] = set()
    rows = list(jsonl_rows(path))
    if len(rows) != EXPECTED_OLD_DOCUMENTS:
        raise ValueError(
            f"expected {EXPECTED_OLD_DOCUMENTS} old-project documents, got {len(rows)}"
        )
    for row in rows:
        filenames = row.get("source_filenames")
        if not isinstance(filenames, list) or not filenames:
            raise ValueError("old-project document lacks source_filenames")
        for filename in filenames:
            if not isinstance(filename, str) or not filename.endswith(".txt"):
                raise ValueError(f"invalid old-project source filename: {filename!r}")
            identifiers.add(filename.removesuffix(".txt"))
    if len(identifiers) != EXPECTED_OLD_DOCUMENTS:
        raise ValueError(
            f"expected {EXPECTED_OLD_DOCUMENTS} unique source IDs, got {len(identifiers)}"
        )
    return sorted(identifiers)


def load_and_validate_cache(
    cache_dir: Path, corpus_path: Path
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    vectors_path = cache_dir / "embeddings.npy"
    metadata_path = cache_dir / "chunk_metadata.jsonl"
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "corpus_sha256": sha256_file(corpus_path),
        "chunk_config": {
            "name": "fixed_600_0",
            "strategy": "fixed",
            "size": 600,
            "overlap": 0,
            "step": 600,
        },
        "normalized": True,
        "dtype": "float32",
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"dense cache manifest mismatch: {mismatches}")
    metadata = list(jsonl_rows(metadata_path))
    embeddings = np.load(vectors_path, mmap_mode="r")
    if embeddings.shape != (EXPECTED_FULL_CHUNKS, EMBEDDING_DIMENSION):
        raise ValueError(f"unexpected embedding shape: {embeddings.shape}")
    if embeddings.dtype != np.float32 or len(metadata) != EXPECTED_FULL_CHUNKS:
        raise ValueError("dense cache dtype or metadata row count mismatch")
    document_ids, _, _, _ = document_layout(metadata)
    if len(document_ids) != EXPECTED_FULL_DOCUMENTS:
        raise ValueError(f"unexpected cache document count: {len(document_ids)}")
    return embeddings, metadata, manifest


def metrics_from_ranks(ranks: list[int]) -> dict[str, Any]:
    count = len(ranks)
    metrics: dict[str, Any] = {"query_count": count}
    for cutoff in CUTOFFS:
        hits = sum(rank <= cutoff for rank in ranks)
        metrics[f"hits_at_{cutoff}"] = hits
        metrics[f"recall_at_{cutoff}"] = hits / count
    metrics["mrr_at_10"] = sum(
        1.0 / rank if rank <= 10 else 0.0 for rank in ranks
    ) / count
    return metrics


def search_group(
    *,
    name: str,
    label: str,
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
    query_vectors: np.ndarray,
    queries: list[dict[str, Any]],
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    document_ids, starts, ends, _ = document_layout(metadata)
    document_id_to_index = {
        str(document_id): index for index, document_id in enumerate(document_ids)
    }
    missing_references = sorted(
        {
            query["reference_document_id"]
            for query in queries
            if query["reference_document_id"] not in document_id_to_index
        }
    )
    if missing_references:
        raise ValueError(f"{name} is missing reference documents: {missing_references}")

    # torch.tensor intentionally copies read-only mmap-backed arrays to the target
    # device; the search path never mutates the source cache.
    document_tensor = torch.tensor(
        np.asarray(embeddings), dtype=torch.float32, device=device
    )
    query_tensor = torch.as_tensor(query_vectors, dtype=torch.float32, device=device)
    rankings: list[dict[str, Any]] = []
    reference_ranks: list[int] = []
    synchronize(device)
    started = time.perf_counter()
    for query_index, query in enumerate(queries):
        chunk_scores = torch.mv(document_tensor, query_tensor[query_index]).cpu().numpy()
        document_scores = np.maximum.reduceat(chunk_scores, starts)
        order = np.lexsort((document_ids, -document_scores))
        reference_index = document_id_to_index[query["reference_document_id"]]
        reference_rank = int(np.flatnonzero(order == reference_index)[0]) + 1
        reference_ranks.append(reference_rank)
        top_documents = []
        for rank, document_index_value in enumerate(order[:TOP_K], 1):
            document_index = int(document_index_value)
            chunk_start = int(starts[document_index])
            chunk_end = int(ends[document_index])
            representative = chunk_start + int(
                np.argmax(chunk_scores[chunk_start:chunk_end])
            )
            top_documents.append(
                {
                    "rank": rank,
                    "document_id": str(document_ids[document_index]),
                    "score": float(document_scores[document_index]),
                    "representative_chunk_id": metadata[representative]["chunk_id"],
                }
            )
        rankings.append(
            {
                "question_id": query["question_id"],
                "reference_document_id": query["reference_document_id"],
                "reference_document_rank": reference_rank,
                "top_documents": top_documents,
            }
        )
        if (query_index + 1) % 50 == 0:
            print(f"{name}: retrieved {query_index + 1}/{len(queries)}", flush=True)
    synchronize(device)
    search_seconds = time.perf_counter() - started
    metrics = {
        "name": name,
        "label": label,
        "document_count": len(document_ids),
        "chunk_count": len(metadata),
        "reference_documents_present": len(queries),
        **metrics_from_ranks(reference_ranks),
        "exact_search_seconds": search_seconds,
        "average_search_seconds": search_seconds / len(queries),
    }
    del document_tensor, query_tensor
    if device == "cuda":
        torch.cuda.empty_cache()
    return rankings, metrics


def validate_full_reproduction(
    current: dict[str, Any], existing_metrics_path: Path
) -> dict[str, Any]:
    existing = json.loads(existing_metrics_path.read_text(encoding="utf-8"))
    fields = [f"recall_at_{cutoff}" for cutoff in CUTOFFS] + ["mrr_at_10"]
    differences = {
        field: current[field] - existing[field]
        for field in fields
    }
    reproduced = existing.get("query_count") == EXPECTED_QUERIES and all(
        abs(value) < 1e-12 for value in differences.values()
    )
    if not reproduced:
        raise ValueError(f"full-corpus metrics do not reproduce existing run: {differences}")
    return {
        "reproduced": True,
        "existing_metrics": str(existing_metrics_path.relative_to(PROJECT_ROOT)),
        "metric_differences": differences,
    }


def markdown_report(payload: dict[str, Any], reproduction_command: str) -> str:
    full = payload["groups"]["A_full_corpus"]
    old = payload["groups"]["B_old_project_prefiltered"]
    lines = [
        "# Dense 搜索范围诊断：fixed 600/0 TRAIN",
        "",
        "本实验只改变 Dense 搜索范围。两组共用一次编码得到的 450 条 TRAIN 查询向量，并复用同一份 BGE Small fixed 600/0 文档向量缓存。未运行 DEV、Base 编码或重排。",
        "",
        "B 组为 **旧项目预筛语料诊断对照**：先按旧项目产物提取的 496 个源文档 ID 筛选全部 Chunk 向量，再在子矩阵内重新检索和排序；不是过滤 A 组 Top-50。",
        "",
        "| 组别 | 文档数 | Chunk 数 | R@1 | R@5 | R@10 | R@20 | R@50 | MRR@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metrics in (full, old):
        lines.append(
            f"| {metrics['label']} | {metrics['document_count']:,} | {metrics['chunk_count']:,} | "
            f"{metrics['recall_at_1']:.4f} | {metrics['recall_at_5']:.4f} | "
            f"{metrics['recall_at_10']:.4f} | {metrics['recall_at_20']:.4f} | "
            f"{metrics['recall_at_50']:.4f} | {metrics['mrr_at_10']:.4f} |"
        )
    comparison = payload["comparison"]
    lines.extend(
        [
            "",
            f"B−A 的 R@5 命中数差为 **{comparison['r5_hit_difference']:+d} 题**，R@50 命中数差为 **{comparison['r50_hit_difference']:+d} 题**。",
            "",
            f"A 组已复现已有 fixed 600/0 Dense 指标：`{str(payload['full_corpus_reproduction']['reproduced']).lower()}`。450 条参考文档在两组均全部存在。",
            "",
            "## 复现命令",
            "",
            "```bash",
            reproduction_command,
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    for name in (
        "old_documents",
        "corpus",
        "queries",
        "cache_dir",
        "existing_metrics",
        "results_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")

    old_document_ids = extract_old_source_document_ids(args.old_documents)
    corpus_ids = {row["document_id"] for row in jsonl_rows(args.corpus)}
    if len(corpus_ids) != EXPECTED_FULL_DOCUMENTS:
        raise ValueError(f"expected {EXPECTED_FULL_DOCUMENTS} corpus documents")
    missing_old_ids = sorted(set(old_document_ids) - corpus_ids)
    if missing_old_ids:
        raise ValueError(f"old-project source IDs missing from new corpus: {missing_old_ids}")
    print(
        f"validated {len(old_document_ids)} old-project source IDs in "
        f"{len(corpus_ids)}-document new corpus",
        flush=True,
    )

    embeddings, metadata, cache_manifest = load_and_validate_cache(
        args.cache_dir, args.corpus
    )
    queries = list(jsonl_rows(args.queries))
    if len(queries) != EXPECTED_QUERIES:
        raise ValueError(f"expected {EXPECTED_QUERIES} TRAIN queries")
    question_ids = [row.get("question_id") for row in queries]
    if len(set(question_ids)) != EXPECTED_QUERIES:
        raise ValueError("TRAIN question IDs are not unique")
    query_texts = [row.get("query") for row in queries]
    if not all(isinstance(text, str) and text for text in query_texts):
        raise ValueError("TRAIN query text is missing or empty")

    device = choose_device(args.device)
    print(f"loading {MODEL_NAME}@{MODEL_REVISION} on {device}", flush=True)
    model = load_model(device)
    query_truncated_count = count_truncated(
        model, query_texts, args.query_batch_size
    )
    query_vectors, query_encoding_seconds = encode_queries(
        model, query_texts, args.query_batch_size, device
    )
    print(
        f"encoded {len(query_vectors)} shared TRAIN queries once in "
        f"{query_encoding_seconds:.2f}s",
        flush=True,
    )

    full_rankings, full_metrics = search_group(
        name="A_full_corpus",
        label="A：全部共享语料",
        embeddings=embeddings,
        metadata=metadata,
        query_vectors=query_vectors,
        queries=queries,
        device=device,
    )
    full_reproduction = validate_full_reproduction(
        full_metrics, args.existing_metrics
    )

    old_id_set = set(old_document_ids)
    selected_chunk_indices = np.fromiter(
        (
            index
            for index, row in enumerate(metadata)
            if row["document_id"] in old_id_set
        ),
        dtype=np.int64,
    )
    subset_metadata = [metadata[int(index)] for index in selected_chunk_indices]
    subset_embeddings = np.asarray(embeddings[selected_chunk_indices], dtype=np.float32)
    subset_document_ids, _, _, _ = document_layout(subset_metadata)
    if set(map(str, subset_document_ids)) != old_id_set:
        raise ValueError("filtered cache document IDs do not match old-project IDs")
    print(
        f"materialized diagnostic subset: {len(subset_document_ids)} documents / "
        f"{len(subset_metadata)} chunks",
        flush=True,
    )
    subset_rankings, subset_metrics = search_group(
        name="B_old_project_prefiltered",
        label="B：旧项目预筛语料诊断对照",
        embeddings=subset_embeddings,
        metadata=subset_metadata,
        query_vectors=query_vectors,
        queries=queries,
        device=device,
    )

    reproduction_command = (
        "./.venv/bin/python "
        "experiments/dense/run_dense_search_scope_diagnostic.py --device cuda"
    )
    payload = {
        "status": "complete",
        "evaluation_split": "TRAIN only",
        "experiment": "Dense fixed 600/0 search-scope diagnostic",
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "query_instruction": None,
            "normalized_embeddings": True,
            "similarity": "exact dot product",
            "document_aggregation": "maximum chunk score per document",
            "document_tie_break": "document_id ascending",
        },
        "query_encoding": {
            "encoded_once_and_shared": True,
            "query_count": len(queries),
            "batch_size": args.query_batch_size,
            "truncated_count": query_truncated_count,
            "seconds": query_encoding_seconds,
            "device": device,
        },
        "old_project_source_ids": {
            "source": str(args.old_documents),
            "unique_count": len(old_document_ids),
            "all_present_in_new_corpus": True,
            "new_corpus_document_count": len(corpus_ids),
        },
        "cache": {
            "directory": str(args.cache_dir.relative_to(PROJECT_ROOT)),
            "manifest_sha256": sha256_file(args.cache_dir / "manifest.json"),
            "embedding_shape": list(embeddings.shape),
            "manifest": cache_manifest,
            "reused": True,
        },
        "groups": {
            "A_full_corpus": full_metrics,
            "B_old_project_prefiltered": {
                **subset_metrics,
                "subset_selection": (
                    "all cached chunk-vector rows whose document_id belongs to the "
                    "496 old-project source IDs; reranked within the materialized subset"
                ),
            },
        },
        "comparison": {
            "r5_hit_difference": (
                subset_metrics["hits_at_5"] - full_metrics["hits_at_5"]
            ),
            "r50_hit_difference": (
                subset_metrics["hits_at_50"] - full_metrics["hits_at_50"]
            ),
            "r5_recall_difference": (
                subset_metrics["recall_at_5"] - full_metrics["recall_at_5"]
            ),
            "r50_recall_difference": (
                subset_metrics["recall_at_50"] - full_metrics["recall_at_50"]
            ),
        },
        "full_corpus_reproduction": full_reproduction,
        "reproduction_command": reproduction_command,
        "excluded_work": ["DEV evaluation", "BGE Base encoding", "reranking"],
    }

    args.results_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(
        args.results_dir / "old_project_source_document_ids.txt",
        "\n".join(old_document_ids) + "\n",
    )
    atomic_jsonl(args.results_dir / "A_full_corpus_rankings_top50.jsonl", full_rankings)
    atomic_jsonl(
        args.results_dir / "B_old_project_prefiltered_rankings_top50.jsonl",
        subset_rankings,
    )
    atomic_json(args.results_dir / "metrics.json", payload)
    atomic_text(
        args.results_dir / "report.md",
        markdown_report(payload, reproduction_command),
    )
    print(json.dumps(payload["groups"], ensure_ascii=False, indent=2), flush=True)
    print(f"report: {args.results_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
