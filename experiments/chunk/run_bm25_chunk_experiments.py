#!/usr/bin/env python3
"""Run the five TechQA character-based chunking experiments with SQLite FTS5 BM25."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
DEFAULT_QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments/chunk/bm25_char_results"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data/processed/techqa/bm25_chunk_indexes"

BM25_K1 = 1.2
BM25_B = 0.75
TOP_K_DOCUMENTS = 10
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
TOKENIZATION_DESCRIPTION = (
    "Unicode alphanumeric runs matched by Python regex [^\\W_]+, casefolded; "
    "underscores and all other punctuation are separators; no stemming; "
    "no stopword filtering; duplicate query terms removed before OR matching"
)


@dataclass(frozen=True)
class ChunkConfig:
    name: str
    strategy: str
    size: int | None = None
    overlap: int = 0


CONFIGS = {
    config.name: config
    for config in (
        ChunkConfig("natural_paragraph", "natural_paragraph"),
        ChunkConfig("fixed_600_0", "fixed", size=600, overlap=0),
        ChunkConfig("fixed_600_150", "fixed", size=600, overlap=150),
        ChunkConfig("paragraph_merge_600_0", "paragraph_merge", size=600),
        ChunkConfig("fixed_1200_300", "fixed", size=1200, overlap=300),
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(CONFIGS),
        default=list(CONFIGS),
        help="Configurations to rebuild and evaluate in the given order.",
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text)]


def split_paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def natural_paragraph_chunks(text: str) -> Iterator[str]:
    yield from split_paragraphs(text)


def fixed_chunks(text: str, size: int, overlap: int) -> Iterator[str]:
    step = size - overlap
    if step <= 0:
        raise ValueError(f"Invalid fixed chunk step: size={size}, overlap={overlap}")
    for start in range(0, len(text), step):
        chunk = text[start : start + size]
        if chunk:
            yield chunk
        if start + size >= len(text):
            break


def paragraph_merge_chunks(text: str, size: int) -> Iterator[str]:
    current = ""
    for paragraph in split_paragraphs(text):
        if len(paragraph) > size:
            if current:
                yield current
                current = ""
            full_chunks, remainder = divmod(len(paragraph), size)
            for chunk_index in range(full_chunks):
                chunk = paragraph[chunk_index * size : (chunk_index + 1) * size]
                if chunk_index == full_chunks - 1 and remainder == 0:
                    current = chunk
                else:
                    yield chunk
            if remainder:
                current = paragraph[full_chunks * size :]
            continue

        if not current:
            current = paragraph
        elif len(current) + 2 + len(paragraph) <= size:
            current = f"{current}\n\n{paragraph}"
        else:
            yield current
            current = paragraph

    if current:
        yield current


def chunks_for(text: str, config: ChunkConfig) -> Iterator[str]:
    if config.strategy == "natural_paragraph":
        yield from natural_paragraph_chunks(text)
    elif config.strategy == "fixed":
        assert config.size is not None
        yield from fixed_chunks(text, config.size, config.overlap)
    elif config.strategy == "paragraph_merge":
        assert config.size is not None
        yield from paragraph_merge_chunks(text, config.size)
    else:
        raise ValueError(f"Unknown chunk strategy: {config.strategy}")


def nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(percentile * len(ordered)) - 1]


def jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for value in values:
                output.write(
                    json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                )
                output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def check_sqlite_bm25() -> None:
    if (BM25_K1, BM25_B) != (1.2, 0.75):
        raise RuntimeError("SQLite FTS5 bm25() only supports k1=1.2 and b=0.75")
    connection = sqlite3.connect(":memory:")
    try:
        enabled = connection.execute(
            "SELECT sqlite_compileoption_used('ENABLE_FTS5')"
        ).fetchone()[0]
        if enabled != 1:
            raise RuntimeError("SQLite was not compiled with FTS5 support")
        connection.execute(
            "CREATE VIRTUAL TABLE smoke USING fts5(tokens, "
            "tokenize='unicode61 remove_diacritics 0')"
        )
        connection.execute("INSERT INTO smoke(tokens) VALUES ('bm25 smoke test')")
        connection.execute(
            "SELECT bm25(smoke) FROM smoke WHERE smoke MATCH 'bm25'"
        ).fetchone()
    finally:
        connection.close()


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA cache_size=-262144;
        CREATE TABLE chunk_meta (
            rowid INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            body_length INTEGER NOT NULL
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            tokens,
            tokenize='unicode61 remove_diacritics 0'
        );
        """
    )


def flush_batches(
    connection: sqlite3.Connection,
    meta_batch: list[tuple[int, str, str, int, int]],
    fts_batch: list[tuple[int, str]],
) -> None:
    connection.executemany(
        "INSERT INTO chunk_meta(rowid, chunk_id, document_id, chunk_index, body_length) "
        "VALUES (?, ?, ?, ?, ?)",
        meta_batch,
    )
    connection.executemany(
        "INSERT INTO chunks_fts(rowid, tokens) VALUES (?, ?)", fts_batch
    )
    meta_batch.clear()
    fts_batch.clear()


def build_index(
    corpus_path: Path, index_path: Path, config: ChunkConfig
) -> dict[str, Any]:
    index_path.unlink(missing_ok=True)
    connection = sqlite3.connect(index_path)
    started = time.perf_counter()
    chunk_lengths: list[int] = []
    document_ids: set[str] = set()
    rowid = 0
    meta_batch: list[tuple[int, str, str, int, int]] = []
    fts_batch: list[tuple[int, str]] = []

    try:
        create_schema(connection)
        connection.execute("BEGIN")
        for document in jsonl_rows(corpus_path):
            document_id = document.get("document_id")
            title = document.get("title")
            text = document.get("text")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("Corpus document has missing document_id")
            if document_id in document_ids:
                raise ValueError(f"Duplicate corpus document_id: {document_id}")
            if not isinstance(title, str) or not isinstance(text, str) or not text:
                raise ValueError(f"Invalid corpus fields for document {document_id}")
            document_ids.add(document_id)
            title_tokens = tokenize(title)

            document_chunk_count = 0
            for chunk_index, body in enumerate(chunks_for(text, config)):
                rowid += 1
                document_chunk_count += 1
                chunk_id = f"{document_id}::{chunk_index:06d}"
                body_length = len(body)
                chunk_lengths.append(body_length)
                index_tokens = " ".join(title_tokens + tokenize(body))
                meta_batch.append(
                    (rowid, chunk_id, document_id, chunk_index, body_length)
                )
                fts_batch.append((rowid, index_tokens))
                if len(meta_batch) >= 2000:
                    flush_batches(connection, meta_batch, fts_batch)
            if document_chunk_count == 0:
                raise ValueError(f"Chunking produced no chunks for document {document_id}")

        if meta_batch:
            flush_batches(connection, meta_batch, fts_batch)
        connection.commit()
        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        connection.commit()
    finally:
        connection.close()

    elapsed = time.perf_counter() - started
    return {
        "document_ids": document_ids,
        "document_count": len(document_ids),
        "chunk_count": len(chunk_lengths),
        "chunk_body_length_p50": nearest_rank(chunk_lengths, 0.50),
        "chunk_body_length_p95": nearest_rank(chunk_lengths, 0.95),
        "index_build_seconds": elapsed,
        "index_size_bytes": index_path.stat().st_size,
    }


def fts_query(tokens: list[str]) -> str:
    unique_tokens = list(dict.fromkeys(tokens))
    return " OR ".join(f'"{token}"' for token in unique_tokens)


def retrieve_top_documents(
    connection: sqlite3.Connection, query: str, top_k: int
) -> list[dict[str, Any]]:
    tokens = tokenize(query)
    if not tokens:
        return []
    match_expression = fts_query(tokens)
    cursor = connection.execute(
        """
        SELECT
            chunk_meta.document_id,
            chunk_meta.chunk_id,
            bm25(chunks_fts) AS bm25_rank
        FROM chunks_fts
        JOIN chunk_meta ON chunk_meta.rowid = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY bm25_rank ASC, chunk_meta.document_id ASC, chunk_meta.chunk_index ASC
        """,
        (match_expression,),
    )
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for document_id, chunk_id, rank_score in cursor:
        if document_id in seen:
            continue
        seen.add(document_id)
        documents.append(
            {
                "rank": len(documents) + 1,
                "document_id": document_id,
                "score": -float(rank_score),
                "representative_chunk_id": chunk_id,
            }
        )
        if len(documents) >= top_k:
            break
    cursor.close()
    return documents


def evaluate(
    index_path: Path,
    query_path: Path,
    valid_document_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries = list(jsonl_rows(query_path))
    if len(queries) != 450:
        raise ValueError(f"Expected 450 TRAIN queries, found {len(queries)}")

    connection = sqlite3.connect(index_path)
    connection.execute("PRAGMA query_only=ON")
    results: list[dict[str, Any]] = []
    durations: list[float] = []
    invalid_returned_ids: set[str] = set()
    queries_with_ten_documents = 0

    try:
        for query_record in queries:
            question_id = query_record.get("question_id")
            query = query_record.get("query")
            reference_id = query_record.get("reference_document_id")
            if not all(isinstance(value, str) and value for value in (question_id, query, reference_id)):
                raise ValueError(f"Invalid query record: {query_record}")

            started = time.perf_counter()
            top_documents = retrieve_top_documents(
                connection, query, TOP_K_DOCUMENTS
            )
            durations.append(time.perf_counter() - started)
            if len(top_documents) == TOP_K_DOCUMENTS:
                queries_with_ten_documents += 1
            invalid_returned_ids.update(
                item["document_id"]
                for item in top_documents
                if item["document_id"] not in valid_document_ids
            )
            reference_rank = next(
                (
                    item["rank"]
                    for item in top_documents
                    if item["document_id"] == reference_id
                ),
                None,
            )
            results.append(
                {
                    "question_id": question_id,
                    "reference_document_id": reference_id,
                    "reference_document_rank": reference_rank,
                    "reference_rank_scope": "top_10; null means not in top 10",
                    "top_documents": top_documents,
                }
            )
    finally:
        connection.close()

    if invalid_returned_ids:
        raise ValueError(
            f"Retriever returned invalid document IDs: {sorted(invalid_returned_ids)}"
        )

    denominator = len(results)
    recall_at_1 = sum(row["reference_document_rank"] == 1 for row in results) / denominator
    recall_at_5 = sum(
        row["reference_document_rank"] is not None
        and row["reference_document_rank"] <= 5
        for row in results
    ) / denominator
    mrr_at_10 = sum(
        1.0 / row["reference_document_rank"]
        if row["reference_document_rank"] is not None
        else 0.0
        for row in results
    ) / denominator
    metrics = {
        "query_count": denominator,
        "queries_with_10_documents": queries_with_ten_documents,
        "invalid_returned_document_ids": 0,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "mrr_at_10": mrr_at_10,
        "average_query_seconds": sum(durations) / len(durations),
    }
    return results, metrics


def run_config(
    config: ChunkConfig,
    corpus_path: Path,
    query_path: Path,
    results_dir: Path,
    index_dir: Path,
) -> dict[str, Any]:
    print(f"[{config.name}] building index", flush=True)
    index_path = index_dir / f"{config.name}.sqlite"
    index_stats = build_index(corpus_path, index_path, config)
    document_ids = index_stats.pop("document_ids")
    print(
        f"[{config.name}] indexed {index_stats['chunk_count']} chunks from "
        f"{index_stats['document_count']} documents in "
        f"{index_stats['index_build_seconds']:.3f}s",
        flush=True,
    )

    print(f"[{config.name}] evaluating 450 TRAIN queries", flush=True)
    query_results, retrieval_metrics = evaluate(
        index_path, query_path, document_ids
    )
    result_dir = results_dir / config.name
    result_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = result_dir / "query_rankings.jsonl"
    write_jsonl(rankings_path, query_results)

    metrics = {
        "config": asdict(config),
        "bm25": {
            "implementation": "Python sqlite3 / SQLite FTS5 bm25()",
            "k1": BM25_K1,
            "b": BM25_B,
            "reranking": False,
            "index_text": "document title + chunk body",
            "document_aggregation": "highest-scoring chunk per document",
            "tie_break": "document_id ascending",
            "tokenization": TOKENIZATION_DESCRIPTION,
            "idf_and_average_length": "recomputed in this configuration's independent index",
        },
        "index": index_stats,
        "retrieval": retrieval_metrics,
        "artifacts": {
            "rankings": str(rankings_path.relative_to(PROJECT_ROOT)),
            "index": str(index_path.relative_to(PROJECT_ROOT)),
        },
        "environment": {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
        },
    }
    write_json(result_dir / "metrics.json", metrics)
    print(
        f"[{config.name}] R@1={retrieval_metrics['recall_at_1']:.6f} "
        f"R@5={retrieval_metrics['recall_at_5']:.6f} "
        f"MRR@10={retrieval_metrics['mrr_at_10']:.6f}",
        flush=True,
    )
    return metrics


def write_comparison(results_dir: Path) -> None:
    rows = []
    for config_name in CONFIGS:
        metrics_path = results_dir / config_name / "metrics.json"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8") as source:
            metrics = json.load(source)
        rows.append(
            {
                "config": config_name,
                "chunk_count": metrics["index"]["chunk_count"],
                "chunk_body_length_p50": metrics["index"]["chunk_body_length_p50"],
                "chunk_body_length_p95": metrics["index"]["chunk_body_length_p95"],
                "recall_at_1": metrics["retrieval"]["recall_at_1"],
                "recall_at_5": metrics["retrieval"]["recall_at_5"],
                "mrr_at_10": metrics["retrieval"]["mrr_at_10"],
                "index_build_seconds": metrics["index"]["index_build_seconds"],
                "average_query_seconds": metrics["retrieval"]["average_query_seconds"],
            }
        )
    write_json(
        results_dir / "comparison.json",
        {
            "evaluation_split": "TRAIN only",
            "query_denominator": 450,
            "top_k_documents": TOP_K_DOCUMENTS,
            "results": rows,
        },
    )


def main() -> int:
    args = parse_args()
    check_sqlite_bm25()
    corpus_path = args.corpus.resolve()
    query_path = args.queries.resolve()
    results_dir = args.results_dir.resolve()
    index_dir = args.index_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"tokenization: {TOKENIZATION_DESCRIPTION}\n"
        f"BM25: k1={BM25_K1}, b={BM25_B}, reranking=false",
        flush=True,
    )
    for config_name in args.configs:
        run_config(
            CONFIGS[config_name],
            corpus_path,
            query_path,
            results_dir,
            index_dir,
        )
        write_comparison(results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
