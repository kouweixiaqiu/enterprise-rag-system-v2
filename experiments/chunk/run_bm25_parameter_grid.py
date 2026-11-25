#!/usr/bin/env python3
"""Run the formal 6 chunk x 9 BM25 parameter grid on TechQA TRAIN."""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.chunk.run_bm25_chunk_experiments import (  # noqa: E402
    TOKENIZATION_DESCRIPTION,
    ChunkConfig,
    chunks_for,
    fts_query,
    jsonl_rows,
    nearest_rank,
    tokenize,
    write_json,
    write_jsonl,
)


DEFAULT_CORPUS = PROJECT_ROOT / "data/processed/techqa/corpus.jsonl"
DEFAULT_QUERIES = PROJECT_ROOT / "data/processed/techqa/train_queries.jsonl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments/chunk/bm25_parameter_grid_train"
DEFAULT_INDEX_DIR = (
    PROJECT_ROOT / "data/processed/techqa/bm25_parameter_grid_indexes"
)
INDEX_SCHEMA_VERSION = 1
TOP_K = 100
BASELINE = (1.2, 0.75)

CONFIGS = {
    config.name: config
    for config in (
        ChunkConfig("fixed_600_0", "fixed", size=600, overlap=0),
        ChunkConfig("fixed_600_150", "fixed", size=600, overlap=150),
        ChunkConfig("fixed_1200_0", "fixed", size=1200, overlap=0),
        ChunkConfig("fixed_1200_300", "fixed", size=1200, overlap=300),
        ChunkConfig("paragraph_merge_600_0", "paragraph_merge", size=600),
        ChunkConfig("paragraph_merge_1200_0", "paragraph_merge", size=1200),
    )
}
DEFAULT_K1 = (0.8, 1.2, 1.6)
DEFAULT_B = (0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS)
    )
    parser.add_argument("--k1-values", nargs="+", type=float, default=DEFAULT_K1)
    parser.add_argument("--b-values", nargs="+", type=float, default=DEFAULT_B)
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help="Validation only: evaluate the first N TRAIN questions; 0 means all 450.",
    )
    return parser.parse_args()


def atomic_replace_sqlite(temporary: Path, final: Path) -> None:
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, final)


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def index_signature(corpus: Path, config: ChunkConfig) -> dict[str, Any]:
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "corpus": source_signature(corpus),
        "config": asdict(config),
        "tokenization": TOKENIZATION_DESCRIPTION,
        "index_text": "document title + chunk body",
    }


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
            body_length INTEGER NOT NULL,
            token_count INTEGER NOT NULL
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            tokens,
            tokenize='unicode61 remove_diacritics 0'
        );
        CREATE TABLE run_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )


def flush_batches(
    connection: sqlite3.Connection,
    metadata: list[tuple[int, str, str, int, int, int]],
    fts_rows: list[tuple[int, str]],
) -> None:
    connection.executemany(
        "INSERT INTO chunk_meta(rowid, chunk_id, document_id, chunk_index, "
        "body_length, token_count) VALUES (?, ?, ?, ?, ?, ?)",
        metadata,
    )
    connection.executemany(
        "INSERT INTO chunks_fts(rowid, tokens) VALUES (?, ?)", fts_rows
    )
    metadata.clear()
    fts_rows.clear()


def read_index_metadata(index_path: Path) -> dict[str, Any] | None:
    if not index_path.is_file():
        return None
    try:
        connection = sqlite3.connect(index_path)
        row = connection.execute(
            "SELECT value FROM run_metadata WHERE key='metadata'"
        ).fetchone()
        connection.close()
        return json.loads(row[0]) if row else None
    except (sqlite3.Error, json.JSONDecodeError):
        return None


def build_index(
    corpus: Path, index_path: Path, config: ChunkConfig
) -> dict[str, Any]:
    signature = index_signature(corpus, config)
    existing = read_index_metadata(index_path)
    if existing and existing.get("signature") == signature:
        return existing["statistics"]

    temporary = index_path.with_name(f".{index_path.name}.tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    started = time.perf_counter()
    metadata_batch: list[tuple[int, str, str, int, int, int]] = []
    fts_batch: list[tuple[int, str]] = []
    document_ids: set[str] = set()
    chunk_lengths: list[int] = []
    token_total = 0
    rowid = 0
    try:
        create_schema(connection)
        connection.execute("BEGIN")
        for document in jsonl_rows(corpus):
            document_id = document.get("document_id")
            title = document.get("title")
            text = document.get("text")
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("corpus document has no document_id")
            if document_id in document_ids:
                raise ValueError(f"duplicate corpus document_id: {document_id}")
            if not isinstance(title, str) or not isinstance(text, str) or not text:
                raise ValueError(f"invalid corpus fields: {document_id}")
            document_ids.add(document_id)
            title_tokens = tokenize(title)
            produced = 0
            for chunk_index, body in enumerate(chunks_for(text, config)):
                body_tokens = tokenize(body)
                all_tokens = title_tokens + body_tokens
                rowid += 1
                produced += 1
                chunk_id = f"{document_id}::{chunk_index:06d}"
                metadata_batch.append(
                    (
                        rowid,
                        chunk_id,
                        document_id,
                        chunk_index,
                        len(body),
                        len(all_tokens),
                    )
                )
                fts_batch.append((rowid, " ".join(all_tokens)))
                chunk_lengths.append(len(body))
                token_total += len(all_tokens)
                if len(metadata_batch) >= 2000:
                    flush_batches(connection, metadata_batch, fts_batch)
            if not produced:
                raise ValueError(f"chunking produced no chunks: {document_id}")
        if metadata_batch:
            flush_batches(connection, metadata_batch, fts_batch)
        connection.commit()
        connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('optimize')")
        statistics = {
            "document_count": len(document_ids),
            "chunk_count": rowid,
            "average_token_count": token_total / rowid,
            "chunk_body_length_p50": nearest_rank(chunk_lengths, 0.50),
            "chunk_body_length_p95": nearest_rank(chunk_lengths, 0.95),
            "index_build_seconds": time.perf_counter() - started,
        }
        metadata = {"signature": signature, "statistics": statistics}
        connection.execute(
            "INSERT INTO run_metadata(key, value) VALUES ('metadata', ?)",
            (json.dumps(metadata, ensure_ascii=False, sort_keys=True),),
        )
        connection.commit()
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    connection.close()
    atomic_replace_sqlite(temporary, index_path)
    statistics["index_build_seconds"] = time.perf_counter() - started
    return statistics


def parameter_name(k1: float, b: float) -> str:
    return f"k1_{k1:g}_b_{b:g}"


def display_path(path: Path) -> str:
    """Prefer repository-relative artifact paths, while allowing /tmp validation."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def idf(chunk_count: int, document_frequency: int) -> float:
    value = math.log(
        (chunk_count - document_frequency + 0.5) / (document_frequency + 0.5)
    )
    return max(value, 1e-6)


def finalize_chunk(
    term_frequencies: Sequence[tuple[int, float]],
    token_count: int,
    document_id: str,
    chunk_id: str,
    chunk_index: int,
    average_length: float,
    parameters: Sequence[tuple[float, float]],
    best: dict[tuple[float, float], dict[str, tuple[float, int, str]]],
) -> None:
    length_ratio = token_count / average_length
    for k1, b in parameters:
        norm = k1 * (1.0 - b + b * length_ratio)
        score = sum(
            term_idf * ((k1 + 1.0) * tf) / (tf + norm)
            for tf, term_idf in term_frequencies
        )
        previous = best[(k1, b)].get(document_id)
        if previous is None or score > previous[0] or (
            score == previous[0] and chunk_index < previous[1]
        ):
            best[(k1, b)][document_id] = (score, chunk_index, chunk_id)


def retrieve_parameter_grid(
    connection: sqlite3.Connection,
    query: str,
    chunk_count: int,
    average_length: float,
    parameters: Sequence[tuple[float, float]],
) -> dict[tuple[float, float], list[dict[str, Any]]]:
    terms = list(dict.fromkeys(tokenize(query)))
    best = {parameter: {} for parameter in parameters}
    if not terms:
        return {parameter: [] for parameter in parameters}
    placeholders = ",".join("?" for _ in terms)
    frequencies = {
        term: count
        for term, count in connection.execute(
            f"SELECT term, doc FROM vocab_row WHERE term IN ({placeholders})",
            terms,
        )
    }
    if not frequencies:
        return {parameter: [] for parameter in parameters}
    term_idf = {
        term: idf(chunk_count, frequencies[term]) for term in frequencies
    }
    present_terms = list(frequencies)
    present_term_set = set(present_terms)
    cursor = connection.execute(
        """
        SELECT f.tokens, m.document_id, m.chunk_id, m.chunk_index, m.token_count
        FROM chunks_fts AS f
        JOIN chunk_meta AS m ON m.rowid = f.rowid
        WHERE chunks_fts MATCH ?
        """,
        (fts_query(present_terms),),
    )
    for token_text, document_id, chunk_id, chunk_index, token_count in cursor:
        counts: dict[str, int] = {}
        for token in token_text.split():
            if token in present_term_set:
                counts[token] = counts.get(token, 0) + 1
        finalize_chunk(
            [(tf, term_idf[term]) for term, tf in counts.items()],
            token_count,
            document_id,
            chunk_id,
            chunk_index,
            average_length,
            parameters,
            best,
        )
    cursor.close()

    output: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for parameter in parameters:
        ordered = sorted(
            best[parameter].items(), key=lambda item: (-item[1][0], item[0])
        )[:TOP_K]
        output[parameter] = [
            {
                "rank": rank,
                "document_id": document_id,
                "representative_chunk_id": values[2],
                "score": values[0],
            }
            for rank, (document_id, values) in enumerate(ordered, 1)
        ]
    return output


def metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ranks = [row["reference_document_rank"] for row in rows]
    result: dict[str, Any] = {"query_count": len(rows)}
    for cutoff in (1, 5, 10, 20, 50, 100):
        hits = sum(rank is not None and rank <= cutoff for rank in ranks)
        result[f"hits_at_{cutoff}"] = hits
        result[f"recall_at_{cutoff}"] = hits / len(rows)
    result["mrr_at_10"] = sum(
        1.0 / rank if rank is not None and rank <= 10 else 0.0 for rank in ranks
    ) / len(rows)
    return result


def evaluate_config(
    config: ChunkConfig,
    index_path: Path,
    index_stats: dict[str, Any],
    queries: Sequence[dict[str, Any]],
    parameters: Sequence[tuple[float, float]],
    results_dir: Path,
) -> list[dict[str, Any]]:
    rows = {parameter: [] for parameter in parameters}
    connection = sqlite3.connect(index_path)
    connection.execute(
        "CREATE VIRTUAL TABLE temp.vocab_row USING "
        "fts5vocab(main, chunks_fts, 'row')"
    )
    connection.execute("PRAGMA query_only=ON")
    started = time.perf_counter()
    try:
        for query_number, query_record in enumerate(queries, 1):
            question_id = query_record["question_id"]
            reference_id = query_record["reference_document_id"]
            rankings = retrieve_parameter_grid(
                connection,
                query_record["query"],
                index_stats["chunk_count"],
                index_stats["average_token_count"],
                parameters,
            )
            for parameter in parameters:
                top_documents = rankings[parameter]
                if len(top_documents) != TOP_K:
                    raise ValueError(
                        f"{question_id} returned {len(top_documents)} documents "
                        f"for {parameter_name(*parameter)}"
                    )
                reference_rank = next(
                    (
                        item["rank"]
                        for item in top_documents
                        if item["document_id"] == reference_id
                    ),
                    None,
                )
                rows[parameter].append(
                    {
                        "question_id": question_id,
                        "reference_document_id": reference_id,
                        "reference_document_rank": reference_rank,
                        "top_documents": top_documents,
                    }
                )
            if len(queries) > 20 and query_number % 100 == 0:
                print(
                    f"[{config.name}] queries {query_number}/{len(queries)}",
                    flush=True,
                )
    finally:
        connection.close()
    elapsed = time.perf_counter() - started

    summaries: list[dict[str, Any]] = []
    for parameter in parameters:
        k1, b = parameter
        name = parameter_name(k1, b)
        output_dir = results_dir / config.name / name
        output_dir.mkdir(parents=True, exist_ok=True)
        rankings_path = output_dir / "rankings_top100.jsonl"
        write_jsonl(rankings_path, rows[parameter])
        metric = metrics(rows[parameter])
        summary = {
            "chunk_config": asdict(config),
            "k1": k1,
            "b": b,
            "baseline": parameter == BASELINE,
            "evaluation_split": "TRAIN only",
            "metrics": metric,
            "artifacts": {
                "rankings": display_path(rankings_path),
                "index": display_path(index_path),
            },
            "ranking": {
                "depth": TOP_K,
                "document_aggregation": "maximum BM25 chunk score per document",
                "representative_chunk": "highest-scoring chunk; lower chunk index on tie",
                "tie_break": "document_id ascending",
                "tokenization": TOKENIZATION_DESCRIPTION,
            },
            "shared_config_query_seconds": elapsed,
        }
        write_json(output_dir / "metrics.json", summary)
        summaries.append(summary)
    return summaries


def validate_arguments(args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.query_limit < 0 or args.query_limit > 450:
        raise ValueError("--query-limit must be between 0 and 450")
    if any(value <= 0 for value in args.k1_values):
        raise ValueError("all --k1-values must be positive")
    if any(value < 0 or value > 1 for value in args.b_values):
        raise ValueError("all --b-values must be between 0 and 1")
    parameters = list(
        dict.fromkeys((float(k1), float(b)) for k1 in args.k1_values for b in args.b_values)
    )
    return parameters


def main() -> int:
    args = parse_args()
    parameters = validate_arguments(args)
    corpus = args.corpus.resolve()
    query_path = args.queries.resolve()
    results_dir = args.results_dir.resolve()
    index_dir = args.index_dir.resolve()
    if not corpus.is_file() or not query_path.is_file():
        raise FileNotFoundError(f"missing corpus or query file: {corpus}, {query_path}")
    queries = list(jsonl_rows(query_path))
    if len(queries) != 450:
        raise ValueError(f"expected 450 TRAIN queries, found {len(queries)}: {query_path}")
    if args.query_limit:
        queries = queries[: args.query_limit]
    results_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    for config_name in args.configs:
        config = CONFIGS[config_name]
        index_path = index_dir / f"{config.name}.sqlite"
        try:
            print(f"[{config.name}] preparing shared index", flush=True)
            index_stats = build_index(corpus, index_path, config)
            print(
                f"[{config.name}] evaluating {len(queries)} queries x "
                f"{len(parameters)} parameters",
                flush=True,
            )
            summaries = evaluate_config(
                config,
                index_path,
                index_stats,
                queries,
                parameters,
                results_dir,
            )
            all_summaries.extend(summaries)
        except Exception as error:
            raise RuntimeError(
                f"failed config={config.name}, corpus={corpus}, "
                f"queries={query_path}, index={index_path}: {error}"
            ) from error

    overall = {
        "status": "complete",
        "evaluation_split": "TRAIN only",
        "query_count": len(queries),
        "full_grid": len(queries) == 450
        and len(args.configs) == len(CONFIGS)
        and len(parameters) == len(DEFAULT_K1) * len(DEFAULT_B),
        "configuration_count": len(all_summaries),
        "chunk_configs": list(args.configs),
        "k1_values": list(dict.fromkeys(args.k1_values)),
        "b_values": list(dict.fromkeys(args.b_values)),
        "baseline": {"k1": BASELINE[0], "b": BASELINE[1]},
        "results": all_summaries,
    }
    summary_path = results_dir / "summary.json"
    write_json(summary_path, overall)
    print(summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
