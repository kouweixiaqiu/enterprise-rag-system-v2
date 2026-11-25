#!/usr/bin/env python3
"""Fuse two saved document rankings with reusable weighted RRF configurations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BM25_ROOT = (
    PROJECT_ROOT / "experiments/chunk/bm25_parameter_grid_train"
)
DEFAULT_PPLX_ROOT = (
    PROJECT_ROOT / "experiments/dense/results/pplx_embed_v1_0.6b_six_chunk_train"
)
EVALUATION_CUTOFFS = (1, 5, 10, 20, 50, 100)


@dataclass(frozen=True)
class RunConfig:
    name: str
    bm25_depth: int
    pplx_depth: int
    bm25_weight: float
    pplx_weight: float
    rrf_k: int


@dataclass(frozen=True)
class BatchInput:
    chunk_config: str
    bm25_config: str
    k1: float
    b: float
    bm25_path: Path
    pplx_path: Path


def parse_run(value: str) -> RunConfig:
    parts = value.split(",")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "--run must be NAME,BM25_DEPTH,PPLX_DEPTH,BM25_WEIGHT,PPLX_WEIGHT,RRF_K"
        )
    name = parts[0]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError("run NAME may contain only letters, digits, ._- ")
    try:
        config = RunConfig(
            name=name,
            bm25_depth=int(parts[1]),
            pplx_depth=int(parts[2]),
            bm25_weight=float(parts[3]),
            pplx_weight=float(parts[4]),
            rrf_k=int(parts[5]),
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid --run numeric value: {value}") from error
    if config.bm25_depth <= 0 or config.pplx_depth <= 0:
        raise argparse.ArgumentTypeError("candidate depths must be positive")
    if config.bm25_weight <= 0 or config.pplx_weight <= 0:
        raise argparse.ArgumentTypeError("route weights must be positive")
    if config.rrf_k < 0:
        raise argparse.ArgumentTypeError("RRF k must be non-negative")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--bm25-rankings", type=Path)
    parser.add_argument("--pplx-rankings", type=Path)
    parser.add_argument("--chunk-config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run,
        metavar="NAME,BM25_DEPTH,PPLX_DEPTH,BM25_WEIGHT,PPLX_WEIGHT,RRF_K",
        help="Repeat to evaluate multiple configurations over the same two rankings.",
    )
    parser.add_argument("--bm25-root", type=Path, default=DEFAULT_BM25_ROOT)
    parser.add_argument("--pplx-root", type=Path, default=DEFAULT_PPLX_ROOT)
    parser.add_argument("--bm25-depths", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--pplx-depths", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--pplx-weights", type=float, nargs="+", default=[1, 2, 3])
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--batch-chunk-configs",
        nargs="+",
        help="Optional batch filter used for targeted reruns or validation.",
    )
    parser.add_argument(
        "--batch-bm25-configs",
        nargs="+",
        help="Optional batch filter such as k1_1.2_b_0.75.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=0,
        help="Validation only: use the first N aligned questions; 0 uses all questions.",
    )
    args = parser.parse_args()
    if args.query_limit < 0:
        parser.error("--query-limit must be non-negative")
    if args.batch:
        if args.run or args.bm25_rankings or args.pplx_rankings or args.chunk_config:
            parser.error(
                "--batch cannot be combined with --run, --bm25-rankings, "
                "--pplx-rankings, or --chunk-config"
            )
        if any(value <= 0 or value > 100 for value in args.bm25_depths):
            parser.error("--bm25-depths must be between 1 and 100")
        if any(value <= 0 or value > 100 for value in args.pplx_depths):
            parser.error("--pplx-depths must be between 1 and 100")
        if any(value <= 0 for value in args.pplx_weights) or args.bm25_weight <= 0:
            parser.error("batch route weights must be positive")
        if args.rrf_k < 0:
            parser.error("--rrf-k must be non-negative")
    else:
        if not all((args.bm25_rankings, args.pplx_rankings, args.chunk_config, args.run)):
            parser.error(
                "single mode requires --bm25-rankings, --pplx-rankings, "
                "--chunk-config, and at least one --run"
            )
        names = [config.name for config in args.run]
        if len(names) != len(set(names)):
            parser.error("--run names must be unique")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"ranking file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON: {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            output.write("\n")
    os.replace(temporary, path)


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def log_line(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as output:
        output.write(f"{timestamp} {message}\n")


def index_rows(rows: Sequence[dict[str, Any]], route: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, 1):
        question_id = row.get("question_id")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(f"{route} row {row_number} has no question_id")
        if question_id in output:
            raise ValueError(f"{route} has duplicate question_id: {question_id}")
        if not isinstance(row.get("reference_document_id"), str):
            raise ValueError(f"{route} {question_id} has no reference_document_id")
        documents = row.get("top_documents")
        if not isinstance(documents, list):
            raise ValueError(f"{route} {question_id} has no top_documents list")
        output[question_id] = row
    return output


def validate_documents(
    row: dict[str, Any], route: str, required_depth: int
) -> list[dict[str, Any]]:
    question_id = row["question_id"]
    documents = row["top_documents"]
    if len(documents) < required_depth:
        raise ValueError(
            f"{route} {question_id} has {len(documents)} candidates, "
            f"needs {required_depth}"
        )
    selected = documents[:required_depth]
    ids: set[str] = set()
    for position, document in enumerate(selected, 1):
        document_id = document.get("document_id")
        chunk_id = document.get("representative_chunk_id")
        if document.get("rank") != position:
            raise ValueError(f"{route} {question_id} ranks are not 1..{required_depth}")
        if not isinstance(document_id, str) or document_id in ids:
            raise ValueError(f"{route} {question_id} has invalid/duplicate document_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"{route} {question_id} {document_id} has no representative chunk")
        ids.add(document_id)
    return selected


def source_fields(document: dict[str, Any], route: str) -> dict[str, Any]:
    return {
        f"{route}_rank": document["rank"],
        f"{route}_representative_chunk_id": document["representative_chunk_id"],
        f"{route}_score": document.get("score"),
    }


def fuse_question(
    bm25_row: dict[str, Any], pplx_row: dict[str, Any], config: RunConfig
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for route, row, depth, weight in (
        ("bm25", bm25_row, config.bm25_depth, config.bm25_weight),
        ("pplx", pplx_row, config.pplx_depth, config.pplx_weight),
    ):
        for document in validate_documents(row, route, depth):
            document_id = document["document_id"]
            candidate = candidates.setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "rrf_score": 0.0,
                    "bm25_rank": None,
                    "bm25_representative_chunk_id": None,
                    "bm25_score": None,
                    "pplx_rank": None,
                    "pplx_representative_chunk_id": None,
                    "pplx_score": None,
                },
            )
            candidate.update(source_fields(document, route))
            candidate["rrf_score"] += weight / (config.rrf_k + document["rank"])
    ordered = sorted(
        candidates.values(), key=lambda item: (-item["rrf_score"], item["document_id"])
    )
    for rank, item in enumerate(ordered, 1):
        item["rank"] = rank
    return ordered


def reference_rank(reference_id: str, documents: Sequence[dict[str, Any]]) -> int | None:
    return next(
        (item["rank"] for item in documents if item["document_id"] == reference_id),
        None,
    )


def metrics(ranks: Sequence[int | None]) -> dict[str, Any]:
    count = len(ranks)
    output: dict[str, Any] = {"query_count": count}
    for cutoff in EVALUATION_CUTOFFS:
        hits = sum(rank is not None and rank <= cutoff for rank in ranks)
        output[f"hits_at_{cutoff}"] = hits
        output[f"recall_at_{cutoff}"] = hits / count
    return output


def comparison(
    pplx_ranks: Sequence[int | None], fused_ranks: Sequence[int | None]
) -> dict[str, Any]:
    new_at_5 = sum(
        (before is None or before > 5) and after is not None and after <= 5
        for before, after in zip(pplx_ranks, fused_ranks, strict=True)
    )
    dropped_at_5 = sum(
        before is not None and before <= 5 and (after is None or after > 5)
        for before, after in zip(pplx_ranks, fused_ranks, strict=True)
    )
    new_at_20 = sum(
        (before is None or before > 20) and after is not None and after <= 20
        for before, after in zip(pplx_ranks, fused_ranks, strict=True)
    )
    dropped_at_20 = sum(
        before is not None and before <= 20 and (after is None or after > 20)
        for before, after in zip(pplx_ranks, fused_ranks, strict=True)
    )
    pplx_hits_20 = sum(rank is not None and rank <= 20 for rank in pplx_ranks)
    fused_hits_20 = sum(rank is not None and rank <= 20 for rank in fused_ranks)
    return {
        "top_5": {
            "new_hits": new_at_5,
            "dropped_hits": dropped_at_5,
            "net_change": new_at_5 - dropped_at_5,
        },
        "top_20": {
            "pplx_hits": pplx_hits_20,
            "rrf_hits": fused_hits_20,
            "new_hits": new_at_20,
            "dropped_hits": dropped_at_20,
            "net_change": fused_hits_20 - pplx_hits_20,
        },
    }


def run_one(
    config: RunConfig,
    question_ids: Sequence[str],
    bm25_by_id: dict[str, dict[str, Any]],
    pplx_by_id: dict[str, dict[str, Any]],
    output_dir: Path,
    chunk_config: str,
    bm25_path: Path,
    pplx_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fused_ranks: list[int | None] = []
    pplx_ranks: list[int | None] = []
    candidate_counts: list[int] = []
    for question_id in question_ids:
        bm25_row = bm25_by_id[question_id]
        pplx_row = pplx_by_id[question_id]
        reference_id = pplx_row["reference_document_id"]
        fused = fuse_question(bm25_row, pplx_row, config)
        fused_rank = reference_rank(reference_id, fused)
        pplx_rank = reference_rank(reference_id, pplx_row["top_documents"])
        fused_ranks.append(fused_rank)
        pplx_ranks.append(pplx_rank)
        candidate_counts.append(len(fused))
        rows.append(
            {
                "question_id": question_id,
                "reference_document_id": reference_id,
                "reference_document_rank": fused_rank,
                "pplx_reference_document_rank": pplx_rank,
                "fused_documents": fused,
            }
        )

    group_dir = output_dir / config.name
    rankings_path = group_dir / "rrf_rankings.jsonl"
    metrics_path = group_dir / "metrics.json"
    atomic_jsonl(rankings_path, rows)
    result = {
        "status": "complete",
        "evaluation_split": "TRAIN only",
        "chunk_config": chunk_config,
        "configuration": asdict(config),
        "evaluation_cutoffs": list(EVALUATION_CUTOFFS),
        "saved_fused_ranking_depth": "full candidate union",
        "rrf": {
            "formula": "sum(route_weight / (k + route_rank))",
            "candidate_merge_key": "document_id",
            "missing_route_contribution": 0,
            "tie_break": "rrf_score descending, then document_id ascending",
        },
        "metrics": metrics(fused_ranks),
        "pplx_baseline_metrics": metrics(pplx_ranks),
        "comparison_vs_pplx": comparison(pplx_ranks, fused_ranks),
        "candidate_union_count": {
            "min": min(candidate_counts),
            "max": max(candidate_counts),
            "mean": sum(candidate_counts) / len(candidate_counts),
        },
        "sources": {
            "bm25_rankings": display_path(bm25_path),
            "pplx_rankings": display_path(pplx_path),
        },
        "artifacts": {
            "rankings": display_path(rankings_path),
            "metrics": display_path(metrics_path),
        },
    }
    atomic_json(metrics_path, result)
    return result


def aligned_routes(
    bm25_path: Path, pplx_path: Path, query_limit: int, bm25_label: str
) -> tuple[
    list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]
]:
    bm25_by_id = index_rows(read_jsonl(bm25_path), f"BM25[{bm25_label}]")
    pplx_by_id = index_rows(read_jsonl(pplx_path), "PPLX")
    if set(bm25_by_id) != set(pplx_by_id):
        only_bm25 = sorted(set(bm25_by_id) - set(pplx_by_id))
        only_pplx = sorted(set(pplx_by_id) - set(bm25_by_id))
        raise ValueError(
            f"route question sets differ; BM25-only={only_bm25[:5]}, "
            f"PPLX-only={only_pplx[:5]}"
        )
    question_ids = list(pplx_by_id)
    for question_id in question_ids:
        if (
            bm25_by_id[question_id]["reference_document_id"]
            != pplx_by_id[question_id]["reference_document_id"]
        ):
            raise ValueError(f"reference document differs for {question_id}")
    if query_limit:
        question_ids = question_ids[:query_limit]
    if not question_ids:
        raise ValueError("no aligned questions to evaluate")
    return question_ids, bm25_by_id, pplx_by_id


def run_single(args: argparse.Namespace, log_path: Path) -> int:
    args.bm25_rankings = args.bm25_rankings.resolve()
    args.pplx_rankings = args.pplx_rankings.resolve()
    question_ids, bm25_by_id, pplx_by_id = aligned_routes(
        args.bm25_rankings,
        args.pplx_rankings,
        args.query_limit,
        args.chunk_config,
    )
    log_line(
        log_path,
        "START "
        + json.dumps(
            {
                "chunk_config": args.chunk_config,
                "query_count": len(question_ids),
                "bm25_rankings": str(args.bm25_rankings),
                "pplx_rankings": str(args.pplx_rankings),
                "runs": [asdict(config) for config in args.run],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    results: list[dict[str, Any]] = []
    for config in args.run:
        result = run_one(
            config,
            question_ids,
            bm25_by_id,
            pplx_by_id,
            args.output_dir,
            args.chunk_config,
            args.bm25_rankings,
            args.pplx_rankings,
        )
        results.append(result)
        log_line(
            log_path,
            f"COMPLETE name={config.name} "
            f"rankings={result['artifacts']['rankings']} "
            f"metrics={result['artifacts']['metrics']}",
        )
    summary = {
        "status": "complete",
        "evaluation_split": "TRAIN only",
        "chunk_config": args.chunk_config,
        "query_count": len(question_ids),
        "validation_subset": args.query_limit > 0,
        "sources": {
            "bm25_rankings": display_path(args.bm25_rankings),
            "pplx_rankings": display_path(args.pplx_rankings),
        },
        "results": results,
    }
    summary_path = args.output_dir / "summary.json"
    atomic_json(summary_path, summary)
    log_line(log_path, f"FINISH summary={summary_path}")
    print(summary_path, flush=True)
    return 0


def parse_bm25_config(name: str) -> tuple[float, float]:
    match = re.fullmatch(r"k1_([^_]+)_b_([^_]+)", name)
    if match is None:
        raise ValueError(f"invalid BM25 parameter directory: {name}")
    try:
        return float(match.group(1)), float(match.group(2))
    except ValueError as error:
        raise ValueError(f"invalid BM25 parameter directory: {name}") from error


def discover_batch_inputs(args: argparse.Namespace) -> tuple[list[BatchInput], list[str]]:
    bm25_root = args.bm25_root.resolve()
    pplx_root = args.pplx_root.resolve()
    if not bm25_root.is_dir():
        raise FileNotFoundError(f"BM25 root not found: {bm25_root}")
    if not pplx_root.is_dir():
        raise FileNotFoundError(f"PPLX root not found: {pplx_root}")
    if args.batch_chunk_configs:
        chunk_names = list(dict.fromkeys(args.batch_chunk_configs))
    else:
        chunk_names = sorted(
            {path.name for path in bm25_root.iterdir() if path.is_dir()}
            | {path.name for path in pplx_root.iterdir() if path.is_dir()}
        )
    requested_bm25 = (
        list(dict.fromkeys(args.batch_bm25_configs))
        if args.batch_bm25_configs
        else None
    )
    discovered: list[BatchInput] = []
    missing: list[str] = []
    for chunk_name in chunk_names:
        bm25_chunk_dir = bm25_root / chunk_name
        pplx_path = pplx_root / chunk_name / "dense_rankings_top100.jsonl"
        if not bm25_chunk_dir.is_dir():
            missing.append(f"missing BM25 chunk directory: {bm25_chunk_dir}")
            continue
        if not pplx_path.is_file():
            missing.append(f"missing PPLX ranking: {pplx_path}")
            continue
        bm25_names = requested_bm25 or sorted(
            path.name
            for path in bm25_chunk_dir.iterdir()
            if path.is_dir() and path.name.startswith("k1_")
        )
        if not bm25_names:
            missing.append(f"no BM25 parameter directories: {bm25_chunk_dir}")
            continue
        for bm25_name in bm25_names:
            bm25_path = bm25_chunk_dir / bm25_name / "rankings_top100.jsonl"
            if not bm25_path.is_file():
                missing.append(f"missing BM25 ranking: {bm25_path}")
                continue
            try:
                k1, b = parse_bm25_config(bm25_name)
            except ValueError as error:
                missing.append(str(error))
                continue
            discovered.append(
                BatchInput(
                    chunk_config=chunk_name,
                    bm25_config=bm25_name,
                    k1=k1,
                    b=b,
                    bm25_path=bm25_path.resolve(),
                    pplx_path=pplx_path.resolve(),
                )
            )
    discovered.sort(key=lambda item: (item.chunk_config, item.k1, item.b))
    return discovered, missing


def number_label(value: float) -> str:
    return f"{value:g}"


def batch_run_configs(args: argparse.Namespace) -> list[RunConfig]:
    bm25_depths = list(dict.fromkeys(args.bm25_depths))
    pplx_depths = list(dict.fromkeys(args.pplx_depths))
    pplx_weights = list(dict.fromkeys(args.pplx_weights))
    configs: list[RunConfig] = []
    for bm25_depth, pplx_depth, pplx_weight in product(
        bm25_depths, pplx_depths, pplx_weights
    ):
        name = (
            f"bm25_d{bm25_depth}_pplx_d{pplx_depth}_"
            f"pplxw{number_label(pplx_weight)}_"
            f"bm25w{number_label(args.bm25_weight)}_k{args.rrf_k}"
        )
        configs.append(
            RunConfig(
                name=name,
                bm25_depth=bm25_depth,
                pplx_depth=pplx_depth,
                bm25_weight=args.bm25_weight,
                pplx_weight=pplx_weight,
                rrf_k=args.rrf_k,
            )
        )
    return configs


def flat_result_row(batch_input: BatchInput, result: dict[str, Any]) -> dict[str, Any]:
    config = result["configuration"]
    row: dict[str, Any] = {
        "chunk_config": batch_input.chunk_config,
        "bm25_config": batch_input.bm25_config,
        "k1": batch_input.k1,
        "b": batch_input.b,
        "bm25_input_depth": config["bm25_depth"],
        "pplx_input_depth": config["pplx_depth"],
        "bm25_weight": config["bm25_weight"],
        "pplx_weight": config["pplx_weight"],
        "rrf_k": config["rrf_k"],
        "fused_evaluation_cutoffs": "1|5|10|20|50|100",
        "saved_fused_ranking_depth": "full candidate union",
        "query_count": result["metrics"]["query_count"],
    }
    for cutoff in EVALUATION_CUTOFFS:
        row[f"hits_at_{cutoff}"] = result["metrics"][f"hits_at_{cutoff}"]
        row[f"recall_at_{cutoff}"] = result["metrics"][f"recall_at_{cutoff}"]
    for cutoff in (5, 20):
        change = result["comparison_vs_pplx"][f"top_{cutoff}"]
        row[f"top_{cutoff}_new_hits"] = change["new_hits"]
        row[f"top_{cutoff}_dropped_hits"] = change["dropped_hits"]
        row[f"top_{cutoff}_net_change"] = change["net_change"]
    row["rankings_path"] = result["artifacts"]["rankings"]
    row["metrics_path"] = result["artifacts"]["metrics"]
    return row


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_batch_summary(
    args: argparse.Namespace,
    status: str,
    expected_groups: int,
    missing: Sequence[str],
    results: Sequence[dict[str, Any]],
    table_rows: Sequence[dict[str, Any]],
) -> tuple[Path, Path]:
    table_path = args.output_dir / "results.csv"
    summary_path = args.output_dir / "summary.json"
    atomic_csv(table_path, table_rows)
    atomic_json(
        summary_path,
        {
            "status": status,
            "evaluation_split": "TRAIN only",
            "validation_subset": args.query_limit > 0,
            "expected_group_count_from_discovered_inputs": expected_groups,
            "completed_group_count": len(results),
            "missing_inputs": list(missing),
            "grid": {
                "bm25_depths": list(dict.fromkeys(args.bm25_depths)),
                "pplx_depths": list(dict.fromkeys(args.pplx_depths)),
                "pplx_weights": list(dict.fromkeys(args.pplx_weights)),
                "bm25_weight": args.bm25_weight,
                "rrf_k": args.rrf_k,
                "evaluation_cutoffs": list(EVALUATION_CUTOFFS),
            },
            "artifacts": {
                "results_table": display_path(table_path),
                "run_log": display_path(args.output_dir / "run.log"),
            },
            "results": list(results),
        },
    )
    return summary_path, table_path


def run_batch(args: argparse.Namespace, log_path: Path) -> int:
    inputs, missing = discover_batch_inputs(args)
    configs = batch_run_configs(args)
    if not inputs:
        raise ValueError("no complete BM25/PPLX ranking pairs were discovered")
    expected_groups = len(inputs) * len(configs)
    log_line(
        log_path,
        f"START_BATCH expected_groups={expected_groups} "
        f"ranking_pairs={len(inputs)} chunk_configs={len(set(x.chunk_config for x in inputs))}",
    )
    for item in missing:
        log_line(log_path, f"MISSING {item}")

    results: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    chunk_names = list(dict.fromkeys(item.chunk_config for item in inputs))
    for chunk_index, chunk_name in enumerate(chunk_names, 1):
        chunk_inputs = [item for item in inputs if item.chunk_config == chunk_name]
        pplx_path = chunk_inputs[0].pplx_path
        pplx_by_id = index_rows(read_jsonl(pplx_path), f"PPLX[{chunk_name}]")
        chunk_completed = 0
        for batch_input in chunk_inputs:
            bm25_by_id = index_rows(
                read_jsonl(batch_input.bm25_path),
                f"BM25[{chunk_name}/{batch_input.bm25_config}]",
            )
            if set(bm25_by_id) != set(pplx_by_id):
                raise ValueError(
                    f"route question sets differ for {chunk_name}/{batch_input.bm25_config}"
                )
            question_ids = list(pplx_by_id)
            for question_id in question_ids:
                if (
                    bm25_by_id[question_id]["reference_document_id"]
                    != pplx_by_id[question_id]["reference_document_id"]
                ):
                    raise ValueError(
                        f"reference document differs for {chunk_name}/"
                        f"{batch_input.bm25_config}/{question_id}"
                    )
            if args.query_limit:
                question_ids = question_ids[: args.query_limit]
            group_root = (
                args.output_dir / chunk_name / batch_input.bm25_config
            )
            for config in configs:
                result = run_one(
                    config,
                    question_ids,
                    bm25_by_id,
                    pplx_by_id,
                    group_root,
                    chunk_name,
                    batch_input.bm25_path,
                    batch_input.pplx_path,
                )
                result["bm25_parameter_config"] = {
                    "name": batch_input.bm25_config,
                    "k1": batch_input.k1,
                    "b": batch_input.b,
                }
                results.append(result)
                table_rows.append(flat_result_row(batch_input, result))
                chunk_completed += 1
        write_batch_summary(
            args,
            "in_progress" if len(results) < expected_groups else "complete",
            expected_groups,
            missing,
            results,
            table_rows,
        )
        log_line(
            log_path,
            f"CHUNK_COMPLETE {chunk_index}/{len(chunk_names)} "
            f"chunk_config={chunk_name} groups={chunk_completed} "
            f"completed_total={len(results)}/{expected_groups}",
        )
        print(
            f"[{chunk_name}] completed groups={chunk_completed}; "
            f"total={len(results)}/{expected_groups}",
            flush=True,
        )
    summary_path, table_path = write_batch_summary(
        args, "complete", expected_groups, missing, results, table_rows
    )
    log_line(
        log_path,
        f"FINISH_BATCH completed_groups={len(results)}/{expected_groups} "
        f"summary={display_path(summary_path)} table={display_path(table_path)}",
    )
    print(summary_path, flush=True)
    print(table_path, flush=True)
    return 0


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "run.log"
    try:
        if args.batch:
            return run_batch(args, log_path)
        return run_single(args, log_path)
    except Exception as error:
        log_line(log_path, f"ERROR {type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
