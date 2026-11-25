#!/usr/bin/env python3
"""Run the selected three TRAIN Top-5 generation chains sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT_ROOT / "scripts/run_train_generation_from_rankings.py"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "experiments/generation/results/siliconflow_train450_three_chain_top5"
)
DEFAULT_PROMPT_TEMPLATE = (
    PROJECT_ROOT / "configs/generation/answer_with_citations_direct_conditions.prompt"
)
A_RANKINGS = (
    PROJECT_ROOT
    / "experiments/hybrid/results/pplx_bm25_rrf_grid_train/fixed_1200_0/"
    "k1_1.6_b_0.5/bm25_d100_pplx_d100_pplxw3_bm25w1_k60/rrf_rankings.jsonl"
)
B_RANKINGS = (
    PROJECT_ROOT
    / "experiments/dense/results/"
    "qwen3_reranker_4b_bf16_top50_fixed_1200_0_rrf_train450/"
    "fixed_1200_0_rrf/rankings.jsonl"
)
C_RANKINGS = (
    PROJECT_ROOT
    / "experiments/hybrid/results/pplx_bm25_rrf_grid_train/paragraph_merge_1200_0/"
    "k1_1.6_b_1/bm25_d50_pplx_d100_pplxw2_bm25w1_k60/rrf_rankings.jsonl"
)
FIXED_METADATA = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/"
    "pplx-embed-v1-0.6b_fixed_1200_0/chunk_metadata.jsonl"
)
PARAGRAPH_METADATA = (
    PROJECT_ROOT
    / "data/processed/techqa/dense_cache/"
    "pplx-embed-v1-0.6b_paragraph_merge_1200_0/chunk_metadata.jsonl"
)


GROUPS = (
    (
        "A_fixed_1200_0_rrf_original",
        A_RANKINGS,
        FIXED_METADATA,
        "fixed_1200_0",
    ),
    (
        "B_fixed_1200_0_rrf_qwen3_4b_bf16_reranked",
        B_RANKINGS,
        FIXED_METADATA,
        "fixed_1200_0",
    ),
    (
        "C_paragraph_merge_1200_0_rrf_original",
        C_RANKINGS,
        PARAGRAPH_METADATA,
        "paragraph_merge_1200_0",
    ),
)


def jsonl_rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def normalized_candidates(row: dict) -> list[tuple[str, str]]:
    if isinstance(row.get("fused_documents"), list):
        return [
            (
                candidate["document_id"],
                candidate.get("pplx_representative_chunk_id")
                or candidate["bm25_representative_chunk_id"],
            )
            for candidate in row["fused_documents"][:50]
        ]
    if isinstance(row.get("reranked_documents"), list):
        return [
            (candidate["document_id"], candidate["scored_chunk_id"])
            for candidate in row["reranked_documents"][:50]
        ]
    raise ValueError(f"unsupported ranking format for {row.get('question_id')}")


def validate_inputs(question_ids: list[str] | None) -> dict[str, object]:
    for path in (A_RANKINGS, B_RANKINGS, C_RANKINGS, FIXED_METADATA, PARAGRAPH_METADATA):
        if not path.is_file():
            raise FileNotFoundError(path)
    a = {row["question_id"]: row for row in jsonl_rows(A_RANKINGS)}
    b = {row["question_id"]: row for row in jsonl_rows(B_RANKINGS)}
    c = {row["question_id"]: row for row in jsonl_rows(C_RANKINGS)}
    if len(a) != 450 or len(b) != 450 or len(c) != 450 or set(a) != set(b) or set(a) != set(c):
        raise ValueError("the three ranking files do not cover the same 450 TRAIN questions")
    selected = question_ids or list(a)
    if not set(selected) <= set(a):
        raise ValueError("selected question is absent from a ranking file")
    for question_id in selected:
        a_candidates = normalized_candidates(a[question_id])
        b_candidates = normalized_candidates(b[question_id])
        if len(a_candidates) != 50 or len(b_candidates) != 50:
            raise ValueError(f"A/B candidate depth is not 50 for {question_id}")
        if set(a_candidates) != set(b_candidates):
            raise ValueError(
                f"A/B differ in candidates or representative chunks, not only order: {question_id}"
            )
    first = selected[0]
    return {
        "status": "passed",
        "question_count": len(selected),
        "a_b_same_top50_document_chunk_pairs": True,
        "example_question_id": first,
        "a_top5": normalized_candidates(a[first])[:5],
        "b_top5": normalized_candidates(b[first])[:5],
        "a_b_top5_order_differs": normalized_candidates(a[first])[:5]
        != normalized_candidates(b[first])[:5],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--question-id", action="append")
    parser.add_argument(
        "--question-ids-file",
        type=Path,
        help="JSON list, or an object containing question_ids, in the required order.",
    )
    parser.add_argument(
        "--prompt-template", type=Path, default=DEFAULT_PROMPT_TEMPLATE
    )
    return parser.parse_args()


def load_question_ids(path: Path | None, explicit: list[str] | None) -> list[str] | None:
    if path is None:
        return explicit
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    ids = payload.get("question_ids") if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
        raise ValueError(f"invalid question ID file: {resolved}")
    if explicit:
        raise ValueError("use either --question-ids-file or --question-id, not both")
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate question ID in {resolved}")
    return ids


def main() -> int:
    args = parse_args()
    question_ids = load_question_ids(args.question_ids_file, args.question_id)
    prompt_template = (
        args.prompt_template
        if args.prompt_template.is_absolute()
        else PROJECT_ROOT / args.prompt_template
    )
    if not prompt_template.is_file():
        raise FileNotFoundError(prompt_template)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preflight = validate_inputs(question_ids)
    preflight["prompt_template"] = str(prompt_template)
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log_path = output_root / "run.log"
    with log_path.open("a", encoding="utf-8") as log:
        line = "preflight=" + json.dumps(preflight, ensure_ascii=False)
        print(line, flush=True)
        log.write(line + "\n")
        for index, (name, rankings, metadata, chunk_config) in enumerate(GROUPS, 1):
            output_dir = output_root / name
            command = [
                sys.executable,
                str(GENERATOR),
                "--question-scope",
                "all_train",
                "--rankings",
                str(rankings),
                "--chunk-metadata",
                str(metadata),
                "--chunk-size",
                "1200",
                "--chunk-overlap",
                "0",
                "--chunk-config",
                chunk_config,
                "--context-mode",
                "representative_only",
                "--no-legacy-reuse",
                "--output-dir",
                str(output_dir),
                "--prompt-template",
                str(prompt_template),
            ]
            for question_id in question_ids or []:
                command.extend(("--question-id", question_id))
            heading = f"[{index}/3] starting {name}"
            print(heading, flush=True)
            log.write(heading + "\n"); log.flush()
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for child_line in process.stdout:
                print(child_line, end="", flush=True)
                log.write(f"[{name}] {child_line}"); log.flush()
            code = process.wait()
            if code != 0:
                raise RuntimeError(f"{name} exited with code {code}")
    print(f"completed=3/3 output_root={output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
