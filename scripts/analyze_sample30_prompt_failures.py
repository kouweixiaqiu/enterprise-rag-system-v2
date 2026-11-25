#!/usr/bin/env python3
"""Align sample-30 Top-5 evidence reviews with final A/B/C answer reviews."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/generation/results/siliconflow_train450_three_chain_top5"
PRE = ROOT / "experiments/hybrid/analysis/top5_evidence_sample30_seed42/reviews.jsonl"
POST = ROOT / "experiments/hybrid/analysis/top5_evidence_sample30_seed42_reranked_4b_bf16/reviews.jsonl"
FINAL = BASE / "final_review/final_reviews.jsonl"
OUT_ROWS = BASE / "final_review/sample30_evidence_generation_diagnosis.jsonl"
OUT_SUMMARY = BASE / "final_review/sample30_evidence_generation_summary.json"
GENERATION_FILES = {
    "A": BASE / "A_fixed_1200_0_rrf_original/generations.jsonl",
    "B": BASE / "B_fixed_1200_0_rrf_qwen3_4b_bf16_reranked/generations.jsonl",
    "C": BASE / "C_paragraph_merge_1200_0_rrf_original/generations.jsonl",
}


def load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    pre = load(PRE)
    post = load(POST)
    evidence = {
        "A": {row["question_id"]: row for row in pre if row["chain"] == "fixed_1200_0_rrf"},
        "B": {row["question_id"]: row for row in post if row["chain"] == "fixed_1200_0_rrf"},
        "C": {row["question_id"]: row for row in pre if row["chain"] == "paragraph_merge_1200_0_rrf"},
    }
    generations = {
        chain: {row["question_id"]: row for row in load(path)}
        for chain, path in GENERATION_FILES.items()
    }
    final = {(row["question_id"], row["chain"]): row for row in load(FINAL)}
    question_sets = {frozenset(rows) for rows in evidence.values()}
    if len(question_sets) != 1 or any(len(rows) != 30 for rows in evidence.values()):
        raise ValueError("A/B/C evidence reviews do not contain the same 30 questions")

    output = []
    cross = {}
    cause_counts = Counter()
    prompt_present_failure_questions = set()
    for chain in "ABC":
        cross[chain] = {}
        for label in ("完整", "部分", "无证据"):
            cross[chain][label] = {"total": 0, "complete_pass": 0, "fail": 0, "fail_question_ids": []}
        for question_id, review in evidence[chain].items():
            saved_sources = generations[chain][question_id]["sources"]
            generated_top5 = [
                (row["document_id"], row["representative_chunk_id"]) for row in saved_sources
            ]
            reviewed_top5 = [(row["document_id"], row["chunk_id"]) for row in review["top5"]]
            if generated_top5 != reviewed_top5:
                raise ValueError(f"Top-5 chunk mismatch for {chain} {question_id}")
            verdict = final[question_id, chain]["final_verdict"]
            label = review["evidence_label"]
            cell = cross[chain][label]
            cell["total"] += 1
            if verdict == "complete_pass":
                cell["complete_pass"] += 1
                diagnosis = None
            else:
                cell["fail"] += 1
                cell["fail_question_ids"].append(question_id)
                if label == "完整":
                    diagnosis = "prompt_present_but_answer_unused_or_misused"
                    prompt_present_failure_questions.add(question_id)
                elif label in {"部分", "无证据"}:
                    diagnosis = "prompt_missing_required_evidence"
                else:
                    diagnosis = "unable_to_determine"
                cause_counts[(chain, diagnosis)] += 1
                cause_counts[("ALL", diagnosis)] += 1
            output.append({
                "question_id": question_id,
                "chain": chain,
                "evidence_label": label,
                "final_verdict": verdict,
                "failure_diagnosis": diagnosis,
                "evidence_review_reason": review["key_evidence_or_missing"],
                "final_review_reason": final[question_id, chain]["review_reason"],
                "top5": review["top5"],
            })

    summary = {
        "scope": {"questions": 30, "chains": 3, "question_chain_cases": 90},
        "top5_chunk_alignment": "90/90 exact",
        "cross_table": cross,
        "failed_case_diagnosis": {
            chain: {
                key: cause_counts[(chain, key)]
                for key in (
                    "prompt_missing_required_evidence",
                    "prompt_present_but_answer_unused_or_misused",
                    "unable_to_determine",
                )
            }
            for chain in ("A", "B", "C", "ALL")
        },
        "prompt_present_but_failed_unique_questions": {
            "count": len(prompt_present_failure_questions),
            "question_ids": sorted(prompt_present_failure_questions),
        },
    }
    OUT_ROWS.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8"
    )
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
