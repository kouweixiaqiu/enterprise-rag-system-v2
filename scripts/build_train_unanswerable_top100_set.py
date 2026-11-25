#!/usr/bin/env python3
"""Audit the conservative Top-5 no-evidence candidates against PPLX Top-100."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / (
    "experiments/generation/results/"
    "siliconflow_train150_unanswerable_pplx_fixed_600_0_top5_neighbors_refusal"
)
TOP5_SUMMARY = BASE / "evidence_audit/summary.json"
CONTEXTS = BASE / "contexts.jsonl"
RANKINGS = ROOT / (
    "experiments/dense/results/"
    "pplx_embed_v1_0.6b_fixed_600_0_train_unanswerable/"
    "dense_rankings_top100.jsonl"
)
OUTPUT = BASE / "evidence_audit_top100"

# These labels apply only to the 55 conservative Top-5 no-evidence candidates.
# Candidate ranks identify the Top-100 document that changed the Top-5 result.
SUFFICIENT = {
    "TRAIN_Q004": 1,
    "TRAIN_Q079": 10,
    "TRAIN_Q146": 4,
    "TRAIN_Q185": 2,
    "TRAIN_Q196": 1,
    "TRAIN_Q213": 1,
    "TRAIN_Q264": 6,
    "TRAIN_Q381": 1,
    "TRAIN_Q479": 1,
}

PARTIAL = {
    "TRAIN_Q035": 1,
    "TRAIN_Q105": 52,
    "TRAIN_Q169": 3,
    "TRAIN_Q268": 13,
    "TRAIN_Q289": 2,
    "TRAIN_Q329": 2,
    "TRAIN_Q404": 36,
    "TRAIN_Q447": 40,
    "TRAIN_Q508": 1,
    "TRAIN_Q571": 2,
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    candidate_ids = json.loads(TOP5_SUMMARY.read_text(encoding="utf-8"))[
        "no_answer_subset_ids"
    ]
    if len(candidate_ids) != 55 or len(set(candidate_ids)) != 55:
        raise ValueError("expected the 55 distinct Top-5 no-evidence candidates")
    if set(SUFFICIENT) & set(PARTIAL):
        raise ValueError("Top-100 review labels overlap")
    if not (set(SUFFICIENT) | set(PARTIAL)) <= set(candidate_ids):
        raise ValueError("Top-100 review contains an unknown question ID")

    contexts = {row["question_id"]: row for row in read_jsonl(CONTEXTS)}
    rankings = {row["question_id"]: row for row in read_jsonl(RANKINGS)}
    if not set(candidate_ids) <= set(contexts) or not set(candidate_ids) <= set(rankings):
        raise ValueError("a reviewed question is missing from contexts or rankings")

    reviews: list[dict] = []
    final_no_answer: list[dict] = []
    for question_id in candidate_ids:
        top_documents = rankings[question_id].get("top_documents")
        if not isinstance(top_documents, list) or len(top_documents) != 100:
            raise ValueError(f"{question_id} does not have exactly 100 candidates")
        ranks = {row.get("rank"): row for row in top_documents}

        if question_id in SUFFICIENT:
            label = "证据足够"
            matched_rank = SUFFICIENT[question_id]
            decision = "移出无答案集合"
            reason = "Top-100 中找到能直接支持核心答案及主要操作或条件的候选文档。"
        elif question_id in PARTIAL:
            label = "部分证据"
            matched_rank = PARTIAL[question_id]
            decision = "移出无答案集合（避免把边界题当负例）"
            reason = "Top-100 中找到直接有用信息，但仍缺少问题要求的步骤、条件或精确版本。"
        else:
            label = "无证据"
            matched_rank = None
            decision = "纳入 Top-100 无答案集合"
            reason = "核查 Top-100 候选后，未发现足以支持核心答案的文本；近似结果存在产品、版本、错误码或问题场景不匹配。"

        matched = ranks.get(matched_rank) if matched_rank is not None else None
        row = {
            "question_id": question_id,
            "question": contexts[question_id]["question"],
            "evidence_scope": "fixed_600_0 PPLX document-level Top-100",
            "evidence_label": label,
            "set_decision": decision,
            "reason": reason,
            "matched_candidate_rank": matched_rank,
            "matched_candidate": matched,
            "top_documents": top_documents,
        }
        reviews.append(row)
        if label == "无证据":
            final_no_answer.append(row)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT / "top100_evidence_review.jsonl", reviews)
    write_jsonl(OUTPUT / "no_answer_top100.jsonl", final_no_answer)
    counts = Counter(row["evidence_label"] for row in reviews)
    summary = {
        "input_top5_no_evidence_candidates": len(candidate_ids),
        "top100_review_counts": dict(counts),
        "final_top100_no_answer_count": len(final_no_answer),
        "final_top100_no_answer_ids": [row["question_id"] for row in final_no_answer],
        "removed_as_sufficient_ids": list(SUFFICIENT),
        "removed_as_partial_ids": list(PARTIAL),
        "scope": "fixed_600_0 PPLX document-level Top-100",
        "definition": (
            "No saved Top-100 candidate contains evidence sufficient to answer the "
            "question's core request with its required product/version/conditions."
        ),
        "limitation": (
            "This is an operational retrieval no-answer set, not proof that no answer "
            "exists outside PPLX Top-100 in the complete shared corpus."
        ),
        "ranking_file": str(RANKINGS.relative_to(ROOT)),
        "review_file": str((OUTPUT / "top100_evidence_review.jsonl").relative_to(ROOT)),
        "no_answer_file": str((OUTPUT / "no_answer_top100.jsonl").relative_to(ROOT)),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
