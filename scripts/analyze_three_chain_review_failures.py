#!/usr/bin/env python3
"""Diagnose failure categories in the saved A/B/C generation reviews."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/generation/results/siliconflow_train450_three_chain_top5"
INITIAL = BASE / "siliconflow_initial_review/initial_reviews.jsonl"
MAPPING = BASE / "siliconflow_initial_review/anonymous_mapping.jsonl"
FINAL = BASE / "final_review/final_reviews.jsonl"
QUESTIONS = BASE / "final_review/codex_review_questions.json"
OUTPUT = BASE / "final_review/failure_diagnosis.json"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


CORE_WRONG = re.compile(
    r"incorrect|wrong|contradict|misidentif|not relevant|irrelevant|unrelated|"
    r"different product|different version|cross.product|跨产品|答非所问|"
    r"does not correctly identify|does not address|does not answer|fails to answer|"
    r"core issue|core answer|not the correct",
    re.I,
)
UNSUPPORTED = re.compile(
    r"unsupported|not supported|does not support|no evidence|without evidence|"
    r"evidence (?:does not|doesn't|fails to)|缺少.*(?:证据|上下文)|"
    r"(?:证据|上下文).*(?:不支持|无法支持)",
    re.I,
)
INCOMPLETE = re.compile(
    r"omit|missing|lack|does not mention|does not provide|does not include|"
    r"does not explicitly|necessary step|necessary condition|prerequisite|"
    r"specific detail|incomplete|遗漏|缺少.*(?:步骤|条件|要点)|必要步骤|适用条件",
    re.I,
)


def category(row):
    if row["final_verdict"] == "complete_pass":
        return "complete_pass", None
    # A verdict-changing review takes precedence. Generic mixed review reasons
    # remain unresolved rather than being forced into one cause.
    if row["review_changed"]:
        text = row["review_reason"]
        if "核心要点、必要步骤或适用条件" in text:
            return "pending_verification", None
    else:
        text = row["initial_reason"]
    if CORE_WRONG.search(text):
        return "core_wrong_or_no_answer", None
    if UNSUPPORTED.search(text):
        coverage = row["semantic_diagnostic"]["prompt_evidence_coverage"]
        if coverage < 0.70:
            subtype = "context_missing_evidence"
        elif coverage >= 0.85:
            subtype = "evidence_present_but_unused_or_misused"
        else:
            subtype = "evidence_availability_unclear"
        return "unsupported_key_claim", subtype
    if INCOMPLETE.search(text):
        return "core_correct_but_incomplete", None
    return "pending_verification", None


def pairwise(by_key, verdict_field):
    question_ids = sorted({question_id for question_id, _ in by_key})
    output = {}
    for other in ("A", "C"):
        new = [
            q for q in question_ids
            if by_key[q, "B"][verdict_field] == "complete_pass"
            and by_key[q, other][verdict_field] != "complete_pass"
        ]
        lost = [
            q for q in question_ids
            if by_key[q, "B"][verdict_field] != "complete_pass"
            and by_key[q, other][verdict_field] == "complete_pass"
        ]
        output[f"B_vs_{other}"] = {
            "new_pass": len(new), "regressed": len(lost), "net": len(new) - len(lost),
            "new_pass_question_ids": new, "regressed_question_ids": lost,
        }
    return output


def main() -> int:
    initial_rows = load_jsonl(INITIAL)
    mapping_rows = load_jsonl(MAPPING)
    final_rows = load_jsonl(FINAL)
    review_questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if len(initial_rows) != 1350 or len(mapping_rows) != 1350 or len(final_rows) != 1350:
        raise ValueError("expected 1,350 rows in all aligned artifacts")
    initial = {(row["question_id"], row["chain"]): row for row in initial_rows}
    mapping = {row["anonymous_id"]: (row["question_id"], row["chain"]) for row in mapping_rows}
    if len(initial) != 1350 or len(mapping) != 1350:
        raise ValueError("duplicate initial-review or anonymous mapping key")
    for row in initial_rows:
        if mapping[row["anonymous_id"]] != (row["question_id"], row["chain"]):
            raise ValueError("anonymous mapping mismatch")
    final = {(row["question_id"], row["chain"]): row for row in final_rows}
    if set(initial) != set(final):
        raise ValueError("initial/final review keys differ")

    groups = {}
    categorized_rows = []
    for chain in "ABC":
        chain_rows = [row for row in final_rows if row["chain"] == chain]
        counts = Counter()
        subtype_counts = Counter()
        ids_by_category = {}
        ids_by_subtype = {}
        for row in chain_rows:
            primary, subtype = category(row)
            counts[primary] += 1
            ids_by_category.setdefault(primary, []).append(row["question_id"])
            if subtype:
                subtype_counts[subtype] += 1
                ids_by_subtype.setdefault(subtype, []).append(row["question_id"])
            categorized_rows.append({
                "question_id": row["question_id"], "chain": chain,
                "final_verdict": row["final_verdict"], "failure_category": primary,
                "evidence_subtype": subtype,
            })
        confirmed_core_correct = (
            counts["complete_pass"] + counts["core_correct_but_incomplete"]
            + counts["unsupported_key_claim"]
        )
        groups[chain] = {
            "total": 450,
            "confirmed_core_correct": confirmed_core_correct,
            "complete_pass": counts["complete_pass"],
            "complete_pass_rate": counts["complete_pass"] / 450,
            "failure_counts": {
                "core_wrong_or_no_answer": counts["core_wrong_or_no_answer"],
                "core_correct_but_incomplete": counts["core_correct_but_incomplete"],
                "unsupported_key_claim": counts["unsupported_key_claim"],
                "pending_verification": counts["pending_verification"],
            },
            "unsupported_breakdown": dict(subtype_counts),
            "pending_question_ids": sorted(ids_by_category.get("pending_verification", [])),
            "unsupported_question_ids_by_subtype": {
                key: sorted(value) for key, value in ids_by_subtype.items()
            },
        }

    changes = {}
    for review_set in ("pairwise_difference", "consistent_random_sample"):
        changes[review_set] = {}
        for chain in "ABC":
            selected = [
                row for row in final_rows
                if row["chain"] == chain and row["review_set"] == review_set
            ]
            up = [row["question_id"] for row in selected if row["initial_verdict"] == "fail" and row["final_verdict"] == "complete_pass"]
            down = [row["question_id"] for row in selected if row["initial_verdict"] == "complete_pass" and row["final_verdict"] == "fail"]
            changes[review_set][chain] = {
                "reviewed_questions": len(selected),
                "fail_to_pass": len(up), "pass_to_fail": len(down),
                "fail_to_pass_question_ids": sorted(up),
                "pass_to_fail_question_ids": sorted(down),
            }
    changed = [row for row in final_rows if row["review_changed"]]
    reviewed_unchanged = [row for row in final_rows if row["codex_reviewed"] and not row["review_changed"]]
    result = {
        "scope": {"questions": 450, "chains": 3, "answers": 1350},
        "groups": groups,
        "review_changes": {
            "by_review_set": changes,
            "verdict_changes": len(changed),
            "reason_or_citation_only_within_112_verdict_changes": 0,
            "reviewed_without_verdict_change": len(reviewed_unchanged),
            "difference_question_count": len(review_questions["difference_question_ids"]),
            "consistent_sample_question_count": len(review_questions["consistent_random_sample_question_ids"]),
        },
        "pairwise_before_review": pairwise(initial, "verdict"),
        "pairwise_after_review": pairwise(final, "final_verdict"),
        "categorized_rows": categorized_rows,
        "method_note": (
            "Categories are derived only from saved verdict reasons and diagnostics. "
            "Ambiguous mixed reasons are retained as pending_verification."
        ),
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "categorized_rows"}, ensure_ascii=False, indent=2))
    print(f"output={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
