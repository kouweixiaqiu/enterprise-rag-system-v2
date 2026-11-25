#!/usr/bin/env python3
"""Save the paired Codex review for the answerable sample-30 refusal-prompt run."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
NEW = (
    ROOT
    / "experiments/generation/results/"
    "siliconflow_train30_answerable_pplx_fixed_600_0_top5_neighbors_refusal/"
    "generations.jsonl"
)
OLD = (
    ROOT
    / "experiments/generation/results/"
    "siliconflow_sample30_dense_vs_c_prompt_v2_codex_review/dense_reviews.jsonl"
)
OUTPUT = (
    ROOT
    / "experiments/generation/results/"
    "siliconflow_train30_answerable_pplx_fixed_600_0_top5_neighbors_refusal/"
    "codex_review"
)


PASS_REASONS = {
    "TRAIN_Q188": "Answers both requested JazzSM inspection paths: page/widgets/wires and an individual widget, with supporting citations.",
    "TRAIN_Q162": "Explains the HTTPComponents visibility change and the required application-bundled libraries that resolve the VerifyError.",
    "TRAIN_Q294": "Gives the compatible CICS Explorer maintenance requirement and the applicable CICS TS V5.3 condition.",
    "TRAIN_Q015": "Keeps the IIB V9/WMB V8 Hypervisor Edition and RHEL scope and gives the supported remediation direction.",
    "TRAIN_Q446": "Directly and correctly states that DataPower is unaffected by the two Bash CVEs asked about.",
    "TRAIN_Q140": "Includes the complete CPE .NET API reinstall procedure needed to resolve the ICC installation error.",
}

ANSWER_FAILURES = {
    "TRAIN_Q441": "Still omits the required fileServingEnabled=true prerequisite and does not preserve the complete supported RewriteRule chain.",
    "TRAIN_Q144": "Gives a fix link but omits the requested vulnerability impact, CVSS base score, and other core bulletin details.",
    "TRAIN_Q462": "Only points to the retrieved bulletin instead of answering with its mechanism, impact, severity, affected versions, and remediation.",
    "TRAIN_Q054": "Mixes unrelated heap-size advice with the 64-bit case and does not accurately explain the compressed-references low-4-GB native-memory cause.",
    "TRAIN_Q018": "Recommends timeout changes but does not answer why Managed Pages makes virtual-portal deletion exceed the server or WCM transaction timeout.",
    "TRAIN_Q154": "Uses the Subscription/Gatekeeper workaround instead of the Statistics 22 prerequisite to install Apple Java SE 6 first.",
    "TRAIN_Q482": "Adds an unrelated Oracle 1000-item rule and omits the applicable inlist_threshold behavior and safe upper setting.",
}


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
    )


def main() -> int:
    generated = rows(NEW)
    previous = {row["question_id"]: row for row in rows(OLD)}
    if len(generated) != 30 or len({row["question_id"] for row in generated}) != 30:
        raise ValueError("new generation does not contain 30 distinct questions")
    if set(previous) != {row["question_id"] for row in generated}:
        raise ValueError("old and new sample-30 question IDs differ")

    reviews: list[dict[str, Any]] = []
    for row in generated:
        question_id = row["question_id"]
        first_line = next(
            (line.strip() for line in row["answer"].splitlines() if line.strip()), ""
        )
        decision = first_line.removeprefix("DECISION: ")
        passed = question_id in PASS_REASONS
        if passed:
            reason = PASS_REASONS[question_id]
        elif question_id in ANSWER_FAILURES:
            reason = ANSWER_FAILURES[question_id]
        elif decision == "INSUFFICIENT_CONTEXT":
            reason = (
                "The system refused this answerable TRAIN question, so it cannot be "
                "a complete answer under the established generation standard."
            )
        else:
            raise ValueError(f"missing review rationale for {question_id}")
        old_passed = previous[question_id].get("verdict") == "complete_pass"
        reviews.append(
            {
                "question_id": question_id,
                "reviewer": "Codex",
                "decision": decision,
                "complete_pass": passed,
                "reason": reason,
                "prompt_v2_complete_pass": old_passed,
                "status_change": (
                    "new_pass" if passed and not old_passed
                    else "regression" if old_passed and not passed
                    else "unchanged_pass" if passed
                    else "unchanged_fail"
                ),
            }
        )

    decision_counts = {
        decision: sum(row["decision"] == decision for row in reviews)
        for decision in ("ANSWER", "INSUFFICIENT_CONTEXT", "INVALID_FORMAT")
    }
    old_pass = sum(row["prompt_v2_complete_pass"] for row in reviews)
    new_pass = sum(row["complete_pass"] for row in reviews)
    new_ids = [row["question_id"] for row in reviews if row["status_change"] == "new_pass"]
    regression_ids = [
        row["question_id"] for row in reviews if row["status_change"] == "regression"
    ]
    summary = {
        "reviewer": "Codex",
        "question_count": 30,
        "decision_counts": decision_counts,
        "prompt_v2_complete_pass": old_pass,
        "refusal_prompt_complete_pass": new_pass,
        "new_pass_count": len(new_ids),
        "new_pass_question_ids": new_ids,
        "regression_count": len(regression_ids),
        "regression_question_ids": regression_ids,
        "net_change": new_pass - old_pass,
        "conclusion": (
            "The refusal prompt is too conservative for answerable questions and "
            "should not replace prompt_v2 in its current form."
        ),
    }
    atomic_jsonl(OUTPUT / "reviews.jsonl", reviews)
    atomic_json(OUTPUT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
