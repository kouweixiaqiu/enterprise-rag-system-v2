#!/usr/bin/env python3
"""Build a conservative no-evidence subset from the 150 TechQA TRAIN N cases.

The exported subset is scoped to the saved fixed_600_0 PPLX Top-5 plus
neighbor context.  It must not be interpreted as proof that no answer exists
anywhere in the complete shared corpus.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / (
    "experiments/generation/results/"
    "siliconflow_train150_unanswerable_pplx_fixed_600_0_top5_neighbors_refusal"
)
OUTPUT_DIR = RUN_DIR / "evidence_audit"

# Conservative manual review of the saved question, actual supplied sources,
# and generated refusal/answer.  Ambiguous and partly useful evidence is kept
# out of the no-evidence export.
NO_EVIDENCE = {
    "TRAIN_Q004", "TRAIN_Q035", "TRAIN_Q074", "TRAIN_Q079", "TRAIN_Q088",
    "TRAIN_Q094", "TRAIN_Q105", "TRAIN_Q119", "TRAIN_Q146", "TRAIN_Q153",
    "TRAIN_Q156", "TRAIN_Q169", "TRAIN_Q176", "TRAIN_Q185", "TRAIN_Q191",
    "TRAIN_Q196", "TRAIN_Q197", "TRAIN_Q213", "TRAIN_Q245", "TRAIN_Q246",
    "TRAIN_Q264", "TRAIN_Q268", "TRAIN_Q275", "TRAIN_Q281", "TRAIN_Q289",
    "TRAIN_Q293", "TRAIN_Q308", "TRAIN_Q311", "TRAIN_Q323", "TRAIN_Q329",
    "TRAIN_Q342", "TRAIN_Q364", "TRAIN_Q381", "TRAIN_Q404", "TRAIN_Q409",
    "TRAIN_Q418", "TRAIN_Q432", "TRAIN_Q447", "TRAIN_Q476", "TRAIN_Q479",
    "TRAIN_Q481", "TRAIN_Q484", "TRAIN_Q485", "TRAIN_Q486", "TRAIN_Q498",
    "TRAIN_Q508", "TRAIN_Q510", "TRAIN_Q530", "TRAIN_Q531", "TRAIN_Q532",
    "TRAIN_Q534", "TRAIN_Q558", "TRAIN_Q563", "TRAIN_Q571", "TRAIN_Q591",
}

SUFFICIENT_EVIDENCE = {
    # Existing ANSWER decisions whose claims are materially supported.
    "TRAIN_Q021", "TRAIN_Q022", "TRAIN_Q023", "TRAIN_Q095", "TRAIN_Q099",
    "TRAIN_Q138", "TRAIN_Q142", "TRAIN_Q143", "TRAIN_Q148", "TRAIN_Q157",
    "TRAIN_Q175", "TRAIN_Q211", "TRAIN_Q220", "TRAIN_Q238", "TRAIN_Q273",
    "TRAIN_Q284", "TRAIN_Q327", "TRAIN_Q356", "TRAIN_Q405", "TRAIN_Q420",
    "TRAIN_Q434", "TRAIN_Q436", "TRAIN_Q459", "TRAIN_Q468", "TRAIN_Q502",
    "TRAIN_Q509", "TRAIN_Q527", "TRAIN_Q542", "TRAIN_Q566", "TRAIN_Q594",
    # Refusals where the supplied text nevertheless contains the direct
    # product/version/problem evidence needed for an answer.
    "TRAIN_Q057", "TRAIN_Q133", "TRAIN_Q147", "TRAIN_Q179", "TRAIN_Q180",
    "TRAIN_Q181", "TRAIN_Q192", "TRAIN_Q218", "TRAIN_Q249", "TRAIN_Q251",
    "TRAIN_Q253", "TRAIN_Q257", "TRAIN_Q265", "TRAIN_Q282", "TRAIN_Q290",
    "TRAIN_Q292", "TRAIN_Q317", "TRAIN_Q325", "TRAIN_Q335", "TRAIN_Q354",
    "TRAIN_Q377", "TRAIN_Q407", "TRAIN_Q419", "TRAIN_Q464", "TRAIN_Q480",
    "TRAIN_Q491", "TRAIN_Q506", "TRAIN_Q529",
}


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    contexts = rows(RUN_DIR / "contexts.jsonl")
    generations = {
        row["question_id"]: row for row in rows(RUN_DIR / "generations.jsonl")
    }
    context_ids = [row["question_id"] for row in contexts]
    if len(context_ids) != 150 or len(set(context_ids)) != 150:
        raise ValueError("expected 150 distinct saved contexts")
    if set(context_ids) != set(generations):
        raise ValueError("contexts and generations have different question sets")
    if NO_EVIDENCE & SUFFICIENT_EVIDENCE:
        raise ValueError("review label sets overlap")
    unknown = (NO_EVIDENCE | SUFFICIENT_EVIDENCE) - set(context_ids)
    if unknown:
        raise ValueError(f"review contains unknown IDs: {sorted(unknown)}")

    audit = []
    no_answer = []
    for context in contexts:
        question_id = context["question_id"]
        generation = generations[question_id]
        if question_id in NO_EVIDENCE:
            label = "无证据"
            decision = "纳入当前检索上下文无答案集"
        elif question_id in SUFFICIENT_EVIDENCE:
            label = "证据足够"
            decision = "排除"
        else:
            label = "部分证据或仍需核实"
            decision = "排除（保守处理）"

        generated_reason = " ".join(
            line.strip()
            for line in generation["answer"].splitlines()[1:]
            if line.strip()
        )
        item = {
            "question_id": question_id,
            "original_answerable_label": context["answerable_label"],
            "question": context["question"],
            "evidence_scope": "fixed_600_0 PPLX Top-5 representative chunks with neighbors",
            "evidence_label": label,
            "set_decision": decision,
            "saved_generation_decision": generation["decision"],
            "review_reason": generated_reason,
            "top_documents": context["top_documents"],
            "sources": context["sources"],
        }
        audit.append(item)
        if question_id in NO_EVIDENCE:
            no_answer.append(item)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "evidence_review.jsonl").open("w", encoding="utf-8") as out:
        for row in audit:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUTPUT_DIR / "no_answer_current_top5.jsonl").open("w", encoding="utf-8") as out:
        for row in no_answer:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter(row["evidence_label"] for row in audit)
    summary = {
        "source_question_count": len(audit),
        "labels": dict(counts),
        "no_answer_subset_count": len(no_answer),
        "no_answer_subset_ids": [row["question_id"] for row in no_answer],
        "scope": "saved fixed_600_0 PPLX Top-5 representative chunks with neighbors",
        "important_limitation": (
            "No evidence in the saved Top-5 context is not proof that no answer "
            "exists elsewhere in the 28,482-document shared corpus."
        ),
        "source_contexts": str((RUN_DIR / "contexts.jsonl").relative_to(ROOT)),
        "source_generations": str((RUN_DIR / "generations.jsonl").relative_to(ROOT)),
        "review_file": str((OUTPUT_DIR / "evidence_review.jsonl").relative_to(ROOT)),
        "no_answer_file": str((OUTPUT_DIR / "no_answer_current_top5.jsonl").relative_to(ROOT)),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
