#!/usr/bin/env python3
"""Save the Codex review of sample30 Dense 600/0 versus C prompt-v2."""

from __future__ import annotations

import json
from pathlib import Path

from finalize_sample30_prompt_ab_codex_review import FAIL_REASON, PASS_REASON


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "experiments/dense/results/qwen3_reranker_4b_bf16_top50_seven_chain_sample30_seed42/sample_question_ids.json"
DENSE_DIR = ROOT / "experiments/generation/results/siliconflow_train450_pplx_fixed_600_0_top5_prompt_v2_representative_with_neighbors"
C_REVIEW = ROOT / "experiments/generation/results/siliconflow_sample30_three_chain_top5_prompt_v2_regeneration/codex_review/reviews.jsonl"
OUTPUT = ROOT / "experiments/generation/results/siliconflow_sample30_dense_vs_c_prompt_v2_codex_review"

DENSE_PASS = {
    "TRAIN_Q069", "TRAIN_Q188", "TRAIN_Q162", "TRAIN_Q144", "TRAIN_Q064",
    "TRAIN_Q462", "TRAIN_Q402", "TRAIN_Q294", "TRAIN_Q018", "TRAIN_Q350",
    "TRAIN_Q015", "TRAIN_Q386", "TRAIN_Q127", "TRAIN_Q494", "TRAIN_Q446",
    "TRAIN_Q482", "TRAIN_Q140",
}
PASS_OVERRIDE = {
    "TRAIN_Q144": "Gives the requested directory-listing impact, CVE identity, base score, affected version, and applicable fix from the supplied bulletin.",
    "TRAIN_Q462": "Gives the authenticated dynamic-queue shared-memory-leak mechanism, impact, base score, affected versions, and fixes.",
    "TRAIN_Q402": "Correctly explains that StaleConnectionException means a pooled database connection is no longer valid and gives supported pool handling guidance.",
}
FAIL_OVERRIDE = {
    "TRAIN_Q014": "Omits the installer-level APAR/next-release conclusion and substitutes prerequisite/manual-install procedures that do not completely answer the issue.",
    "TRAIN_Q415": "Explains the 256-bit policy symptom but omits the required applicability condition that Maximum-strength keys were selected during Workplace XT installation.",
}


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> int:
    ids = json.loads(SAMPLE.read_text(encoding="utf-8"))["question_ids"]
    dense = {r["question_id"]: r for r in load(DENSE_DIR / "generations.jsonl")}
    c_rows = {r["question_id"]: r for r in load(C_REVIEW) if r["chain"] == "C"}
    if not set(ids) <= set(dense) or set(c_rows) != set(ids):
        raise ValueError("Dense and C do not cover the same saved sample30 IDs")
    c_pass = {q for q in ids if c_rows[q]["regenerated_prompt_v2_verdict"] == "complete_pass"}

    reviews = []
    for q in ids:
        passed = q in DENSE_PASS
        reason = (
            PASS_OVERRIDE.get(q, PASS_REASON.get(q, "Core answer and necessary conditions are supported by the supplied Prompt."))
            if passed
            else (FAIL_OVERRIDE[q] if q in FAIL_OVERRIDE else FAIL_REASON[q])
        )
        reviews.append({
            "question_id": q,
            "strategy": "fixed_600_0_pplx_dense_top5_representative_with_neighbors",
            "verdict": "complete_pass" if passed else "fail",
            "reason": reason,
            "citation_map": dense[q].get("citation_map", []),
            "generation_file": str(DENSE_DIR / "generations.jsonl"),
            "context_file": str(DENSE_DIR / "contexts.jsonl"),
        })

    dense_only = sorted(DENSE_PASS - c_pass)
    c_only = sorted(c_pass - DENSE_PASS)
    both = sorted(DENSE_PASS & c_pass)
    neither = sorted(set(ids) - DENSE_PASS - c_pass)
    summary = {
        "reviewer": "Codex",
        "question_count": len(ids),
        "question_ids_match": set(ids) == set(c_rows),
        "dense": {"complete_pass": len(DENSE_PASS), "rate": len(DENSE_PASS) / len(ids)},
        "c": {"complete_pass": len(c_pass), "rate": len(c_pass) / len(ids)},
        "paired": {
            "dense_only": len(dense_only), "dense_only_question_ids": dense_only,
            "c_only": len(c_only), "c_only_question_ids": c_only,
            "both_pass": len(both), "both_pass_question_ids": both,
            "neither_pass": len(neither), "neither_pass_question_ids": neither,
        },
        "dense_configuration": {
            "chunk_config": "fixed_600_0",
            "ranking": "PPLX Dense original Top-5",
            "context_mode": "representative_with_neighbors",
            "prompt": "configs/generation/answer_with_citations_direct_conditions.prompt",
            "model": dense[ids[0]]["model"],
            "temperature": dense[ids[0]]["temperature"],
            "max_tokens": dense[ids[0]]["max_tokens"],
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "dense_reviews.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reviews), encoding="utf-8"
    )
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
