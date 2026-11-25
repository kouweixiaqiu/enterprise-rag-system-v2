#!/usr/bin/env python3
"""Save Codex judgments for the fresh sample30 prompt-v2 regeneration."""

from __future__ import annotations

import json
from pathlib import Path

from finalize_sample30_prompt_ab_codex_review import FAIL_REASON, PASS_REASON


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "experiments/generation/results/siliconflow_train450_three_chain_top5"
REGENERATED = ROOT / "experiments/generation/results/siliconflow_sample30_three_chain_top5_prompt_v2_regeneration"
SCOPE = ROOT / "experiments/generation/results/siliconflow_sample30_three_chain_top5_scope_checked_prompt"
SAMPLE = ROOT / "experiments/dense/results/qwen3_reranker_4b_bf16_top50_seven_chain_sample30_seed42/sample_question_ids.json"
CHAINS = {
    "A": "A_fixed_1200_0_rrf_original",
    "B": "B_fixed_1200_0_rrf_qwen3_4b_bf16_reranked",
    "C": "C_paragraph_merge_1200_0_rrf_original",
}
PASS = {
    "A": {"TRAIN_Q069", "TRAIN_Q014", "TRAIN_Q188", "TRAIN_Q162", "TRAIN_Q064", "TRAIN_Q294", "TRAIN_Q018", "TRAIN_Q350", "TRAIN_Q015", "TRAIN_Q127", "TRAIN_Q446", "TRAIN_Q482", "TRAIN_Q140"},
    "B": {"TRAIN_Q069", "TRAIN_Q188", "TRAIN_Q162", "TRAIN_Q064", "TRAIN_Q054", "TRAIN_Q294", "TRAIN_Q018", "TRAIN_Q154", "TRAIN_Q350", "TRAIN_Q015", "TRAIN_Q386", "TRAIN_Q127", "TRAIN_Q494", "TRAIN_Q446", "TRAIN_Q482", "TRAIN_Q140"},
    "C": {"TRAIN_Q069", "TRAIN_Q014", "TRAIN_Q188", "TRAIN_Q162", "TRAIN_Q089", "TRAIN_Q402", "TRAIN_Q294", "TRAIN_Q018", "TRAIN_Q154", "TRAIN_Q350", "TRAIN_Q015", "TRAIN_Q386", "TRAIN_Q446", "TRAIN_Q482", "TRAIN_Q140"},
}
OVERRIDE_FAIL_REASON = {
    ("TRAIN_Q014", "B"): "The depot/prerequisite workaround is relevant, but the answer adds an unsupported Agent Builder/manual-install operation as part of the proposed solution.",
    ("TRAIN_Q054", "C"): "Explains low-memory exhaustion but shifts to a 31-bit zSeries scenario and unrelated WebContainer advice instead of the question's compressed-references condition and applicable remedy.",
    ("TRAIN_Q494", "C"): "Mentions oracle.jdbc.autoCommitSpecCompliant=false but incorrectly casts it as possibly inapplicable and does not recommend the evidence-supported fix.",
}


def load(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> int:
    ids = json.loads(SAMPLE.read_text(encoding="utf-8"))["question_ids"]
    original_review = {(r["question_id"], r["chain"]): r for r in load(ORIGINAL / "final_review/final_reviews.jsonl")}
    diagnosis = {(r["question_id"], r["chain"]): r for r in load(ORIGINAL / "final_review/sample30_evidence_generation_diagnosis.jsonl")}
    scope_review = {(r["question_id"], r["chain"]): r for r in load(SCOPE / "codex_review/reviews.jsonl")}
    regenerated = {
        c: {r["question_id"]: r for r in load(REGENERATED / d / "generations.jsonl")}
        for c, d in CHAINS.items()
    }
    if any(set(regenerated[c]) != set(ids) for c in CHAINS):
        raise ValueError("regenerated A/B/C do not cover the saved sample30")

    aligned = 0
    rows = []
    summary = {"reviewer": "Codex", "groups": {}, "vs_scope_checked": {}, "complete_evidence_old_failures": {}}
    for chain, directory in CHAINS.items():
        original_generation = {r["question_id"]: r for r in load(ORIGINAL / directory / "generations.jsonl")}
        old_pass = {q for q in ids if original_review[q, chain]["final_verdict"] == "complete_pass"}
        scope_pass = {q for q in ids if scope_review[q, chain]["new_verdict"] == "complete_pass"}
        new_pass = PASS[chain]
        for q in ids:
            old_sources = [(s["document_id"], s["representative_chunk_id"], s["included_chunk_ids"]) for s in original_generation[q]["sources"]]
            new_sources = [(s["document_id"], s["representative_chunk_id"], s["included_chunk_ids"]) for s in regenerated[chain][q]["sources"]]
            if old_sources != new_sources:
                raise ValueError(f"source mismatch: {chain} {q}")
            aligned += 1
            passed = q in new_pass
            rows.append({
                "question_id": q,
                "chain": chain,
                "original_prompt_v2_verdict": original_review[q, chain]["final_verdict"],
                "regenerated_prompt_v2_verdict": "complete_pass" if passed else "fail",
                "scope_checked_verdict": scope_review[q, chain]["new_verdict"],
                "evidence_label": diagnosis[q, chain]["evidence_label"],
                "reason": PASS_REASON[q] if passed else (
                    OVERRIDE_FAIL_REASON[(q, chain)]
                    if (q, chain) in OVERRIDE_FAIL_REASON
                    else FAIL_REASON[q]
                ),
                "citation_map": regenerated[chain][q].get("citation_map", []),
                "generation_file": str(REGENERATED / directory / "generations.jsonl"),
                "context_file": str(REGENERATED / directory / "contexts.jsonl"),
            })
        added, lost = sorted(new_pass - old_pass), sorted(old_pass - new_pass)
        summary["groups"][chain] = {
            "original_prompt_v2_pass": len(old_pass),
            "regenerated_prompt_v2_pass": len(new_pass),
            "new_pass": len(added), "new_pass_question_ids": added,
            "regressed": len(lost), "regressed_question_ids": lost,
            "net": len(added) - len(lost),
        }
        summary["vs_scope_checked"][chain] = {
            "scope_checked_pass": len(scope_pass),
            "regenerated_prompt_v2_pass": len(new_pass),
            "prompt_v2_only_pass": sorted(new_pass - scope_pass),
            "scope_checked_only_pass": sorted(scope_pass - new_pass),
            "net": len(new_pass) - len(scope_pass),
        }
        eligible = {q for q in ids if diagnosis[q, chain].get("failure_diagnosis") == "prompt_present_but_answer_unused_or_misused"}
        rescued = sorted(eligible & new_pass)
        summary["complete_evidence_old_failures"][chain] = {"eligible": len(eligible), "rescued": len(rescued), "question_ids": rescued}
    summary["input_alignment"] = {
        "matched_question_chain_pairs": aligned,
        "same_top5_document_chunk_order": aligned == 90,
        "prompt_template": "configs/generation/answer_with_citations_direct_conditions.prompt",
        "model": regenerated["A"][ids[0]]["model"],
        "temperature": regenerated["A"][ids[0]]["temperature"],
        "max_tokens": regenerated["A"][ids[0]]["max_tokens"],
    }
    summary["complete_evidence_old_failures"]["total"] = {
        "eligible": sum(summary["complete_evidence_old_failures"][c]["eligible"] for c in CHAINS),
        "rescued": sum(summary["complete_evidence_old_failures"][c]["rescued"] for c in CHAINS),
    }
    out = REGENERATED / "codex_review"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reviews.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
