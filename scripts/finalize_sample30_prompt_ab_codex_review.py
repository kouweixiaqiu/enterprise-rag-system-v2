#!/usr/bin/env python3
"""Write the completed Codex review for the sample30 old/new Prompt comparison."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "experiments/generation/results/siliconflow_train450_three_chain_top5"
NEW = ROOT / "experiments/generation/results/siliconflow_sample30_three_chain_top5_scope_checked_prompt"
SAMPLE = ROOT / "experiments/dense/results/qwen3_reranker_4b_bf16_top50_seven_chain_sample30_seed42/sample_question_ids.json"
CHAINS = {
    "A": "A_fixed_1200_0_rrf_original",
    "B": "B_fixed_1200_0_rrf_qwen3_4b_bf16_reranked",
    "C": "C_paragraph_merge_1200_0_rrf_original",
}

PASS = {
    "A": {"TRAIN_Q069", "TRAIN_Q014", "TRAIN_Q188", "TRAIN_Q162", "TRAIN_Q294", "TRAIN_Q350", "TRAIN_Q015", "TRAIN_Q446", "TRAIN_Q482", "TRAIN_Q140"},
    "B": {"TRAIN_Q069", "TRAIN_Q014", "TRAIN_Q188", "TRAIN_Q162", "TRAIN_Q064", "TRAIN_Q294", "TRAIN_Q154", "TRAIN_Q350", "TRAIN_Q015", "TRAIN_Q127", "TRAIN_Q494", "TRAIN_Q446", "TRAIN_Q140"},
    "C": {"TRAIN_Q069", "TRAIN_Q014", "TRAIN_Q089", "TRAIN_Q162", "TRAIN_Q054", "TRAIN_Q402", "TRAIN_Q294", "TRAIN_Q018", "TRAIN_Q350", "TRAIN_Q015", "TRAIN_Q386", "TRAIN_Q446", "TRAIN_Q140"},
}

PASS_REASON = {
    "TRAIN_Q069": "Correctly explains the firmware-6 TLS 1.1/1.2 change, legacy-server negotiation failure, and applicable remedies.",
    "TRAIN_Q014": "Provides the depot/prerequisite handling needed to install the NMA bundle and keeps the procedure within the supplied product context.",
    "TRAIN_Q188": "Gives both requested JazzSM methods: page/widgets/wires information and information for an individual widget.",
    "TRAIN_Q162": "Identifies the HttpCore/HttpClient visibility side effect and tells the application to bundle the required libraries.",
    "TRAIN_Q089": "States WESB applicability and gives the requested CVE-2015-2613 mechanism, score, and vector from the supplied evidence.",
    "TRAIN_Q064": "Directly concludes that RHEL 5.x is unsupported and preserves the RHEL 6 minimum condition.",
    "TRAIN_Q054": "Explains exhaustion of the low 4-GB native-memory region and gives an evidence-supported mitigation.",
    "TRAIN_Q402": "Identifies the unsupported Oracle/JDBC driver match as the cause and recommends using the supported driver required by the system requirements.",
    "TRAIN_Q294": "Correctly recommends a compatible/newer CICS Explorer and preserves the applicable 5.2.0.7-or-later condition.",
    "TRAIN_Q018": "Explains the Managed Pages deletion cost and retains both the server and WCM transaction-timeout conditions and remedies.",
    "TRAIN_Q154": "Gives the required prerequisite in the correct order: install Apple Java SE 6 before installing Statistics 22.",
    "TRAIN_Q350": "Directly gives the required action: lock one attribute, such as name, to expose the remaining component attributes.",
    "TRAIN_Q015": "Points to the matching IIB V9/WMB V8 Hypervisor Edition Red Hat security document without crossing product scope.",
    "TRAIN_Q386": "Gives the requested CVE-2014-6593 JSSE impact, base score, and correct CVSS vector.",
    "TRAIN_Q127": "Identifies slow/closed persistent connections and retains packet-trace/debug-log diagnosis.",
    "TRAIN_Q494": "Gives the required Oracle 12c JVM property oracle.jdbc.autoCommitSpecCompliant=false for the Monitor server.",
    "TRAIN_Q446": "Directly states that DataPower is unaffected because it does not use Bash.",
    "TRAIN_Q482": "Correctly states there is no fixed IN-list limit and retains the practical virtual-table and compilation-memory conditions.",
    "TRAIN_Q140": "Reproduces the complete reinstall procedure, including local media and all .NET 4.5 subcomponents.",
}

FAIL_REASON = {
    "TRAIN_Q441": "Does not preserve the required fileServingEnabled=true prerequisite and complete RewriteRule chain; some variants substitute unrelated IHS configuration.",
    "TRAIN_Q513": "The actual Prompt lacks the XML4J/XML4C mechanism, 100% CPU impact, and CVSS details, so the answer cannot supply the core bulletin details.",
    "TRAIN_Q188": "Although both methods are stated, the answer then incorrectly says widget release information is unavailable despite complete evidence.",
    "TRAIN_Q144": "Omits the requested directory-listing vulnerability impact and its CVSS score/vector, replacing the core details with a document or fix pointer.",
    "TRAIN_Q089": "Does not accurately give the requested CVE-2015-2613 mechanism and scoring; one variant reports the wrong base score.",
    "TRAIN_Q511": "The answer either lacks the CVE mechanism or imports it from a different product without evidence connecting that product to DataPower; key claims also lack numbered citations.",
    "TRAIN_Q064": "Does not preserve the decisive RHEL 6 minimum; one variant incorrectly concludes the RHEL 5 configuration can be supported.",
    "TRAIN_Q462": "Omits the authenticated dynamic-queue shared-memory-leak mechanism and/or the complete CVSS details; a variant also gives an unsupported fix level.",
    "TRAIN_Q375": "Does not provide the GHOST/gethostbyname buffer-overflow mechanism and complete score/vector required by the question.",
    "TRAIN_Q054": "Uses unrelated native-memory remedies or gives incorrect heap-placement advice instead of the compressed-references low-4-GB cause and applicable mitigation.",
    "TRAIN_Q402": "Substitutes speculative Oracle timeout/listener changes for the evidence-supported JDBC-driver compatibility cause and fix.",
    "TRAIN_Q018": "Mentions only the server transaction timeout and omits the separate WCM transaction-timeout condition required for a complete answer.",
    "TRAIN_Q017": "The supplied chunks do not establish Portal 6.1.0.6 support for Oracle 11gR2 11.2.0.1, and the answer asserts other versions from unrelated products.",
    "TRAIN_Q058": "The Prompt lacks the parsing-code APAR and 8.5.5.6 fix-pack condition; the answer proposes unrelated installation causes and fixes.",
    "TRAIN_Q139": "Only points to the bulletin and omits the MD5 ServerKeyExchange collision/MITM mechanism and CVSS details.",
    "TRAIN_Q154": "Answers with the Subscription/Gatekeeper workaround instead of the Statistics 22 prerequisite to install Apple Java SE 6 first.",
    "TRAIN_Q415": "Does not state that Maximum-strength keys (>128 bit) were selected during Workplace XT installation; variants invent unsupported Java-policy edits.",
    "TRAIN_Q386": "The answer is incomplete or gives AC:L instead of the evidence-supported AC:H CVSS vector.",
    "TRAIN_Q127": "Attributes the error to an incorrect timeout rule and misses the evidence-supported connection-failure diagnosis and debug/trace confirmation.",
    "TRAIN_Q494": "Misses oracle.jdbc.autoCommitSpecCompliant=false and either recommends an unsupported driver upgrade or incorrectly says the direct fix is absent.",
    "TRAIN_Q482": "Mixes an Oracle-specific 1,000-item limit into a different database context, contradicting the applicable no-fixed-limit evidence.",
    "TRAIN_Q288": "The Prompt lacks the required libstdc++.so.5/libgcc_s.so.1 and compat-libstdc++-33 dependency evidence; the proposed resource-file fixes are unsupported.",
}

UNSUPPORTED_NEW = {
    ("TRAIN_Q402", "A"),
    ("TRAIN_Q054", "B"),
    ("TRAIN_Q402", "B"),
    ("TRAIN_Q482", "C"),
}
FALSE_INSUFFICIENCY = {("TRAIN_Q188", "C"), ("TRAIN_Q494", "C")}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ids = json.loads(SAMPLE.read_text(encoding="utf-8"))["question_ids"]
    old = {(r["question_id"], r["chain"]): r for r in load(OLD / "final_review/final_reviews.jsonl")}
    diagnosis = {(r["question_id"], r["chain"]): r for r in load(OLD / "final_review/sample30_evidence_generation_diagnosis.jsonl")}
    generated = {
        chain: {r["question_id"]: r for r in load(NEW / directory / "generations.jsonl")}
        for chain, directory in CHAINS.items()
    }
    if any(set(generated[c]) != set(ids) for c in CHAINS):
        raise ValueError("new A/B/C outputs do not match the saved sample30 IDs")

    aligned = 0
    for chain, directory in CHAINS.items():
        old_generated = {
            r["question_id"]: r for r in load(OLD / directory / "generations.jsonl")
        }
        for q in ids:
            old_sources = [
                (s["document_id"], s["representative_chunk_id"], s["included_chunk_ids"])
                for s in old_generated[q]["sources"]
            ]
            new_sources = [
                (s["document_id"], s["representative_chunk_id"], s["included_chunk_ids"])
                for s in generated[chain][q]["sources"]
            ]
            if old_sources != new_sources:
                raise ValueError(f"old/new source mismatch: {chain} {q}")
            aligned += 1

    review_rows = []
    summary = {
        "reviewer": "Codex",
        "input_alignment": {
            "matched_old_new_question_chain_pairs": aligned,
            "expected": 90,
            "same_top5_document_chunk_order": aligned == 90,
            "model": generated["A"][ids[0]]["model"],
            "temperature": generated["A"][ids[0]]["temperature"],
            "max_tokens": generated["A"][ids[0]]["max_tokens"],
        },
        "groups": {},
        "complete_evidence_old_failures": {},
    }
    for chain, directory in CHAINS.items():
        old_pass = {q for q in ids if old[q, chain]["final_verdict"] == "complete_pass"}
        new_pass = PASS[chain]
        for q in ids:
            row = generated[chain][q]
            passed = q in new_pass
            review_rows.append({
                "question_id": q,
                "chain": chain,
                "old_verdict": old[q, chain]["final_verdict"],
                "new_verdict": "complete_pass" if passed else "fail",
                "changed": (old[q, chain]["final_verdict"] == "complete_pass") != passed,
                "evidence_label": diagnosis[q, chain]["evidence_label"],
                "reason": PASS_REASON[q] if passed else FAIL_REASON[q],
                "citation_map": row.get("citation_map", []),
                "new_unsupported_conclusion": (q, chain) in UNSUPPORTED_NEW,
                "false_insufficient_context_with_complete_evidence": (q, chain) in FALSE_INSUFFICIENCY,
                "generation_file": str(NEW / directory / "generations.jsonl"),
                "context_file": str(NEW / directory / "contexts.jsonl"),
            })
        added = sorted(new_pass - old_pass)
        lost = sorted(old_pass - new_pass)
        summary["groups"][chain] = {
            "old_complete_pass": len(old_pass),
            "new_complete_pass": len(new_pass),
            "new_pass": len(added),
            "new_pass_question_ids": added,
            "regressed": len(lost),
            "regressed_question_ids": lost,
            "net": len(added) - len(lost),
        }
        eligible = {
            q for q in ids
            if diagnosis[q, chain].get("failure_diagnosis") == "prompt_present_but_answer_unused_or_misused"
        }
        rescued = sorted(eligible & new_pass)
        summary["complete_evidence_old_failures"][chain] = {
            "eligible": len(eligible), "rescued": len(rescued), "question_ids": rescued
        }
    summary["complete_evidence_old_failures"]["total"] = {
        "eligible": sum(v["eligible"] for k, v in summary["complete_evidence_old_failures"].items() if k != "total"),
        "rescued": sum(v["rescued"] for k, v in summary["complete_evidence_old_failures"].items() if k != "total"),
    }
    summary["quality_regressions"] = {
        "new_unsupported_conclusion_count": len(UNSUPPORTED_NEW),
        "new_unsupported_conclusion_pairs": [list(x) for x in sorted(UNSUPPORTED_NEW)],
        "false_insufficient_context_with_complete_evidence_count": len(FALSE_INSUFFICIENCY),
        "false_insufficient_context_pairs": [list(x) for x in sorted(FALSE_INSUFFICIENCY)],
    }

    out = NEW / "codex_review"
    out.mkdir(parents=True, exist_ok=True)
    (out / "reviews.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in review_rows), encoding="utf-8"
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
