#!/usr/bin/env python3
"""Persist the Codex paired review for the 30-question C-chain context ablation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_REVIEW = ROOT / "experiments/generation/results/siliconflow_sample30_three_chain_top5_prompt_v2_regeneration/codex_review/reviews.jsonl"
NEW_GENERATIONS = ROOT / "experiments/generation/results/siliconflow_sample30_c_prompt_v2_context_ablation/representative_with_neighbors/generations.jsonl"
OUT_DIR = ROOT / "experiments/generation/results/siliconflow_sample30_c_prompt_v2_context_ablation/codex_review"


NEW_REVIEW = {
    "TRAIN_Q441": (False, "Uses an IHS rewrite procedure but omits/contradicts the decisive fileServingEnabled configuration in the reference answer."),
    "TRAIN_Q069": (True, "Correctly explains the firmware-6 TLS 1.1/1.2 default and legacy-server negotiation failure, with an applicable remedy."),
    "TRAIN_Q014": (True, "Correctly identifies the packaged prerequisite/architecture issue and gives the supported depot workaround."),
    "TRAIN_Q513": (False, "Points to fixes but omits the XML4J/XML4C crafted-content denial-of-service mechanism that is the core answer."),
    "TRAIN_Q188": (True, "Gives both JazzSM methods: page/widgets/wires information and information for an individual widget."),
    "TRAIN_Q162": (True, "Identifies the HttpCore/HttpClient visibility change and tells the application to bundle the required libraries."),
    "TRAIN_Q144": (True, "Now states the directory-listing disclosure, affected WMB V8 scope, base score, and applicable fix from source [1]."),
    "TRAIN_Q089": (True, "States WESB applicability and the requested CVE-2015-2613 JCE information-disclosure mechanism from source [1]."),
    "TRAIN_Q511": (False, "Gives fix levels but omits the PKCS#7 missing-ContentInfo/NULL-dereference denial-of-service mechanism."),
    "TRAIN_Q064": (False, "Incorrectly says RHEL 5.8 is supported although the supplied minimum is RHEL 6."),
    "TRAIN_Q462": (True, "Now states the authenticated dynamic-queue shared-memory leak, resource impact, affected versions, and fix location."),
    "TRAIN_Q375": (True, "Now states the glibc gethostbyname buffer overflow/GHOST impact, affected levels, and fixed levels."),
    "TRAIN_Q054": (False, "Discusses generic native-memory remedies but does not preserve the question's compressed-references condition and corresponding remedy."),
    "TRAIN_Q402": (True, "Identifies the unsupported Oracle/JDBC driver mismatch and recommends the supported driver/configuration actions."),
    "TRAIN_Q294": (True, "Correctly recommends a compatible CICS Explorer and preserves the 5.2.0.7-or-later condition."),
    "TRAIN_Q018": (True, "Explains the Managed Pages deletion cost and retains both server and WCM timeout conditions and remedies."),
    "TRAIN_Q017": (False, "Concludes Oracle 11g is latest but cites WAS/ISAM evidence that does not establish Portal 6.1.0.6 database support."),
    "TRAIN_Q058": (False, "Proposes unrelated installation causes and omits the parsing-code fix targeted for 8.5.5.6."),
    "TRAIN_Q139": (False, "Gives fix pointers but omits the MD5 ServerKeyExchange collision/MITM mechanism that is the core bulletin detail."),
    "TRAIN_Q154": (False, "Answers with a Gatekeeper right-click workaround instead of the required Apple Java SE 6 prerequisite before Statistics 22."),
    "TRAIN_Q350": (True, "Directly gives the required action: lock one attribute to expose the remaining component attributes."),
    "TRAIN_Q415": (True, "Now identifies the unsupported 256-bit/maximum-strength key condition and gives both supported remedies."),
    "TRAIN_Q015": (True, "Keeps the requested IIB V9/WMB V8 Hypervisor Edition Red Hat security scope and points to the matching action information."),
    "TRAIN_Q386": (True, "Gives the requested CVE-2014-6593 JSSE impact and severity information from the matching bulletin."),
    "TRAIN_Q127": (True, "Now explains that the error is normally a connection failure, lists the two principal causes, and recommends trace/debug evidence."),
    "TRAIN_Q494": (False, "Attributes the failure to an unrelated WAS 6.1 defect and omits oracle.jdbc.autoCommitSpecCompliant=false."),
    "TRAIN_Q446": (True, "Directly states that DataPower is unaffected because it does not use Bash."),
    "TRAIN_Q482": (True, "Correctly states no fixed IN-list limit and retains the virtual-table and compilation-memory conditions."),
    "TRAIN_Q288": (False, "Invents a missing resource-file diagnosis and omits libstdc++.so.5/libgcc_s.so.1 and compat-libstdc++-33."),
    "TRAIN_Q140": (True, "Reproduces the complete reinstall procedure, including local media and all .NET 4.5 subcomponents."),
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    old_rows = [row for row in read_jsonl(OLD_REVIEW) if row["chain"] == "C"]
    old = {row["question_id"]: row for row in old_rows}
    generations = read_jsonl(NEW_GENERATIONS)
    qids = [row["question_id"] for row in generations]
    if set(qids) != set(old) or set(qids) != set(NEW_REVIEW):
        raise RuntimeError("Question IDs do not align across old review, new generations, and paired review")

    output = []
    for row in generations:
        qid = row["question_id"]
        old_pass = old[qid]["regenerated_prompt_v2_verdict"] == "complete_pass"
        new_pass, reason = NEW_REVIEW[qid]
        transition = (
            "new_pass" if new_pass and not old_pass else
            "regression" if old_pass and not new_pass else
            "both_pass" if old_pass and new_pass else
            "both_fail"
        )
        output.append({
            "question_id": qid,
            "old_representative_only": "complete_pass" if old_pass else "fail",
            "new_representative_with_neighbors": "complete_pass" if new_pass else "fail",
            "transition": transition,
            "reason": reason,
            "new_generation_file": str(NEW_GENERATIONS),
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "reviews.jsonl").open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    old_pass_count = sum(row["old_representative_only"] == "complete_pass" for row in output)
    new_pass_count = sum(row["new_representative_with_neighbors"] == "complete_pass" for row in output)
    summary = {
        "reviewer": "Codex",
        "question_count": len(output),
        "old_representative_only_complete_pass": old_pass_count,
        "new_representative_with_neighbors_complete_pass": new_pass_count,
        "new_pass": [row["question_id"] for row in output if row["transition"] == "new_pass"],
        "regression": [row["question_id"] for row in output if row["transition"] == "regression"],
        "net_change": new_pass_count - old_pass_count,
        "reviews_file": str(OUT_DIR / "reviews.jsonl"),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
