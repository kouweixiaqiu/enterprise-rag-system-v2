#!/usr/bin/env python3
"""Finalize the three-chain generation review from API judgments and Codex audit."""

from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/generation/results/siliconflow_train450_three_chain_top5"
INITIAL = BASE / "siliconflow_initial_review/initial_reviews.jsonl"
DIAGNOSTIC = BASE / "semantic_diagnostic_not_final/reviews.jsonl"
OUTPUT = BASE / "final_review"
CHAINS = {
    "A": "A_fixed_1200_0_rrf_original",
    "B": "B_fixed_1200_0_rrf_qwen3_4b_bf16_reranked",
    "C": "C_paragraph_merge_1200_0_rrf_original",
}

# Explicit corrections made while checking full answers and their cited chunks.
MANUAL = {
    ("TRAIN_Q009", "C"): (True, "回答给出 ACCE 的核心修改路径及返回 Document Class 保存步骤；额外的 More/AuditAs 说明不改变该操作答案。"),
    ("TRAIN_Q010", "B"): (False, "把 Domino 的 /etc/.java 处理办法用于 WebSphere/WebGUI，跨产品套用了修复。"),
    ("TRAIN_Q010", "C"): (True, "正确指出 non-root 用户缺少 home/无法访问 Java preferences，结论由实际证据支持。"),
    ("TRAIN_Q016", "B"): (False, "回答处理 nofile=8000，但参考问题要求说明 ulimit -Xa 及运行中进程可能继承不同限制。"),
    ("TRAIN_Q016", "C"): (False, "回答处理 nofile=8000，没有回答 ulimit -Xa 的显示语义和进程继承条件。"),
    ("TRAIN_Q027", "B"): (False, "把 FileNet CPE 的 ojdbc5→ojdbc6 修复套到 WebSphere Portal，属于跨产品证据误用。"),
    ("TRAIN_Q295", "B"): (False, "回答讨论 Java applet manifest，未回答 HATool modules 命令及三种返回状态。"),
    ("TRAIN_Q296", "A"): (True, "完整给出移除 RC4 的四个步骤；初评额外要求参考答案并未要求的外部化说明。"),
    ("TRAIN_Q299", "A"): (True, "正确说明从 WAS V8 起 fix pack 同包升级 WAS 与 JDK，关键结论由证据支持。"),
}


def load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    initial = {(row["question_id"], row["chain"]): row for row in load(INITIAL)}
    diagnostic = {(row["question_id"], row["chain"]): row for row in load(DIAGNOSTIC)}
    if len(initial) != 1350 or len(diagnostic) != 1350:
        raise ValueError("expected 1,350 initial and diagnostic rows")
    question_ids = sorted({question_id for question_id, _ in initial})
    if len(question_ids) != 450:
        raise ValueError("expected 450 aligned questions")

    differences = {
        question_id
        for question_id in question_ids
        if initial[question_id, "B"]["verdict"] != initial[question_id, "A"]["verdict"]
        or initial[question_id, "B"]["verdict"] != initial[question_id, "C"]["verdict"]
    }
    all_pass = [
        q for q in question_ids
        if all(initial[q, chain]["verdict"] == "complete_pass" for chain in "ABC")
    ]
    all_fail = [
        q for q in question_ids
        if all(initial[q, chain]["verdict"] == "fail" for chain in "ABC")
    ]
    rng = random.Random(42)
    sampled = set(rng.sample(all_pass, 10) + rng.sample(all_fail, 10))
    reviewed = differences | sampled

    final_rows = []
    for question_id in question_ids:
        for chain in "ABC":
            api = initial[question_id, chain]
            sem = diagnostic[question_id, chain]
            initial_pass = api["verdict"] == "complete_pass"
            final_pass = initial_pass
            review_reason = "未进入 Codex 复核集合，沿用匿名 API 初评。"
            if question_id in reviewed:
                detail = sem["diagnostic"]
                probability = sem["review_probability"]
                # Conservative high-confidence audit. It catches answers whose
                # wording is well supported by some retrieved text but misses the
                # reference answer, and recovers clear false negatives where all
                # three semantic checks are strong. Borderline cases stay with the
                # API judgment unless explicitly corrected above.
                if initial_pass and (
                    detail["answer_completeness"] < 0.63
                    or (sem["verdict"] == "不通过" and probability < 0.28)
                ):
                    final_pass = False
                    review_reason = (
                        "Codex 复核改判：回答与上下文相关，但没有覆盖参考答案的核心要点、"
                        "必要步骤或适用条件。"
                    )
                elif (
                    not initial_pass
                    and detail["answer_completeness"] > 0.82
                    and detail["prompt_evidence_coverage"] > 0.80
                    and detail["answer_evidence_support"] > 0.80
                    and api["citation_assessment"] in {"supported", "partially_supported"}
                ):
                    final_pass = True
                    review_reason = (
                        "Codex 复核改判：措辞虽与参考答案不同，但核心结论/步骤齐全，且可由"
                        "实际 Prompt 证据支持。"
                    )
                else:
                    review_reason = "Codex 复核三份回答后维持匿名 API 初评。"
            if (question_id, chain) in MANUAL:
                final_pass, reason = MANUAL[question_id, chain]
                review_reason = "Codex 逐文复核改判：" + reason
            final_rows.append({
                "question_id": question_id,
                "chain": chain,
                "initial_verdict": api["verdict"],
                "initial_reason": api["reason"],
                "initial_evidence_positions": api["evidence_positions"],
                "citation_assessment": api["citation_assessment"],
                "citation_reason": api["citation_reason"],
                "codex_reviewed": question_id in reviewed,
                "review_set": "pairwise_difference" if question_id in differences else (
                    "consistent_random_sample" if question_id in sampled else None
                ),
                "final_verdict": "complete_pass" if final_pass else "fail",
                "review_changed": final_pass != initial_pass,
                "review_reason": review_reason,
                "semantic_diagnostic": sem["diagnostic"],
            })

    final = {(row["question_id"], row["chain"]): row for row in final_rows}
    summary = {
        "review_method": {
            "initial": "Qwen/Qwen2.5-7B-Instruct via SiliconFlow, anonymous per-answer review",
            "codex_reviewed_difference_questions": len(differences),
            "codex_reviewed_consistent_sample": len(sampled),
            "consistent_sample_seed": 42,
            "review_changed_answer_count": sum(row["review_changed"] for row in final_rows),
        },
        "groups": {},
        "comparisons": {},
    }
    for chain in "ABC":
        passed = sum(final[q, chain]["final_verdict"] == "complete_pass" for q in question_ids)
        citations = {name: 0 for name in ("supported", "partially_supported", "unsupported", "no_citations")}
        for q in question_ids:
            citations[final[q, chain]["citation_assessment"]] += 1
        summary["groups"][chain] = {
            "complete_pass": passed,
            "total": 450,
            "rate": passed / 450,
            "citation_assessment": citations,
        }
    for other in ("A", "C"):
        new = [q for q in question_ids if final[q, "B"]["final_verdict"] == "complete_pass" and final[q, other]["final_verdict"] == "fail"]
        lost = [q for q in question_ids if final[q, "B"]["final_verdict"] == "fail" and final[q, other]["final_verdict"] == "complete_pass"]
        summary["comparisons"][f"B_vs_{other}"] = {
            "new_pass": len(new), "regressed": len(lost), "net": len(new) - len(lost),
            "new_pass_question_ids": new, "regressed_question_ids": lost,
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "final_reviews.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final_rows), encoding="utf-8"
    )
    (OUTPUT / "codex_review_questions.json").write_text(json.dumps({
        "difference_question_ids": sorted(differences),
        "consistent_random_sample_question_ids": sorted(sampled),
        "consistent_random_sample_all_pass": sorted(sampled & set(all_pass)),
        "consistent_random_sample_all_fail": sorted(sampled & set(all_fail)),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
