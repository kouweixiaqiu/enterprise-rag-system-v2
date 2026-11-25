#!/usr/bin/env python3
"""Finalize the 600/0 Dense vs paragraph-1200 RRF generation review."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/generation/results/train450_fixed600_dense_vs_paragraph1200_rrf_prompt_v2_review"
INITIAL = BASE / "initial_reviews.jsonl"
OUTPUT = BASE / "final_review"
LEFT = "fixed_600_0_dense"
RIGHT = "paragraph_merge_1200_0_rrf"
LABELS = (LEFT, RIGHT)


# Corrections made after checking the question/reference, both answers, and—when
# support was in doubt—the actual evidence supplied to generation.
MANUAL = {
    ("TRAIN_Q016", LEFT): (False, "回答给出提高 nofile 的通用操作，但没有回答参考答案要求的 ulimit -Xa 语义及运行中进程可能继承不同限制。"),
    ("TRAIN_Q020", RIGHT): (True, "回答给出 HATS 作为 RAD 插件的获取方式、Passport Advantage 步骤和所需 eAssembly，满足安装问题的核心要求。"),
    ("TRAIN_Q050", LEFT): (True, "回答正确说明 JazzSM 默认要求 HTTPS，HTTP 即使配置也不受支持；末尾的保守说明不改变核心结论。"),
    ("TRAIN_Q050", RIGHT): (False, "结论虽称 HTTP 不受支持，却错误地把 1.1.2.1 说成早于 1.1.0.3，并通过错误版本推断得出结论。"),
    ("TRAIN_Q080", RIGHT): (True, "回答保留了适用产品、RHEL 6.2 条件和联系 Red Hat 获取并安装修复的核心动作；列出题目涉及的 CVE 不构成缺失。"),
    ("TRAIN_Q117", RIGHT): (True, "回答明确说明相关 MobileFirst/Worklight 产品不正式支持 Oracle JDK 8，并正确区分局部临时修复与产品支持状态。"),
    ("TRAIN_Q139", RIGHT): (False, "回答主要给出修复位置，却没有回答参考答案的 MD5 ServerKeyExchange 碰撞、MITM 影响等核心漏洞机制。"),
    ("TRAIN_Q144", RIGHT): (True, "回答完整给出目录列表信息泄露、WMB V8 范围、基础分和 8.0.0.8/IT16698 修复，且由来源[1]支持。"),
    ("TRAIN_Q193", RIGHT): (True, "10月30日与参考答案的10月底上线条件一致，回答还保留了入口、PMR转Case及支持渠道，没有实质错误。"),
    ("TRAIN_Q252", RIGHT): (False, "回答只把异步解析当作确定原因，没有保留参考答案列出的错误 ruleset path、通知失败和解析失败回退等其他必要可能性。"),
    ("TRAIN_Q296", LEFT): (True, "回答完整复现移除 RC4 的四步；额外提到其他密码套件不影响该问题的正确操作。"),
    ("TRAIN_Q296", RIGHT): (True, "回答完整复现移除 RC4 的四步；关于无法外部化列表的附加说明不否定已给出的正确答案。"),
    ("TRAIN_Q305", RIGHT): (True, "核心要求是联系 Student/GradPack 供应商；回答明确要求联系供应商取得正确授权码，满足参考答案。"),
    ("TRAIN_Q350", RIGHT): (True, "回答直接给出锁定任一属性以显示其他属性的完整操作；未重复版本范围不影响该具体操作答案。"),
    ("TRAIN_Q390", LEFT): (True, "回答已经给出属性文件位置、容量取值、所有集群节点、重启、正确主节点和禁止尾随空格等完整条件；自称缺少信息不构成实质遗漏。"),
    ("TRAIN_Q414", LEFT): (True, "回答正确说明 MQ 7.0.0/7.0.1 支持终止，并进一步区分捆绑组件和资源适配器的支持期限。"),
    ("TRAIN_Q426", RIGHT): (True, "初审理由实际复述了回答中的正确受影响版本和修复级别；回答还给出正确 CVSS 分数和向量，应判通过。"),
    ("TRAIN_Q482", RIGHT): (True, "回答明确说明 IN 列表没有硬上限、默认阈值为5且复杂查询受内存约束，足以回答具体问题。"),
    ("TRAIN_Q526", RIGHT): (False, "问题询问 Rhapsody 8.1.2 x64 中缺少 ReporterPlus，回答却转向 Q language 单匹配限制和活动图模板，未回答兼容性原因。"),
}


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    rows = load(INITIAL)
    by_key = {(row["question_id"], row["chain"]): row for row in rows}
    if len(rows) != 900 or len(by_key) != 900:
        raise ValueError("expected 900 unique initial review rows")
    question_ids = sorted({question_id for question_id, _ in by_key})
    if len(question_ids) != 450:
        raise ValueError("expected 450 aligned questions")
    if {chain for _, chain in by_key} != set(LABELS):
        raise ValueError("unexpected chain labels")
    if any((question_id, chain) not in by_key for question_id in question_ids for chain in LABELS):
        raise ValueError("the two initial-review question sets do not align")

    differences = {
        question_id
        for question_id in question_ids
        if by_key[question_id, LEFT]["verdict"]
        != by_key[question_id, RIGHT]["verdict"]
    }
    both_pass = [
        q for q in question_ids
        if all(by_key[q, chain]["verdict"] == "complete_pass" for chain in LABELS)
    ]
    both_fail = [
        q for q in question_ids
        if all(by_key[q, chain]["verdict"] == "fail" for chain in LABELS)
    ]
    rng = random.Random(42)
    sampled_pass = set(rng.sample(both_pass, 10))
    sampled_fail = set(rng.sample(both_fail, 10))
    sampled = sampled_pass | sampled_fail
    reviewed = differences | sampled

    final_rows = []
    for question_id in question_ids:
        for chain in LABELS:
            initial = by_key[question_id, chain]
            initial_pass = initial["verdict"] == "complete_pass"
            final_pass = initial_pass
            if question_id in reviewed:
                review_reason = "Codex复核问题、参考答案和初审具体理由后，维持初审判定。"
            else:
                review_reason = "未进入Codex复核集合，沿用匿名API初审。"
            if (question_id, chain) in MANUAL:
                final_pass, reason = MANUAL[question_id, chain]
                review_reason = "Codex展开回答与实际证据后改判：" + reason
            final_rows.append(
                {
                    "question_id": question_id,
                    "chain": chain,
                    "initial_verdict": initial["verdict"],
                    "initial_reason": initial["reason"],
                    "initial_evidence_positions": initial["evidence_positions"],
                    "citation_assessment": initial["citation_assessment"],
                    "citation_reason": initial["citation_reason"],
                    "codex_reviewed": question_id in reviewed,
                    "review_set": (
                        "pairwise_difference" if question_id in differences else
                        "consistent_random_sample" if question_id in sampled else None
                    ),
                    "final_verdict": "complete_pass" if final_pass else "fail",
                    "review_changed": final_pass != initial_pass,
                    "review_reason": review_reason,
                }
            )

    final = {(row["question_id"], row["chain"]): row for row in final_rows}
    summary = {
        "review_method": {
            "initial": "Qwen/Qwen2.5-7B-Instruct via SiliconFlow; anonymous per-answer review of all 900 answers",
            "codex_reviewed_difference_questions": len(differences),
            "codex_reviewed_consistent_sample": len(sampled),
            "consistent_sample_seed": 42,
            "codex_review_scope": "all pairwise-difference questions plus 10 both-pass and 10 both-fail questions; ambiguous cases expanded to answers and supplied evidence",
            "review_changed_answer_count": sum(row["review_changed"] for row in final_rows),
        },
        "groups": {},
        "comparison": {},
    }
    for chain in LABELS:
        passed = sum(
            final[q, chain]["final_verdict"] == "complete_pass"
            for q in question_ids
        )
        citations = Counter(final[q, chain]["citation_assessment"] for q in question_ids)
        summary["groups"][chain] = {
            "complete_pass": passed,
            "fail": 450 - passed,
            "total": 450,
            "rate": passed / 450,
            "citation_assessment": dict(citations),
        }

    right_only = [
        q for q in question_ids
        if final[q, RIGHT]["final_verdict"] == "complete_pass"
        and final[q, LEFT]["final_verdict"] == "fail"
    ]
    left_only = [
        q for q in question_ids
        if final[q, LEFT]["final_verdict"] == "complete_pass"
        and final[q, RIGHT]["final_verdict"] == "fail"
    ]
    both_final_pass = [
        q for q in question_ids
        if all(final[q, chain]["final_verdict"] == "complete_pass" for chain in LABELS)
    ]
    both_final_fail = [
        q for q in question_ids
        if all(final[q, chain]["final_verdict"] == "fail" for chain in LABELS)
    ]
    summary["comparison"] = {
        "paragraph1200_only_pass": len(right_only),
        "fixed600_only_pass": len(left_only),
        "net_paragraph1200_over_fixed600": len(right_only) - len(left_only),
        "both_pass": len(both_final_pass),
        "both_fail": len(both_final_fail),
        "paragraph1200_only_pass_question_ids": right_only,
        "fixed600_only_pass_question_ids": left_only,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "final_reviews.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final_rows),
        encoding="utf-8",
    )
    (OUTPUT / "codex_review_questions.json").write_text(
        json.dumps(
            {
                "difference_question_ids": sorted(differences),
                "consistent_random_sample_question_ids": sorted(sampled),
                "consistent_random_sample_both_pass": sorted(sampled_pass),
                "consistent_random_sample_both_fail": sorted(sampled_fail),
                "manual_override_keys": [
                    {"question_id": q, "chain": chain}
                    for q, chain in sorted(MANUAL)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
