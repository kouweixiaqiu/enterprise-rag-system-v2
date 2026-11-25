#!/usr/bin/env python3
"""Save the Codex review for the balanced refusal-prompt experiment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / (
    "experiments/generation/results/"
    "refusal_prompt_balanced_36_answerable_36_no_answer"
)

POSITIVE_PASSES = {
    "baseline": {
        "TRAIN_Q015", "TRAIN_Q140", "TRAIN_Q162", "TRAIN_Q188", "TRAIN_Q294",
        "TRAIN_Q446", "TRAIN_Q503", "TRAIN_Q172", "TRAIN_Q186", "TRAIN_Q340",
        "TRAIN_Q070",
    },
    "balanced": {
        "TRAIN_Q015", "TRAIN_Q069", "TRAIN_Q140", "TRAIN_Q162", "TRAIN_Q188",
        "TRAIN_Q294", "TRAIN_Q441", "TRAIN_Q446", "TRAIN_Q503", "TRAIN_Q186",
        "TRAIN_Q580", "TRAIN_Q340", "TRAIN_Q535", "TRAIN_Q070",
    },
}

FAIL_REASONS = {
    "TRAIN_Q018": "给出了超时调整，但没有回答 Managed Pages 导致删除操作超过服务器或 WCM 事务超时这一原因。",
    "TRAIN_Q064": "把最低支持版本列表外推成可能可运行，加入了证据没有支持的兼容性判断。",
    "TRAIN_Q144": "只给修复方向，遗漏漏洞机制、影响和问题要求的公告核心信息。",
    "TRAIN_Q402": "提供通用连接池排查并混入其他场景，没有准确解释该 Oracle OALL8 错误。",
    "TRAIN_Q415": "核心方向接近，但把 IBM JRE 的策略文件处理写成从 Oracle 下载，操作依据不匹配。",
    "TRAIN_Q441": "旧版遗漏 fileServingEnabled 前置条件和完整重写规则。",
    "TRAIN_Q462": "只声称公告包含答案，没有给出漏洞机制、影响、版本和修复内容。",
    "TRAIN_Q482": "混入无关的 Oracle 1000 项限制，且遗漏适用的 inlist_threshold 行为和安全上限。",
    "TRAIN_Q236": "先给不适用于 SPSS 19/Sierra 的授权步骤，尽管末尾承认该版本不受支持。",
    "TRAIN_Q426": "只指向公告，没有实际回答漏洞影响、适用版本和修复。",
    "TRAIN_Q394": "错误地把 db2iupdt 当作完整更新系统目录的解决步骤，与参考结论不符。",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    overall = {}
    for variant in ("baseline", "balanced"):
        generations = read_jsonl(RUN / variant / "generations.jsonl")
        reviews = []
        for row in generations:
            if row["class"] == "no_answer":
                passed = row["decision"] == "INSUFFICIENT_CONTEXT"
                reason = (
                    "Top-100 无核心答案证据，模型正确拒答。"
                    if passed
                    else "Top-100 无核心答案证据，但模型仍生成了操作性答案。"
                )
            elif row["decision"] != "ANSWER":
                passed = False
                reason = "实际上下文包含完整标注答案，但模型错误拒答。"
            else:
                passed = row["question_id"] in POSITIVE_PASSES[variant]
                reason = (
                    "核心答案正确，必要条件和操作得到实际上下文支持。"
                    if passed
                    else FAIL_REASONS.get(
                        row["question_id"],
                        "回答不满足核心答案、必要条件和证据支持的完整通过标准。",
                    )
                )
            reviews.append(
                {
                    "case_key": row["case_key"],
                    "question_id": row["question_id"],
                    "class": row["class"],
                    "split": row["split"],
                    "decision": row["decision"],
                    "complete_pass": passed,
                    "reason": reason,
                    "reviewer": "Codex",
                }
            )
        write_jsonl(RUN / variant / "codex_reviews.jsonl", reviews)
        by_split = {}
        for split in ("tune", "holdout", "all"):
            selected = reviews if split == "all" else [r for r in reviews if r["split"] == split]
            by_split[split] = {
                "case_count": len(selected),
                "complete_pass": sum(r["complete_pass"] for r in selected),
                "no_answer_pass": sum(
                    r["class"] == "no_answer" and r["complete_pass"] for r in selected
                ),
                "answerable_pass": sum(
                    r["class"] == "answerable" and r["complete_pass"] for r in selected
                ),
            }
        overall[variant] = by_split

    baseline = {
        row["case_key"]: row for row in read_jsonl(RUN / "baseline/codex_reviews.jsonl")
    }
    balanced = {
        row["case_key"]: row for row in read_jsonl(RUN / "balanced/codex_reviews.jsonl")
    }
    newly_passed = sorted(
        key for key in baseline if not baseline[key]["complete_pass"] and balanced[key]["complete_pass"]
    )
    regressed = sorted(
        key for key in baseline if baseline[key]["complete_pass"] and not balanced[key]["complete_pass"]
    )
    summary = {
        "reviewer": "Codex",
        "standard": (
            "Core answer correct, required steps and applicability conditions retained, "
            "and key claims supported by the actual supplied context."
        ),
        "results": overall,
        "balanced_vs_baseline": {
            "new_complete_pass": len(newly_passed),
            "new_complete_pass_case_keys": newly_passed,
            "regressed": len(regressed),
            "regressed_case_keys": regressed,
            "net_change": len(newly_passed) - len(regressed),
        },
    }
    (RUN / "codex_review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
