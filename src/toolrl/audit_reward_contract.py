"""Generate Gate 0 reports comparing strict evaluation with upstream reward.

This module only calls the frozen evaluator and the unmodified upstream reward.
It does not load a model, use a GPU, or alter the frozen data/manifest.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from toolrl.contract_cases import CASES, REFERENCE, render
from toolrl.eval_strict import evaluate_prediction
from toolrl.reward_contract import DEFAULT_REWARD_PATH, score_upstream_reward


def _number(value: float) -> str:
    return f"{value:.6g}"


def _strict_label(result: dict) -> str:
    if result["call_multiset_exact"]:
        return "通过"
    error = result["prediction_error"]
    return f"不通过（{error}）" if error else "不通过（调用不一致）"


def build_divergence_report(reward_path: Path) -> str:
    baseline = score_upstream_reward(REFERENCE, REFERENCE, reward_path)
    rows: list[str] = []
    divergences: list[str] = []

    for case in CASES:
        strict = evaluate_prediction(case["prediction"], REFERENCE)
        reward = score_upstream_reward(case["prediction"], REFERENCE, reward_path)
        strict_pass = strict["call_multiset_exact"]
        upstream_rejects = math.isclose(reward["correctness"], -3.0)

        if strict_pass and math.isclose(reward["total"], baseline["total"]):
            difference = "一致：严格通过，且上游给满分"
        elif not strict_pass and upstream_rejects:
            difference = "一致：严格拒绝，上游正确性分降至最低"
        elif not strict_pass:
            difference = "分歧：严格拒绝，但上游正确性分未降至最低"
            divergences.append(case["name"])
        else:
            difference = "分歧：严格通过，但上游未给满分"
            divergences.append(case["name"])

        upstream = (
            f"总分 {_number(reward['total'])}；格式 {_number(reward['format'])}；"
            f"正确性 {_number(reward['correctness'])}"
        )
        rows.append(
            f"| `{case['name']}` | {_strict_label(strict)} | {upstream} | {difference} |"
        )

    divergence_list = "、".join(f"`{name}`" for name in divergences)
    return f"""# 上游奖励与严格评测器分歧记录

## 判定口径

- 严格评测器输出布尔判定；调用多重集合必须完全一致才通过。
- 上游奖励函数原生输出连续分数，并没有通过阈值。本报告不人为添加阈值，完整记录总分、格式分和正确性分。
- 为了定位语义分歧，将“严格拒绝，但上游正确性分没有降至其最低值 -3”记为分歧。完全正确调用的上游基线为：格式 `{_number(baseline['format'])}`、正确性 `{_number(baseline['correctness'])}`、总分 `{_number(baseline['total'])}`。

## 同一批 12 类对抗样例

| 用例名 | 严格评测器判定 | 上游奖励判定（实际分数） | 差异说明 |
|---|---|---|---|
{chr(10).join(rows)}

## 分歧汇总

共发现 **{len(divergences)}** 个按上述口径定义的分歧用例：{divergence_list}。

其中 `two_tool_call_blocks` 风险最高：严格评测器以 `invalid_tool_call_count` 拒绝，上游却给满正确性分 `3`，仅损失格式分。`extra_parameter`、`extra_call` 和 `duplicate_call` 也获得接近满分的奖励，说明上游奖励是稠密的局部匹配分数，不能替代最终严格成功率。

本报告只记录差异，没有修改上游奖励函数或严格评测器。
"""


def build_hacking_report(reward_path: Path) -> str:
    single_call = {"name": "search", "parameters": {"query": "cats", "limit": 1}}
    unrelated_call = {"name": "clock", "parameters": {"timezone": "UTC"}}
    single_reference = render([single_call])
    parallel_reference = REFERENCE

    audits = [
        (
            "正确调用后复制一次正确调用",
            render([single_call, single_call]),
            single_reference,
            "仍得到 81.25% 的满分，构成高分漏洞；策略可能学会重复调用，而不是精确停止。",
        ),
        (
            "正确调用后追加无关调用",
            render([single_call, unrelated_call]),
            single_reference,
            "仍得到 81.25% 的满分，构成高分漏洞；策略可能学会过度调用工具，因为无关调用惩罚不足。",
        ),
        (
            "工具名正确但 parameters 为空",
            render([{"name": "search", "parameters": {}}]),
            single_reference,
            "得分为负，不构成高分漏洞；当前奖励会压低只猜工具名、遗漏全部参数的输出。",
        ),
        (
            "并行调用换序",
            render(list(reversed([
                {"name": "search", "parameters": {"query": "cats", "limit": 1}},
                {"name": "weather", "parameters": {"city": "Paris"}},
            ]))),
            parallel_reference,
            "并行调用本来没有语义顺序；与基线同分属于预期的不变性。",
        ),
        (
            "长段无关解释后给出正确调用",
            ("这是一段与工具执行无关的解释。" * 80) + "\n" + single_reference,
            single_reference,
            "仍得到 75% 的满分，构成高分漏洞；策略可能保留冗长前缀或格式污染，只靠末尾正确调用取分。",
        ),
    ]

    rows: list[str] = []
    findings: list[str] = []
    for name, prediction, reference, behavior in audits:
        baseline = score_upstream_reward(reference, reference, reward_path)
        score = score_upstream_reward(prediction, reference, reward_path)
        at_least = score["total"] >= baseline["total"] - 1e-12
        rows.append(
            f"| {name} | {_number(score['format'])} | {_number(score['correctness'])} | "
            f"{_number(score['total'])} | {_number(baseline['total'])} | {'是' if at_least else '否'} |"
        )
        findings.append(
            f"- **{name}**：实际总分 `{_number(score['total'])}/{_number(baseline['total'])}`。{behavior}"
        )

    return f"""# 上游奖励漏洞检查

## 方法

直接调用只读上游 `rlla.py` 的默认奖励配置。每种攻击都和同一参考答案的完全正确输出比较；单调用攻击使用单调用基线，并行换序使用双调用基线。两种完全正确基线的得分均为：格式 `1`、正确性 `3`、总分 `4`。

## 实际得分

| 构造输入 | 格式分 | 正确性分 | 总分 | 完全正确总分 | 是否大于或等于完全正确 |
|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

## 对 GRPO 行为的影响

{chr(10).join(findings)}

奖励漏洞的核心不是这些输入超过满分，而是部分错误行为仍可能获得很高的稠密奖励。若训练样本中这类输出经常出现，GRPO 会减弱“精确停止、只调用所需工具、严格遵守唯一输出区块”的学习压力。尤其应同时监控严格成功率，避免平均奖励上升而部署可用率没有同步上升。

本报告只审计实际得分，没有修改上游奖励实现。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-path", type=Path, default=DEFAULT_REWARD_PATH)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "reward_vs_evaluator_divergence.md").write_text(
        build_divergence_report(args.reward_path), encoding="utf-8"
    )
    (args.output_dir / "reward_hacking_audit.md").write_text(
        build_hacking_report(args.reward_path), encoding="utf-8"
    )
    print(args.output_dir / "reward_vs_evaluator_divergence.md")
    print(args.output_dir / "reward_hacking_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
