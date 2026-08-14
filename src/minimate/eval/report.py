"""评测报告渲染 —— Markdown（人读）+ JSON（机器读/归档）"""

from __future__ import annotations

import json
import time

from ..llm import get_model
from .case import EvalCaseResult, EvalSummary


def render_json(results: list[EvalCaseResult], summary: EvalSummary) -> str:
    """渲染 JSON 报告（逐条结果 + 汇总）"""
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": get_model(),
        "summary": {
            "total": summary.total,
            "passed": summary.passed,
            "pass_rate": round(summary.pass_rate, 4),
            "total_duration": round(summary.total_duration, 2),
            "avg_duration": round(summary.avg_duration, 2),
            "total_tokens": summary.total_tokens,
            "avg_tokens": round(summary.avg_tokens, 1),
            "by_mode": summary.by_mode,
        },
        "cases": [
            {
                "id": r.case.id,
                "name": r.case.name,
                "mode": r.case.mode,
                "passed": r.passed,
                "duration": round(r.duration, 2),
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "total_tokens": r.total_tokens,
                "checker_reason": r.checker_reason,
                "error": r.error,
                "output": (r.output or "")[:300],
            }
            for r in results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render_markdown(
    suite: str, results: list[EvalCaseResult], summary: EvalSummary
) -> str:
    """渲染 Markdown 报告：汇总 + 按模式分组 + 逐条明细"""
    lines = [
        f"# MiniMate Agent 评测报告",
        "",
        f"- 评测集：`{suite}`",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- LLM 模型：{get_model()}",
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 用例总数 | {summary.total} |",
        f"| 通过数 | {summary.passed} |",
        f"| 通过率 | {summary.pass_rate * 100:.1f}% |",
        f"| 平均耗时 | {summary.avg_duration:.1f}s |",
        f"| Token 总消耗 | {summary.total_tokens} |",
        f"| 平均 Token / 用例 | {summary.avg_tokens:.0f} |",
        "",
        "## 按模式分组",
        "",
        "| 模式 | 用例数 | 通过 | 通过率 |",
        "|------|--------|------|--------|",
    ]
    for mode in sorted(summary.by_mode):
        bucket = summary.by_mode[mode]
        rate = bucket["passed"] / bucket["total"] * 100 if bucket["total"] else 0
        lines.append(
            f"| {mode} | {bucket['total']} | {bucket['passed']} | {rate:.0f}% |"
        )

    lines += ["", "## 逐条明细", ""]
    for r in results:
        mark = "✅ 通过" if r.passed else "❌ 失败"
        lines += [
            f"### {r.case.id} {r.case.name}（{r.case.mode}）{mark}",
            "",
            f"- **任务**：{r.case.prompt[:200]}",
            f"- **判定**：{r.checker_reason or '执行异常'}",
            f"- **耗时**：{r.duration:.1f}s ｜ **Token**：{r.total_tokens}"
            f"（prompt {r.prompt_tokens} + completion {r.completion_tokens}）",
        ]
        if r.error:
            lines.append(f"- **异常**：{r.error}")
        if r.output:
            lines += ["", "输出摘要：", "", "```text", r.output[:500], "```"]
        lines.append("")

    lines += [
        "---",
        "",
        "## 评测方法论",
        "",
        "- 判定使用**确定性断言 checker**（文件存在/内容包含/数值比对/命令退出码），"
        "不使用 LLM 自评，结果可复现",
        "- 每条用例在**独立临时沙箱目录**执行，工具只能操作沙箱内文件，互不影响",
        "- 评测集为本地确定性任务（不依赖网络），覆盖 chat / react / plan / multi 四种模式",
        "",
        "复现命令：`minimate --eval basic`",
    ]
    return "\n".join(lines)
