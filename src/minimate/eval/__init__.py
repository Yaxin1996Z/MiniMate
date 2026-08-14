"""Agent 评测系统 —— 可复现、可量化、可逐条复核

设计要点：
  - 评测集分模式覆盖 chat / react / plan / multi，任务为确定性本地任务（不依赖网络）
  - 判定用确定性断言 checker，而非 LLM 自评，保证可复现与可信
  - 每条用例在独立临时沙箱目录执行，工具只能操作沙箱内文件
  - 输出 Markdown + JSON 报告：逐条通过/失败 + 判定理由 + 耗时 + Token 消耗

用法：minimate --eval basic
"""

from .case import EvalCase, EvalCaseResult, EvalSummary
from .checkers import run_checker
from .suites import get_suite, list_suites
from .runner import EvalRunner
from .report import render_markdown, render_json

__all__ = [
    "EvalCase",
    "EvalCaseResult",
    "EvalSummary",
    "run_checker",
    "get_suite",
    "list_suites",
    "EvalRunner",
    "render_markdown",
    "render_json",
]
