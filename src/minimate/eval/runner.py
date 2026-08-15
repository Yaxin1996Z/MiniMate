"""评测运行器 —— 沙箱隔离 + 逐条执行 + 汇总统计

流程：为每条用例创建独立临时目录（沙箱）→ 可选 setup 准备环境 →
按模式执行 Agent → 确定性 checker 判定 → 记录耗时与 Token → 生成报告。
"""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional

from .. import llm
from ..agent import Agent, MultiAgentOrchestrator
from ..tools import ToolExecutor, register_all_tools
from .case import EvalCase, EvalCaseResult, EvalSummary
from .checkers import run_checker
from .report import render_json, render_markdown
from .suites_ai import get_suite


class EvalRunner:
    """评测运行器

    mode_filter：只运行指定模式（如 {"react", "multi"}）；None 表示全部
    max_cases：最多运行的用例数（便于快速验证）
    """

    def __init__(
        self,
        mode_filter: Optional[set[str]] = None,
        case_ids: Optional[set[str]] = None,
        max_cases: Optional[int] = None,
        max_steps: int = 8,
        results_dir: str = "",
    ):
        self.mode_filter = mode_filter
        self.case_ids = case_ids
        self.max_cases = max_cases
        self.max_steps = max_steps
        self.results_dir = os.path.abspath(
            results_dir
            or os.path.join(
                os.path.dirname(__file__), "..", "..", "..", "eval", "results"
            )
        )
        # 工具执行器全程复用（纯函数 + 相对路径基于沙箱 cwd），避免重复装配
        self._tools = ToolExecutor()
        register_all_tools(self._tools)

    # ----------------------------------------------------------
    # 入口
    # ----------------------------------------------------------

    def run_suite(self, suite: str = "basic") -> tuple[list[EvalCaseResult], EvalSummary, dict]:
        """运行整个评测集，返回 (逐条结果, 汇总统计, 报告文件路径)"""
        cases = self._load_suite(suite)
        if self.mode_filter:
            cases = [c for c in cases if c.mode in self.mode_filter]
        if self.case_ids:
            cases = [c for c in cases if c.id in self.case_ids]
        if self.max_cases is not None and self.max_cases > 0:
            cases = cases[: self.max_cases]

        results = [self.run_case(case) for case in cases]
        summary = self._summarize(results)
        paths = self._save_report(suite, results, summary)
        return results, summary, paths

    @staticmethod
    def _load_suite(suite: str) -> list[EvalCase]:
        """按套件名加载用例：basic/ai 为 AI 生成用例，real 为真实用户用例"""
        if suite == "real":
            from .suites_real import get_suite as _get_suite
        else:
            from .suites_ai import get_suite as _get_suite
        return _get_suite(suite)

    # ----------------------------------------------------------
    # 单条用例
    # ----------------------------------------------------------

    def run_case(self, case: EvalCase) -> EvalCaseResult:
        """在独立临时沙箱中执行一条用例"""
        sandbox = tempfile.mkdtemp(prefix="minimate_eval_")
        old_cwd = os.getcwd()
        try:
            os.chdir(sandbox)
            self._prepare(case)

            start = time.time()
            stats_before = llm.get_stats()
            output, trace = self._execute(case)
            stats_after = llm.get_stats()
            duration = time.time() - start

            passed, reason = run_checker(case.checker, case.expected, output, cwd=sandbox)
            return EvalCaseResult(
                case=case,
                passed=passed,
                output=(output or "")[:800],
                trace=trace,
                checker_reason=reason,
                duration=duration,
                prompt_tokens=max(0, stats_after["prompt_tokens"] - stats_before["prompt_tokens"]),
                completion_tokens=max(
                    0, stats_after["completion_tokens"] - stats_before["completion_tokens"]
                ),
            )
        except Exception as e:
            return EvalCaseResult(
                case=case,
                passed=False,
                error=str(e),
                duration=0.0,
            )
        finally:
            os.chdir(old_cwd)
            shutil.rmtree(sandbox, ignore_errors=True)

    def _prepare(self, case: EvalCase) -> None:
        """执行 setup：内置 write/mkdir 操作 + shell 命令"""
        for op in case.setup or []:
            if isinstance(op, dict):
                if "write" in op:
                    info = op["write"]
                    path = str(info["path"])
                    parent = os.path.dirname(path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(str(info["content"]))
                elif "mkdir" in op:
                    os.makedirs(str(op["mkdir"]), exist_ok=True)
                continue
            subprocess.run(
                str(op), shell=True, cwd=os.getcwd(), capture_output=True, timeout=60
            )

    def _execute(self, case: EvalCase) -> tuple[str, str]:
        """按模式执行 Agent，捕获 stdout 作为交互 trace"""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            if case.mode == "multi":
                orchestrator = MultiAgentOrchestrator(
                    tools=self._tools,
                    memory=None,
                    max_steps=self.max_steps,
                )
                output = orchestrator.run(case.prompt)
            else:
                agent = Agent(
                    role="评测执行者",
                    goal="严格按照要求完成评测任务",
                    tools=self._tools,
                    max_steps=self.max_steps,
                )
                output = agent.run(case.prompt, mode=case.mode)
        return (output or "").strip(), buf.getvalue()

    # ----------------------------------------------------------
    # 统计与报告
    # ----------------------------------------------------------

    def _summarize(self, results: list[EvalCaseResult]) -> EvalSummary:
        summary = EvalSummary(total=len(results))
        for r in results:
            if r.passed:
                summary.passed += 1
            mode = r.case.mode
            bucket = summary.by_mode.setdefault(mode, {"total": 0, "passed": 0})
            bucket["total"] += 1
            if r.passed:
                bucket["passed"] += 1
            summary.total_duration += r.duration
            summary.total_tokens += r.total_tokens
        return summary

    def _save_report(
        self,
        suite: str,
        results: list[EvalCaseResult],
        summary: EvalSummary,
    ) -> dict:
        """报告输出到 eval/results/<时间戳>/，返回 {markdown, json} 路径"""
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(self.results_dir, stamp)
        os.makedirs(out_dir, exist_ok=True)

        md_path = os.path.join(out_dir, "report.md")
        json_path = os.path.join(out_dir, "report.json")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(render_markdown(suite, results, summary))
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(render_json(results, summary))
        return {"markdown": md_path, "json": json_path}
