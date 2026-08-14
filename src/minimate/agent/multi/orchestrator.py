"""MultiAgentOrchestrator —— 多 Agent 编排器（主从架构）

编排器是"主"，子 Agent 是"从"：所有消息都经编排器路由，子 Agent 之间不直接对话。

六阶段工作流：
  1. 规划   ：规划者把用户任务拆成 JSON 执行计划
  2. 解析   ：解析成 ExecutionStep 列表，建立依赖关系（DAG）
  3. 执行   ：按拓扑分层，同批无依赖步骤并行派发给 Worker
  4. 审查   ：每步完成后检查者验收，不通过带反馈重试（最多 MAX_RETRIES 次）
  5. 残留   ：依赖失败导致无法执行的步骤显式标记 SKIPPED
  6. 汇总   ：按步骤顺序汇总结果，返回最终答案
"""

from __future__ import annotations

import contextlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ... import llm
from ...tools import ToolExecutor, truncate
from ...memory import MemoryManager
from ...logging import logger
from ..agent import _section
from ..task import topo_sort
from .role import AgentRole
from .message import AgentMessageType
from .sub_agent import SubAgent


class StepStatus(Enum):
    """计划步骤状态机"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExecutionStep:
    """计划步骤：id + 描述 + 类型 + 依赖 + 状态 + 结果

    字段名兼容 topo_sort（task_id / depends_on），可直接参与 DAG 分层。
    """

    task_id: str
    goal: str
    step_type: str = "ANALYSIS"
    detail: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: str = ""
    issues: list[str] = field(default_factory=list)
    retries: int = 0

    @classmethod
    def from_plan_dict(cls, data: dict, index: int) -> "ExecutionStep":
        deps = data.get("dependencies") or data.get("depends_on") or []
        return cls(
            task_id=str(data.get("id") or data.get("step") or f"step_{index}"),
            goal=(
                data.get("description")
                or data.get("goal")
                or data.get("step")
                or "步骤"
            ),
            step_type=str(data.get("type") or "ANALYSIS").upper(),
            detail=str(data.get("detail") or ""),
            depends_on=[str(d) for d in deps],
        )


# ============================================================
# JSON 解析（保守策略）
# ============================================================

def _extract_json_array(text: str) -> str:
    """提取文本中的 JSON 数组（兼容 ```json 代码块包裹与嵌套结构）"""
    m = re.search(r"```(?:json)?\s*(\[.*)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start = text.find("[")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _extract_json_object(text: str) -> str:
    """提取文本中的 JSON 对象（兼容 ```json 代码块包裹与嵌套花括号）"""
    m = re.search(r"```(?:json)?\s*(\{.*)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return ""


def _parse_plan(text: str) -> list[dict]:
    """解析规划者输出：兼容裸数组与 {"summary":..., "steps":[...]} 两种格式"""
    if not text or text.startswith("[API 错误]"):
        return []

    # 1) 整体是数组
    array_text = _extract_json_array(text)
    if array_text:
        try:
            data = json.loads(array_text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]

    # 2) 对象包裹的 steps 字段
    obj_text = _extract_json_object(text)
    if obj_text:
        try:
            data = json.loads(obj_text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            steps = data.get("steps")
            if isinstance(steps, list):
                return [d for d in steps if isinstance(d, dict)]
    return []


def _parse_review_feedback(text: str) -> tuple[bool, list[str], list[str]]:
    """解析检查者输出：返回 (approved, issues, suggestions)

    保守策略：空内容 / JSON 解析失败 / 缺少 approved 字段 → 默认不通过；
    仅 approved 显式为 true 才放行。宁可多审一次，不可放过一个问题。
    """
    if not text or text.startswith("[API 错误]"):
        return False, [], []

    obj_text = _extract_json_object(text)
    if obj_text:
        try:
            data = json.loads(obj_text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            approved = data.get("approved")
            issues = data.get("issues") or []
            suggestions = data.get("suggestions") or []
            if not isinstance(issues, list):
                issues = []
            if not isinstance(suggestions, list):
                suggestions = []
            if isinstance(approved, bool):
                return approved, [str(i) for i in issues], [str(s) for s in suggestions]
            return False, [], []

    # 非 JSON 兜底：必须同时有肯定关键词且无否定关键词才视为通过
    has_negative = ("未通过" in text) or ("不通过" in text)
    has_positive = ("通过" in text) or ("合格" in text)
    if has_negative:
        return False, [text[:500]], []
    if not has_positive:
        return False, [text[:500]], []
    return True, [], []


# ============================================================
# 编排器
# ============================================================

class MultiAgentOrchestrator:
    """多 Agent 协作编排器：拿着任务找规划者拆、找执行者干、找检查者验"""

    def __init__(
        self,
        tools: Optional[ToolExecutor] = None,
        memory: Optional[MemoryManager] = None,
        planner: Optional[SubAgent] = None,
        workers: Optional[list[SubAgent]] = None,
        reviewer: Optional[SubAgent] = None,
        max_steps: int = 8,
        worker_count: int = 2,
        max_review_retries: int = 2,
        result_preview: int = 500,
    ):
        self.tools = tools
        self.memory = memory
        self.planner = planner or SubAgent(
            "planner", AgentRole.PLANNER, tools, memory
        )
        self.workers = workers or [
            SubAgent(f"worker-{i + 1}", AgentRole.WORKER, tools, memory, max_steps)
            for i in range(worker_count)
        ]
        self.reviewer = reviewer or SubAgent(
            "reviewer", AgentRole.REVIEWER, tools, memory
        )
        self.max_review_retries = max_review_retries
        self.result_preview = result_preview
        self._worker_index = 0

    # ----------------------------------------------------------
    # 主入口
    # ----------------------------------------------------------

    def run(self, task: str, context: str = "") -> str:
        """多 Agent 协作入口：规划 → 解析 → 执行 → 审查 → 残留 → 汇总"""
        print(_section("Multi-Agent", f"任务：{task}", fg="magenta"))

        # 1) 规划
        plan_msg = self.planner.plan(task, context)
        if plan_msg.type is AgentMessageType.ERROR or not plan_msg.content.strip():
            logger.error("Multi-Agent 规划失败：%s", plan_msg.content[:200])
            print(_section("系统", f"规划阶段失败：{plan_msg.content[:200]}", fg="red"))
            return plan_msg.content or "规划失败，无法生成执行计划。"
        print(_section("规划者", plan_msg.content, fg="yellow"))

        # 2) 解析计划
        plan_dicts = _parse_plan(plan_msg.content)
        if not plan_dicts:
            logger.warning("Multi-Agent 计划解析失败，回退到单 Agent ReAct")
            print(_section("系统", "计划解析失败，回退到单 Agent ReAct", fg="magenta"))
            return self._fallback_react(task, context)

        steps = [
            ExecutionStep.from_plan_dict(d, i) for i, d in enumerate(plan_dicts, 1)
        ]
        # 缺失依赖时默认线性依赖前一步，保持计划可执行
        for i, step in enumerate(steps):
            if not step.depends_on and i > 0:
                step.depends_on = [steps[i - 1].task_id]

        # 3) 拓扑分层（同批内无依赖，可并行）
        try:
            batches = topo_sort(steps)
        except ValueError as e:
            logger.warning("多 Agent 计划 DAG 异常（%s），按顺序执行", e)
            print(_section("系统", f"计划依赖异常（{e}），按顺序执行", fg="magenta"))
            batches = [[s] for s in steps]

        plan_summary = "\n".join(
            f"  {s.task_id}. {s.goal} [{s.step_type}]"
            f"{' -> 依赖 ' + ','.join(s.depends_on) if s.depends_on else ''}"
            for s in steps
        )
        print(_section(
            f"执行计划（共 {len(steps)} 步，{len(batches)} 层）",
            plan_summary,
            fg="magenta",
        ))

        # 4) 执行 + 审查（分层并行，输出按 step_id 顺序）
        results: dict[str, str] = {}
        for batch in batches:
            self._run_batch(batch, steps, results)

        # 5) 残留处理
        skipped = [s for s in steps if s.status is StepStatus.SKIPPED]
        failed = [s for s in steps if s.status is StepStatus.FAILED]
        if skipped or failed:
            print(_section("系统", self._residual_report(skipped, failed), fg="yellow"))

        # 6) 汇总
        final = self._summarize(task, context, steps)
        print(_section("助手回答", final, fg="green"))
        return final

    # ----------------------------------------------------------
    # 执行 + 审查
    # ----------------------------------------------------------

    def _run_batch(
        self,
        batch: list[ExecutionStep],
        all_steps: list[ExecutionStep],
        results: dict[str, str],
    ) -> None:
        """执行一批可并行的步骤：先标记跳过项，再并行执行，最后按原顺序输出"""
        pending: list[ExecutionStep] = []
        skipped_outputs: dict[str, str] = {}

        for step in batch:
            if not self._deps_ok(step, results):
                step.status = StepStatus.SKIPPED
                results[step.task_id] = "[已跳过] 依赖步骤失败或未完成"
                skipped_outputs[step.task_id] = _section(
                    "系统",
                    f"步骤 {step.task_id} 已跳过（依赖步骤失败或未完成）",
                    fg="yellow",
                )
            else:
                pending.append(step)

        if not pending:
            for step in batch:
                if step.task_id in skipped_outputs:
                    print(skipped_outputs[step.task_id])
            return

        if len(pending) == 1:
            step = pending[0]
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    results[step.task_id] = self._run_step(step, all_steps, results)
                except Exception as e:
                    step.status = StepStatus.FAILED
                    results[step.task_id] = f"[执行异常] {e}"
                    logger.exception("多 Agent 单步执行异常：%s", e)
            print(buf.getvalue(), end="")
            return

        buffers: dict[str, io.StringIO] = {}
        futures: dict = {}
        with ThreadPoolExecutor(
            max_workers=min(len(pending), len(self.workers))
        ) as executor:
            for step in pending:
                buf = io.StringIO()
                buffers[step.task_id] = buf
                futures[executor.submit(self._run_step_captured, step, all_steps, results, buf)] = step
            for future in as_completed(futures):
                step = futures[future]
                try:
                    results[step.task_id] = future.result()
                except Exception as e:
                    step.status = StepStatus.FAILED
                    results[step.task_id] = f"[执行异常] {e}"
                    logger.exception("多 Agent 并行步骤异常：%s", e)

        # 按 batch 原顺序 flush，保证用户看到的执行过程有序
        for step in batch:
            if step.task_id in skipped_outputs:
                print(skipped_outputs[step.task_id])
            buf = buffers.get(step.task_id)
            if buf:
                print(buf.getvalue(), end="")

    def _run_step_captured(
        self,
        step: ExecutionStep,
        all_steps: list[ExecutionStep],
        results: dict[str, str],
        buf: io.StringIO,
    ) -> str:
        """并行执行封装：把该步骤的全部 stdout 捕获到独立 buffer"""
        with contextlib.redirect_stdout(buf):
            return self._run_step(step, all_steps, results)

    def _run_step(
        self,
        step: ExecutionStep,
        all_steps: list[ExecutionStep],
        results: dict[str, str],
    ) -> str:
        """单步：执行者干活 → 检查者验收 → 不通过带反馈重试（最多 N 次）"""
        step.status = StepStatus.RUNNING
        print(_section(
            "系统", f"执行任务 {step.task_id}（{step.step_type}）：{step.goal}", fg="magenta"
        ))

        worker = self._next_worker()
        print(_section(
            "任务分派", f"步骤 {step.task_id} -> {worker.name}（{worker.role.display}）", fg="cyan"
        ))

        step_text = self._build_step_text(step)
        dep_ctx = self._build_step_context(all_steps, step, results)

        result_msg = worker.execute(step_text, dep_ctx)
        result = result_msg.content
        print(_section("执行者返回", truncate(result, 1000), fg="green"))
        worker.clear_history()

        # 审查（保守策略：无法解析默认不通过）
        approved, issues, suggestions = self._review(step_text, result)
        retries = 0
        while not approved and retries < self.max_review_retries:
            retries += 1
            step.retries = retries
            print(_section(
                "检查者",
                f"审查未通过（第 {retries} 次重试）\n"
                f"问题：{issues or '无'}\n建议：{suggestions or '无'}",
                fg="yellow",
            ))
            feedback = self._build_feedback(issues, suggestions)
            result_msg = worker.execute(step_text, dep_ctx + feedback)
            result = result_msg.content
            print(_section("执行者返回（重试）", truncate(result, 1000), fg="green"))
            worker.clear_history()
            approved, issues, suggestions = self._review(step_text, result)

        if approved:
            step.status = StepStatus.COMPLETED
            print(_section("检查者", "审查通过，放行", fg="green"))
        else:
            # 重试耗尽：保留当前结果，显式提示用户（不再死磕）
            step.status = StepStatus.COMPLETED
            step.issues = issues
            print(_section(
                "检查者",
                f"重试 {self.max_review_retries} 次仍未通过，保留当前结果（建议人工复核）",
                fg="red",
            ))

        step.result = result
        self.reviewer.clear_history()
        return result

    def _review(
        self, step_text: str, result: str
    ) -> tuple[bool, list[str], list[str]]:
        """调用检查者审查，返回 (approved, issues, suggestions)"""
        review_msg = self.reviewer.review(step_text, result)
        if review_msg.type is AgentMessageType.ERROR:
            print(_section("检查者", f"审查调用失败：{review_msg.content[:200]}", fg="red"))
            return False, [], []
        print(_section("检查者", review_msg.content, fg="cyan"))
        return _parse_review_feedback(review_msg.content)

    # ----------------------------------------------------------
    # 上下文构建
    # ----------------------------------------------------------

    def _deps_ok(self, step: ExecutionStep, results: dict[str, str]) -> bool:
        """当前步骤的所有依赖都必须已完成"""
        return all(
            dep in results and not results[dep].startswith(("[已跳过]", "[执行异常]"))
            for dep in step.depends_on
        )

    def _build_step_text(self, step: ExecutionStep) -> str:
        """步骤描述 → 派发给执行者的任务文本"""
        text = f"执行计划步骤 {step.task_id}：{step.goal}"
        if step.detail:
            text += f"\n具体要求：{step.detail}"
        if step.step_type in ("VERIFICATION", "VERIFY"):
            text += "\n请使用工具实际执行验证命令并确认结果，不要直接回答。"
        return text

    def _build_step_context(
        self,
        all_steps: list[ExecutionStep],
        current: ExecutionStep,
        results: dict[str, str],
    ) -> str:
        """注入已完成依赖步骤的结果摘要（截断到 result_preview，防止撑爆上下文）"""
        lines = ["总任务上下文："]
        for step in all_steps:
            if (
                step.status is StepStatus.COMPLETED
                and step.task_id in current.depends_on
                and step.task_id in results
            ):
                preview = truncate(results[step.task_id], self.result_preview)
                lines.append(f"已完成的依赖步骤 [{step.task_id}]: {step.goal}")
                lines.append(f"结果：{preview}")
        if len(lines) > 1:
            return "\n".join(lines)
        return ""

    def _build_feedback(
        self, issues: list[str], suggestions: list[str]
    ) -> str:
        """审查拒绝原因 → 注入执行者上下文的反馈文本"""
        parts = []
        if issues:
            parts.append("问题：\n" + "\n".join(f"- {i}" for i in issues))
        if suggestions:
            parts.append("改进建议：\n" + "\n".join(f"- {s}" for s in suggestions))
        body = "\n".join(parts) or "请自行检查执行结果是否遗漏或出错，并重新完成。"
        return "\n\n之前的执行结果被审查拒绝，原因：\n" + body

    # ----------------------------------------------------------
    # 残留处理 / 汇总
    # ----------------------------------------------------------

    def _residual_report(
        self, skipped: list[ExecutionStep], failed: list[ExecutionStep]
    ) -> str:
        """残留步骤提示：依赖失败导致无法执行的步骤显式告知用户"""
        lines = []
        if failed:
            lines.append("失败步骤：" + ", ".join(s.task_id for s in failed))
        if skipped:
            lines.append(
                "以下步骤因依赖失败被跳过："
                + ", ".join(f"{s.task_id}({s.goal})" for s in skipped)
            )
        lines.append("请人工检查以上步骤，必要时重新发起任务。")
        return "\n".join(lines)

    def _summarize(
        self, task: str, context: str, steps: list[ExecutionStep]
    ) -> str:
        """汇总各步骤结果，生成最终完整答案"""
        body = []
        for i, step in enumerate(steps, 1):
            status_mark = {
                StepStatus.COMPLETED: "完成",
                StepStatus.FAILED: "失败",
                StepStatus.SKIPPED: "跳过",
            }.get(step.status, "未知")
            text = (step.result or "(无输出)").strip()
            if step.issues:
                text += f"\n（审查未完全通过，遗留问题：{step.issues[:3]}）"
            body.append(
                f"【步骤 {i}（{step.task_id}）- {status_mark}】\n{truncate(text, 2000)}"
            )
        prompt = (
            f"以下是多 Agent 团队按计划执行「{task}」各步骤得到的结果：\n\n"
            + "\n\n".join(body)
            + "\n\n请基于这些结果，输出任务的最终完整答案"
              "（结构清晰、直接可用，不要复述执行过程；"
              "若有步骤被跳过或失败，请明确指出）。"
        )
        if context:
            prompt += f"\n\n参考上下文：\n{truncate(context, 1000)}"
        return llm.call(
            prompt,
            system="你是多 Agent 协作结果汇总助手。",
            temperature=0.3,
        )

    def _fallback_react(self, task: str, context: str) -> str:
        """计划解析失败时回退到第一个执行者的 ReAct 循环"""
        if not self.workers:
            return "无可用执行者。"
        msg = self.workers[0].execute(task, context)
        return msg.content

    def _next_worker(self) -> SubAgent:
        """轮询分配 Worker：一个在干活时，另一个可接下一个步骤（为并行做准备）"""
        worker = self.workers[self._worker_index % len(self.workers)]
        self._worker_index += 1
        return worker
