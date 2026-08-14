"""多 Agent 编排单元测试 —— 不调用真实 LLM，使用 fake 子 Agent 验证编排逻辑

覆盖：
  - _parse_plan 计划解析（数组 / 对象 / 代码块 / 垃圾输入）
  - _parse_review_feedback 保守策略（缺 approved / 非 JSON 关键词兜底）
  - ExecutionStep 建模 + topo_sort 分层
  - 编排器六阶段流程：规划 → 执行 → 审查 → 重试 → 残留跳过 → 汇总
"""

import sys
import unittest
from unittest.mock import patch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from minimate.agent import AgentMessage, AgentMessageType, AgentRole
from minimate.agent.multi.orchestrator import (
    ExecutionStep,
    MultiAgentOrchestrator,
    StepStatus,
    _parse_plan,
    _parse_review_feedback,
)
from minimate.agent.task import topo_sort


class FakeSubAgent:
    """编排器测试用假子 Agent：可编程返回计划 / 执行 / 审查结果"""

    def __init__(
        self,
        name: str,
        role: AgentRole,
        plan_result: str = "",
        execute_results: list[str] | None = None,
        review_results: list[str] | None = None,
        raise_on_execute: bool = False,
    ):
        self.name = name
        self.role = role
        self.plan_result = plan_result
        self.execute_results = list(execute_results or [])
        self.review_results = list(review_results or [])
        self.raise_on_execute = raise_on_execute
        self.execute_calls: list[tuple[str, str]] = []
        self.review_calls: list[tuple[str, str]] = []
        self.clear_count = 0

    def plan(self, task: str, context: str = "") -> AgentMessage:
        return AgentMessage(
            self.name, self.role, self.plan_result, AgentMessageType.RESULT
        )

    def execute(self, task: str, context: str = "") -> AgentMessage:
        self.execute_calls.append((task, context))
        if self.raise_on_execute:
            raise RuntimeError("模拟执行异常")
        content = self.execute_results.pop(0) if self.execute_results else "执行完成"
        return AgentMessage(self.name, self.role, content, AgentMessageType.RESULT)

    def review(self, task: str, result: str) -> AgentMessage:
        self.review_calls.append((task, result))
        content = (
            self.review_results.pop(0)
            if self.review_results
            else '{"approved": true}'
        )
        return AgentMessage(self.name, self.role, content, AgentMessageType.RESULT)

    def clear_history(self):
        self.clear_count += 1


def _make_orchestrator(
    plan_result: str,
    execute_results: list[str] | None = None,
    review_results: list[str] | None = None,
    raise_on_execute: bool = False,
) -> MultiAgentOrchestrator:
    planner = FakeSubAgent("planner", AgentRole.PLANNER, plan_result=plan_result)
    workers = [
        FakeSubAgent("worker-1", AgentRole.WORKER, execute_results=execute_results),
        FakeSubAgent("worker-2", AgentRole.WORKER, execute_results=execute_results),
    ]
    reviewer = FakeSubAgent(
        "reviewer", AgentRole.REVIEWER, review_results=review_results
    )
    return MultiAgentOrchestrator(
        tools=None,
        memory=None,
        planner=planner,
        workers=workers,
        reviewer=reviewer,
        max_review_retries=2,
    )


_PLAN_OK = (
    '{"summary": "测试", "steps": ['
    '{"id": "step_1", "description": "创建文件", "type": "FILE_WRITE", "dependencies": []},'
    '{"id": "step_2", "description": "验证文件", "type": "VERIFICATION", "dependencies": ["step_1"]}'
    "]}"
)


class ParsePlanTest(unittest.TestCase):
    """规划者输出解析"""

    def test_object_with_steps(self):
        steps = _parse_plan(_PLAN_OK)
        self.assertEqual(len(steps), 2)
        self.assertEqual(steps[0]["id"], "step_1")
        self.assertEqual(steps[1]["dependencies"], ["step_1"])

    def test_bare_array(self):
        raw = '[{"id": "step_1", "description": "a", "type": "ANALYSIS"}]'
        steps = _parse_plan(raw)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["id"], "step_1")

    def test_code_fence(self):
        raw = "```json\n" + _PLAN_OK + "\n```"
        self.assertEqual(len(_parse_plan(raw)), 2)

    def test_garbage(self):
        self.assertEqual(_parse_plan(""), [])
        self.assertEqual(_parse_plan("[API 错误] timeout"), [])
        self.assertEqual(_parse_plan("完全不是 JSON"), [])


class ParseReviewTest(unittest.TestCase):
    """检查者审批解析（保守策略）"""

    def test_approved_true(self):
        ok, issues, suggestions = _parse_review_feedback(
            '{"approved": true, "summary": "s", "issues": [], "suggestions": []}'
        )
        self.assertTrue(ok)

    def test_approved_false(self):
        ok, issues, _ = _parse_review_feedback(
            '{"approved": false, "issues": ["缺测试"], "suggestions": ["补测试"]}'
        )
        self.assertFalse(ok)
        self.assertEqual(issues, ["缺测试"])

    def test_nested_object_in_json(self):
        """嵌套花括号不应截断 JSON 提取"""
        text = (
            '```json\n{"approved": true, '
            '"issues": [{"code": "E1", "msg": "缺测试"}], "suggestions": []}\n```'
        )
        ok, issues, _ = _parse_review_feedback(text)
        self.assertTrue(ok)
        self.assertIn("E1", issues[0])

    def test_missing_approved_default_reject(self):
        ok, _, _ = _parse_review_feedback('{"summary": "no approved field"}')
        self.assertFalse(ok)

    def test_empty_default_reject(self):
        self.assertFalse(_parse_review_feedback("")[0])
        self.assertFalse(_parse_review_feedback("[API 错误] 网络超时")[0])

    def test_keyword_fallback(self):
        self.assertTrue(_parse_review_feedback("结果通过，合格")[0])
        self.assertFalse(_parse_review_feedback("结果未通过，需要重做")[0])
        self.assertFalse(_parse_review_feedback("无法判断")[0])


class ExecutionStepTest(unittest.TestCase):
    """步骤建模 + DAG 拓扑分层"""

    def test_from_plan_dict(self):
        step = ExecutionStep.from_plan_dict(
            {"id": "step_1", "description": "创建文件", "type": "FILE_WRITE"}, 1
        )
        self.assertEqual(step.task_id, "step_1")
        self.assertEqual(step.step_type, "FILE_WRITE")
        self.assertEqual(step.status, StepStatus.PENDING)

    def test_topo_sort_batches(self):
        steps = [
            ExecutionStep("step_1", "a", depends_on=[]),
            ExecutionStep("step_2", "b", depends_on=["step_1"]),
            ExecutionStep("step_3", "c", depends_on=["step_1"]),
        ]
        batches = topo_sort(steps)
        self.assertEqual([s.task_id for s in batches[0]], ["step_1"])
        self.assertEqual(
            {s.task_id for s in batches[1]}, {"step_2", "step_3"}
        )


class OrchestratorFlowTest(unittest.TestCase):
    """编排器六阶段流程"""

    def test_happy_path(self):
        orch = _make_orchestrator(
            _PLAN_OK,
            execute_results=["文件已写入", "验证通过"],
            review_results=['{"approved": true}', '{"approved": true}'],
        )
        with patch(
            "minimate.agent.multi.orchestrator.llm.call",
            return_value="汇总完成",
        ):
            answer = orch.run("创建并验证文件")

        self.assertEqual(answer, "汇总完成")
        steps = [orch.planner, orch.workers[0], orch.workers[1], orch.reviewer]
        self.assertEqual(orch.workers[0].clear_count, 1)
        self.assertEqual(orch.reviewer.clear_count, 2)

    def test_review_reject_then_retry(self):
        orch = _make_orchestrator(
            _PLAN_OK,
            execute_results=["第一次结果", "修改后结果", "验证通过"],
            review_results=[
                '{"approved": false, "issues": ["缺内容"], "suggestions": ["补内容"]}',
                '{"approved": true}',
                '{"approved": true}',
            ],
        )
        with patch(
            "minimate.agent.multi.orchestrator.llm.call",
            return_value="汇总完成",
        ):
            orch.run("创建并验证文件")

        # 第一步被拒绝一次后重试成功
        worker = orch.workers[0]
        self.assertEqual(len(worker.execute_calls), 2)
        self.assertIn("之前的执行结果被审查拒绝", worker.execute_calls[1][1])

    def test_execute_exception_skips_dependents(self):
        orch = _make_orchestrator(
            _PLAN_OK,
            execute_results=["不会用到"],
            raise_on_execute=True,
        )
        with patch(
            "minimate.agent.multi.orchestrator.llm.call",
            return_value="汇总完成",
        ):
            orch.run("创建并验证文件")

        # step_1 执行异常 → FAILED；step_2 依赖失败 → SKIPPED
        self.assertEqual(orch.workers[0].execute_calls[0][0], "执行计划步骤 step_1：创建文件")
        # step_2 不应被派发给 worker（依赖失败直接跳过）
        self.assertEqual(len(orch.workers[0].execute_calls), 1)

    def test_plan_parse_failure_falls_back(self):
        orch = _make_orchestrator(
            "无法解析的计划文本",
            execute_results=["fallback 结果"],
        )
        with patch(
            "minimate.agent.multi.orchestrator.llm.call",
            return_value="fallback 结果",
        ):
            answer = orch.run("测试任务")
        self.assertEqual(answer, "fallback 结果")
        self.assertEqual(len(orch.workers[0].execute_calls), 1)


if __name__ == "__main__":
    unittest.main()
