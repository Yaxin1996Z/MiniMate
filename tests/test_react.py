"""
ReAct 循环单元测试 —— 不调用真实 LLM，使用 mock 验证循环逻辑

运行：python -m unittest discover -s tests
"""

import unittest
from unittest.mock import patch
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from minimate.orchestrator import Agent, parse_plan
from minimate.tools import Tool, ToolExecutor, parse_react, truncate


def _make_agent(max_steps: int = 5, with_tool: bool = True) -> Agent:
    tools = ToolExecutor()
    if with_tool:
        tools.register(
            Tool(
                name="fake_tool",
                description="测试工具，返回固定结果",
                func=lambda args: f"结果[{args}]",
            )
        )
    return Agent(
        role="测试研究员",
        goal="验证 ReAct 循环",
        tools=tools,
        max_steps=max_steps,
    )


class ParseReactTest(unittest.TestCase):
    """ReAct 输出解析"""

    def test_standard_format(self):
        text = (
            "Thought: 需要查询知识库\n"
            "Action: fake_tool\n"
            "Action Input: 鲁迅\n"
        )
        parsed = parse_react(text)
        self.assertEqual(parsed["thought"], "需要查询知识库")
        self.assertEqual(parsed["action"], "fake_tool")
        self.assertEqual(parsed["action_input"], "鲁迅")
        self.assertEqual(parsed["final_answer"], "")

    def test_final_answer(self):
        parsed = parse_react("Thought: 信息足够\nFinal Answer: 这是最终答案")
        self.assertEqual(parsed["final_answer"], "这是最终答案")
        self.assertEqual(parsed["action"], "")

    def test_json_format(self):
        text = '{"thought": "查一下", "action": "fake_tool", "action_input": "x"}'
        parsed = parse_react(text)
        self.assertEqual(parsed["action"], "fake_tool")
        self.assertEqual(parsed["action_input"], "x")

    def test_empty_text(self):
        parsed = parse_react("")
        self.assertEqual(parsed["thought"], "")
        self.assertEqual(parsed["action"], "")


class TruncateTest(unittest.TestCase):
    """Observation 截断"""

    def test_short_kept(self):
        self.assertEqual(truncate("hello"), "hello")

    def test_long_truncated(self):
        result = truncate("a" * 5000, max_chars=100)
        self.assertLessEqual(len(result), 150)
        self.assertIn("已截断", result)


class ToolExecutorTest(unittest.TestCase):
    """ReAct 工具执行"""

    def test_execute_action_ok(self):
        tools = ToolExecutor()
        tools.register(Tool(name="fake_tool", description="t", func=lambda a: f"OK:{a}"))
        self.assertEqual(tools.execute_action("fake_tool", "abc"), "OK:abc")

    def test_execute_action_unknown(self):
        tools = ToolExecutor()
        result = tools.execute_action("no_such", "")
        self.assertIn("未知工具", result)

    def test_tools_prompt_uses_react_format(self):
        tools = ToolExecutor()
        tools.register(Tool(name="fake_tool", description="t", func=lambda a: a))
        prompt = tools.get_tools_prompt()
        self.assertIn("Thought:", prompt)
        self.assertIn("Final Answer:", prompt)
        self.assertNotIn("TOOL_CALL", prompt)


class AgentReactLoopTest(unittest.TestCase):
    """Agent.run 的 ReAct 循环行为（Function Calling 通道）"""

    @staticmethod
    def _fc_call(name="fake_tool", args=None):
        return {
            "id": "call_1",
            "name": name,
            "arguments": json.dumps(args or {"args": "q1"}),
        }

    @patch("minimate.orchestrator.llm.chat_tools")
    def test_loop_ends_on_final_answer(self, mock_chat_tools):
        """先调一次工具，再输出最终内容 → 返回答案"""
        mock_chat_tools.side_effect = [
            {"content": "", "tool_calls": [self._fc_call()]},
            {"content": "研究结论", "tool_calls": []},
        ]
        agent = _make_agent()
        result = agent.run("测试任务")

        self.assertEqual(result, "研究结论")
        self.assertEqual(mock_chat_tools.call_count, 2)

        # 验证工具结果已以 tool 角色回填第二轮消息
        second_call_messages = mock_chat_tools.call_args_list[1].args[0]
        last = second_call_messages[-1]
        self.assertEqual(last["role"], "tool")
        self.assertIn("结果[q1]", last["content"])

    @patch("minimate.orchestrator.llm.chat_tools")
    def test_stops_at_max_steps(self, mock_chat_tools):
        """模型一直调工具不结束 → 达到 max_steps 强制停止"""
        mock_chat_tools.return_value = {"content": "", "tool_calls": [self._fc_call()]}
        agent = _make_agent(max_steps=3)
        result = agent.run("测试任务")

        self.assertEqual(mock_chat_tools.call_count, 3)
        self.assertEqual(result, "")

    @patch("minimate.orchestrator.llm.chat_tools")
    def test_direct_content_is_final(self, mock_chat_tools):
        """FC 模式无工具调用时，content 即最终答案"""
        mock_chat_tools.return_value = {"content": "这是一段直接回答", "tool_calls": []}
        agent = _make_agent()
        result = agent.run("测试任务")
        self.assertEqual(result, "这是一段直接回答")

    @patch("minimate.orchestrator.llm.chat")
    @patch("minimate.orchestrator.llm.chat_tools")
    def test_fc_fallback_to_text(self, mock_chat_tools, mock_chat):
        """FC 通道异常时降级到文本协议"""
        mock_chat_tools.side_effect = RuntimeError("tools 不支持")
        mock_chat.return_value = "Thought: 直接回答\nFinal Answer: 文本协议答案"
        agent = _make_agent()
        result = agent.run("测试任务")
        self.assertEqual(result, "文本协议答案")
        mock_chat.assert_called_once()

    @patch("minimate.orchestrator.llm.call")
    def test_no_tools_uses_single_call(self, mock_call):
        """无工具 Agent 不进入 ReAct 循环，单次调用"""
        mock_call.return_value = "规划结果"
        agent = _make_agent(with_tool=False)
        result = agent.run("制定计划")
        self.assertEqual(result, "规划结果")
        mock_call.assert_called_once()


class ParsePlanTest(unittest.TestCase):
    """计划 JSON 解析"""

    def test_plain_json_array(self):
        text = '[{"step": "分析", "goal": "明确目标", "detail": "分解需求"}, {"step": "执行", "goal": "完成任务", "detail": "实施"}]'
        plan = parse_plan(text)
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0]["goal"], "明确目标")

    def test_codeblock_wrapped(self):
        text = '```json\n[{"step": "a", "goal": "g1"}]\n```'
        plan = parse_plan(text)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["goal"], "g1")

    def test_with_surrounding_text(self):
        text = '以下是计划：\n[{"step": "a", "goal": "g1"}]\n请执行。'
        plan = parse_plan(text)
        self.assertEqual(len(plan), 1)

    def test_invalid_returns_empty(self):
        self.assertEqual(parse_plan(""), [])
        self.assertEqual(parse_plan("没有计划"), [])
        self.assertEqual(parse_plan('{"not": "array"}'), [])


class AgentModeTest(unittest.TestCase):
    """三种执行模式"""

    @patch("minimate.orchestrator.llm.call")
    def test_chat_mode_single_call(self, mock_call):
        """chat 模式：无论有无工具，都只调用一次 LLM"""
        mock_call.return_value = "直接答案"
        agent = _make_agent()  # 带工具
        result = agent.run("问题", mode="chat")
        self.assertEqual(result, "直接答案")
        mock_call.assert_called_once()

    @patch("minimate.orchestrator.llm.call")
    @patch("minimate.orchestrator.llm.chat")
    def test_plan_mode_executes_steps(self, mock_chat, mock_call):
        """plan 模式：生成计划 → 逐步执行 → 汇总（无工具 Agent 每步走单次调用）"""
        plan_json = '[{"step": "a", "goal": "步骤一", "detail": "做A"}, {"step": "b", "goal": "步骤二", "detail": "做B"}]'
        mock_call.side_effect = [plan_json, "步骤一结果", "步骤二结果", "最终汇总答案"]
        agent = _make_agent(with_tool=False)
        result = agent.run("复杂任务", mode="plan")

        self.assertEqual(result, "最终汇总答案")
        self.assertEqual(mock_call.call_count, 4)  # 计划 + 2 步骤 + 汇总
        mock_chat.assert_not_called()

    def test_invalid_mode_raises(self):
        agent = _make_agent()
        with self.assertRaises(ValueError):
            agent.run("任务", mode="unknown")


if __name__ == "__main__":
    unittest.main()
