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

from minimate.orchestrator import Agent, PlanTask, parse_plan, topo_sort
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

    @patch("minimate.agent.agent.llm.chat_tools")
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

    @patch("minimate.agent.agent.llm.chat_tools")
    def test_multiple_tool_calls_single_assistant_message(self, mock_chat_tools):
        """一次返回多个 tool_calls：只回填一条 assistant（含全部调用）+ 每条结果一个 tool 消息"""
        mock_chat_tools.side_effect = [
            {
                "content": "两个工具",
                "tool_calls": [
                    {"id": "call_1", "name": "fake_tool", "arguments": '{"args": "q1"}'},
                    {"id": "call_2", "name": "fake_tool", "arguments": '{"args": "q2"}'},
                ],
            },
            {"content": "最终答案", "tool_calls": []},
        ]
        agent = _make_agent()
        result = agent.run("并行调用测试")

        self.assertEqual(result, "最终答案")
        second_call_messages = mock_chat_tools.call_args_list[1].args[0]
        assistant_msgs = [m for m in second_call_messages if m["role"] == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(len(assistant_msgs[0]["tool_calls"]), 2)
        tool_msgs = [m for m in second_call_messages if m["role"] == "tool"]
        self.assertEqual(len(tool_msgs), 2)
        self.assertEqual({m["tool_call_id"] for m in tool_msgs}, {"call_1", "call_2"})

    @patch("minimate.agent.agent.llm.chat_tools")
    def test_stops_at_max_steps(self, mock_chat_tools):
        """模型一直调工具不结束 → 达到 max_steps 后追加一次强制收尾调用"""
        mock_chat_tools.return_value = {"content": "", "tool_calls": [self._fc_call()]}
        agent = _make_agent(max_steps=3)
        result = agent.run("测试任务")

        self.assertEqual(mock_chat_tools.call_count, 4)  # 3 轮工具 + 1 次强制收尾
        self.assertEqual(result, "")

    @patch("minimate.agent.agent.llm.chat_tools")
    def test_forced_final_answer_at_max_steps(self, mock_chat_tools):
        """步数耗尽后强制收尾：最后一次调用给出答案则返回，而不是 (无输出)"""
        mock_chat_tools.side_effect = [
            {"content": "", "tool_calls": [self._fc_call()]},
            {"content": "", "tool_calls": [self._fc_call()]},
            {"content": "最终答案", "tool_calls": []},
        ]
        agent = _make_agent(max_steps=2)
        result = agent.run("测试任务")

        self.assertEqual(result, "最终答案")
        self.assertEqual(mock_chat_tools.call_count, 3)

    @patch("minimate.agent.agent.llm.chat_tools")
    def test_direct_content_is_final(self, mock_chat_tools):
        """FC 模式无工具调用时，content 即最终答案"""
        mock_chat_tools.return_value = {"content": "这是一段直接回答", "tool_calls": []}
        agent = _make_agent()
        result = agent.run("测试任务")
        self.assertEqual(result, "这是一段直接回答")

    @patch("minimate.agent.agent.llm.chat_tools")
    def test_tool_calls_recorded_in_tool_memory(self, mock_chat_tools):
        """每次真实工具执行都写入工具调用记忆"""
        mock_chat_tools.side_effect = [
            {"content": "", "tool_calls": [self._fc_call()]},
            {"content": "最终答案", "tool_calls": []},
        ]
        agent = _make_agent()
        result = agent.run("测试任务")

        self.assertEqual(result, "最终答案")
        self.assertEqual(len(agent.tool_memory), 1)
        record = agent.tool_memory.records[0]
        self.assertEqual(record.tool_name, "fake_tool")
        self.assertTrue(record.ok)
        self.assertTrue(record.loop_id.startswith("loop-"))

    @patch("minimate.agent.agent.llm.chat_tools")
    def test_repeat_call_detected(self, mock_chat_tools):
        """相同工具相同参数连续调用 → 第二次起不执行，回灌重复提示"""
        calls = [self._fc_call()]
        mock_chat_tools.side_effect = [
            {"content": "", "tool_calls": calls},
            {"content": "", "tool_calls": calls},
            {"content": "最终答案", "tool_calls": []},
        ]
        agent = _make_agent()
        result = agent.run("测试任务")

        self.assertEqual(result, "最终答案")
        # 第二次调用的最后一条 tool 消息应为重复提示，而非真实执行结果
        second_messages = mock_chat_tools.call_args_list[1].args[0]
        last = second_messages[-1]
        self.assertEqual(last["role"], "tool")
        self.assertIn("[重复调用]", last["content"])
        self.assertNotIn("结果[q1]", last["content"])

    @patch("minimate.agent.agent.llm.chat")
    @patch("minimate.agent.agent.llm.chat_tools")
    def test_fc_fallback_to_text(self, mock_chat_tools, mock_chat):
        """FC 通道异常时降级到文本协议"""
        mock_chat_tools.side_effect = RuntimeError("tools 不支持")
        mock_chat.return_value = "Thought: 直接回答\nFinal Answer: 文本协议答案"
        agent = _make_agent()
        result = agent.run("测试任务")
        self.assertEqual(result, "文本协议答案")
        mock_chat.assert_called_once()

    @patch("minimate.agent.agent.llm.call")
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

    @patch("minimate.agent.agent.llm.call")
    def test_chat_mode_single_call(self, mock_call):
        """chat 模式：无论有无工具，都只调用一次 LLM"""
        mock_call.return_value = "直接答案"
        agent = _make_agent()  # 带工具
        result = agent.run("问题", mode="chat")
        self.assertEqual(result, "直接答案")
        mock_call.assert_called_once()

    @patch("minimate.agent.agent.llm.call")
    @patch("minimate.agent.agent.llm.chat")
    def test_plan_mode_executes_steps(self, mock_chat, mock_call):
        """plan 模式：生成计划 → 逐步执行 → 汇总（无工具 Agent 每步走单次调用）"""
        plan_json = '[{"step": "a", "goal": "步骤一", "detail": "做A"}, {"step": "b", "goal": "步骤二", "detail": "做B"}]'
        mock_call.side_effect = [plan_json, "步骤一结果", "步骤二结果", "最终汇总答案"]
        agent = _make_agent(with_tool=False)
        result = agent.run("复杂任务", mode="plan")

        self.assertEqual(result, "最终汇总答案")
        self.assertEqual(mock_call.call_count, 4)  # 计划 + 2 步骤 + 汇总
        mock_chat.assert_not_called()

    @patch("minimate.agent.agent.llm.chat_tools")
    @patch("minimate.agent.agent.llm.call")
    def test_plan_action_direct(self, mock_call, mock_chat_tools):
        """action 步骤直连工具，LLM 不参与（只计划 + 汇总两次调用）"""
        plan_json = json.dumps([{
            "id": "step1",
            "goal": "写入文件",
            "type": "action",
            "tool": "fake_tool",
            "args": {"args": "hello"},
            "depends_on": [],
        }], ensure_ascii=False)
        mock_call.side_effect = [plan_json, "最终答案"]
        agent = _make_agent()
        result = agent.run("任务", mode="plan")
        self.assertEqual(result, "最终答案")
        mock_chat_tools.assert_not_called()  # action 直连，不进 ReAct
        self.assertEqual(mock_call.call_count, 2)  # 计划 + 汇总

    @patch("minimate.agent.agent.llm.chat_tools", return_value={"content": "降级处理", "tool_calls": []})
    @patch("minimate.agent.agent.llm.call")
    @patch("minimate.agent.agent.time.sleep")
    def test_action_retry_on_retryable(self, mock_sleep, mock_call, mock_chat_tools):
        """action 直连遇 [可重试] 错误：自动重试一次，而非立即降级 LLM"""
        calls = {"n": 0}

        def flaky(args):
            calls["n"] += 1
            return "[可重试] 网络错误"

        tools = ToolExecutor()
        tools.register(Tool(name="flaky", description="t", func=flaky))
        agent = Agent(role="测试", goal="g", tools=tools, max_steps=3)
        plan_json = json.dumps([{
            "id": "s1", "goal": "x", "type": "action",
            "tool": "flaky", "args": {"args": "a"},
        }], ensure_ascii=False)
        mock_call.side_effect = [plan_json, "汇总"]

        agent.run("任务", mode="plan")

        self.assertEqual(calls["n"], 2)  # 首试 + 自动重试
        mock_sleep.assert_called()       # 重试有间隔


class PlanDagTest(unittest.TestCase):
    """DAG 拓扑排序与参数引用"""

    def test_topo_sort_batches(self):
        tasks = [
            PlanTask("a", "A", depends_on=[]),
            PlanTask("b", "B", depends_on=["a"]),
            PlanTask("c", "C", depends_on=["a"]),
            PlanTask("d", "D", depends_on=["b", "c"]),
        ]
        batches = topo_sort(tasks)
        self.assertEqual([t.task_id for t in batches[0]], ["a"])
        self.assertEqual(sorted(t.task_id for t in batches[1]), ["b", "c"])
        self.assertEqual([t.task_id for t in batches[2]], ["d"])

    def test_topo_sort_cycle_raises(self):
        tasks = [
            PlanTask("a", "A", depends_on=["b"]),
            PlanTask("b", "B", depends_on=["a"]),
        ]
        with self.assertRaises(ValueError):
            topo_sort(tasks)

    def test_topo_sort_missing_dep_raises(self):
        tasks = [PlanTask("a", "A", depends_on=["nope"])]
        with self.assertRaises(ValueError):
            topo_sort(tasks)

    def test_resolve_args_ref(self):
        agent = _make_agent()
        resolved = agent._resolve_args(
            {"content": {"ref": "step1"}, "path": "x.txt"},
            {"step1": "结果X"},
        )
        self.assertEqual(resolved["content"], "结果X")
        self.assertEqual(resolved["path"], "x.txt")

    def test_resolve_args_inline_ref(self):
        """内联引用 {ref:step1} 与 {step1} 均被替换"""
        agent = _make_agent()
        resolved = agent._resolve_args(
            {"content": "标题\n{ref:step1}\n{step2}"},
            {"step1": "搜索A", "step2": "搜索B"},
        )
        self.assertEqual(resolved["content"], "标题\n搜索A\n搜索B")

    def test_resolve_args_missing_ref_marked(self):
        """缺失引用替换为可见标记，不静默写入错误内容"""
        agent = _make_agent()
        resolved = agent._resolve_args(
            {"content": "{ref:nope}"},
            {"step1": "X"},
        )
        self.assertIn("[未解析引用: nope]", resolved["content"])

    def test_infer_verify_command_py(self):
        """verify 缺命令时，从依赖步骤的 write_file 结果推断 python 执行命令"""
        agent = _make_agent()
        task = PlanTask("v", "验证", step_type="verify", depends_on=["a"])
        cmd = agent._infer_verify_command(task, {
            "a": "文件已写入：example/topological_sort.py（1517 字符）",
        })
        self.assertEqual(cmd, "python example/topological_sort.py")

    def test_infer_verify_command_no_file(self):
        agent = _make_agent()
        task = PlanTask("v", "验证", step_type="verify", depends_on=["a"])
        cmd = agent._infer_verify_command(task, {"a": "没有文件信息"})
        self.assertEqual(cmd, "")

    def test_invalid_mode_raises(self):
        agent = _make_agent()
        with self.assertRaises(ValueError):
            agent.run("任务", mode="unknown")


if __name__ == "__main__":
    unittest.main()
