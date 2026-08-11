"""
编排引擎 —— Agent / Task / Crew 核心

设计：
  Agent  = 一个 LLM 角色 + 工具
  Task   = 一个任务单元（描述 + 分配给谁）
  Crew   = 任务调度器（支持 sequential / hierarchical）
"""

import re
import json
import os
from typing import Optional

from . import llm
from .tools import ToolExecutor, parse_react, truncate
from .memory import ResearchMemory
from .colors import color
from .logging import logger


# ============================================================
# 交互过程观测
# ============================================================

_LINE = "-" * 52


def _section(title: str, content: str = "", fg: str = "cyan") -> str:
    """生成带彩色分割线的过程分节，便于观测 Agent 交互

    fg 颜色约定：用户输入=blue, AI思考=yellow, 工具调用/返回=cyan/magenta,
    AI回答=green, 计划=magenta
    """
    head = f"{color.paint(_LINE, fg)}\n{color.paint(f'[{title}]', fg, bold=True)}"
    if content:
        return f"{head}\n{content}\n{color.paint(_LINE, fg)}"
    return f"{head}\n{color.paint(_LINE, fg)}"


# ============================================================
# Agent
# ============================================================

class Agent:
    """Agent = 角色设定 + 工具 + 记忆访问"""

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str = "",
        tools: Optional[ToolExecutor] = None,
        memory: Optional[ResearchMemory] = None,
        max_steps: int = 8,
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools
        self.memory = memory
        self.max_steps = max_steps

    def build_system_prompt(
        self,
        include_tools: bool = True,
        include_protocol: bool = True,
    ) -> str:
        prompt = f"你是{self.role}。\n目标：{self.goal}"
        if self.backstory:
            prompt += f"\n\n{self.backstory}"
        # 注入运行环境，帮助模型选择适配的命令/路径
        env = "Windows（cmd）" if os.name == "nt" else "Linux/macOS（bash）"
        prompt += f"\n\n运行环境：{env}"
        if include_tools and self.tools:
            prompt += f"\n\n{self.tools.get_tools_prompt(protocol=include_protocol)}"
        return prompt

    def run(self, task: str, context: str = "", mode: str = "react") -> str:
        """执行任务，mode 可选：chat（纯问答）/ react（ReAct 循环）/ plan（Plan & Execute）"""
        if mode not in ("chat", "react", "plan"):
            raise ValueError(f"未知模式 '{mode}'，可选：chat / react / plan")
        if mode == "chat":
            return self._run_chat(task, context)
        if mode == "plan":
            return self._run_plan(task, context)
        return self._run_react(task, context)

    def _run_chat(self, task: str, context: str = "") -> str:
        """纯问答：只调用一次 LLM 就返回答案"""
        system = self.build_system_prompt(include_tools=False)
        user_msg = self._build_user_message(task, context)
        print(_section("用户输入", self._display_user_input(task, context), fg="blue"))
        print(_section("助手思考", f"单次调用 LLM，直接作答（{self.role}）", fg="yellow"))
        result = llm.call(user_msg, system)
        print(_section("助手回答", result, fg="green"))
        return result

    def _run_react(self, task: str, context: str = "") -> str:
        """ReAct 循环：Thought → Action → Observation → Final Answer"""
        system = self.build_system_prompt()
        user_msg = self._build_user_message(task, context)
        print(_section("用户输入", self._display_user_input(task, context), fg="blue"))

        # 无工具：退化为单次调用
        if not self.tools or not self.tools.tool_list:
            print(_section("助手思考", f"无可用工具，退化为单次调用（{self.role}）", fg="yellow"))
            result = llm.call(user_msg, system)
            print(_section("助手回答", result, fg="green"))
            return result

        # 有工具：推理链逐轮累积到 messages
        print(_section("助手思考", f"进入 ReAct 循环，最多 {self.max_steps} 步", fg="yellow"))

        # 双通道：优先 Function Calling，异常时降级到文本协议
        recent_calls: list[tuple[str, str]] = []  # (工具名, 归一化参数)，用于重复调用检测
        try:
            # FC 通道：system 不含文本协议格式说明，避免模型输出 Final Answer 前缀
            fc_system = self.build_system_prompt(include_tools=True, include_protocol=False)
            messages_fc = []
            if fc_system:
                messages_fc.append({"role": "system", "content": fc_system})
            messages_fc.append({"role": "user", "content": user_msg})
            return self._run_react_fc(messages_fc, recent_calls)
        except Exception as e:
            logger.warning("Function Calling 降级到文本协议：%s", e)
            print(_section("助手思考", f"Function Calling 不可用（{e}），降级到文本协议", fg="yellow"))
            messages_text = []
            if system:
                messages_text.append({"role": "system", "content": system})
            messages_text.append({"role": "user", "content": user_msg})
            return self._run_react_text(messages_text, recent_calls)

    def _run_react_fc(self, messages: list[dict], recent_calls: list[tuple[str, str]]) -> str:
        """Function Calling 通道：模型原生输出 tool_calls，零正则解析"""
        schemas = self.tools.schemas
        print(_section("助手思考", f"Function Calling 通道（{len(schemas)} 个工具）", fg="yellow"))
        last_content = ""

        for step in range(1, self.max_steps + 1):
            result = llm.chat_tools(messages, tools=schemas)
            tool_calls = result["tool_calls"]

            if tool_calls:
                for tc in tool_calls:
                    name = tc["name"]
                    try:
                        args = json.loads(tc["arguments"] or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    print(_section(
                        f"工具调用（第 {step} 步）",
                        f"工具：{name}\n参数：{json.dumps(args, ensure_ascii=False)}",
                        fg="cyan",
                    ))
                    repeat_hint = self._check_repeat(name, args, recent_calls)
                    if repeat_hint:
                        observation = repeat_hint
                    else:
                        observation = self.tools.execute(name, **args)
                    print(_section("工具返回", truncate(observation, 1000), fg="magenta"))

                    # 消息回填：assistant 必须带原始 tool_calls，结果以 tool 角色返回
                    messages.append({
                        "role": "assistant",
                        "content": result["content"] or "",
                        "tool_calls": [{
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": observation,
                    })
                continue

            # 无工具调用：content 即最终回答
            content = (result["content"] or "").strip()
            if content:
                print(_section(f"助手回答（第 {step} 步）", content, fg="green"))
                return content
            last_content = content

        print(_section(
            "助手回答",
            f"（达到最大步数 {self.max_steps}，返回最后输出）\n{last_content or '(无输出)'}",
            fg="green",
        ))
        logger.warning("ReAct 达到最大步数 %d 未完成（FC 通道）", self.max_steps)
        return last_content

    def _run_react_text(self, messages: list[dict], recent_calls: list[tuple[str, str]]) -> str:
        """文本协议通道（fallback）：Thought/Action/Observation + 正则解析"""
        last_output = ""
        for step in range(1, self.max_steps + 1):
            last_output = llm.chat(messages)
            parsed = parse_react(last_output)

            # 1) 模型判定任务完成
            if parsed["final_answer"]:
                print(_section(f"助手回答（第 {step} 步）", parsed["final_answer"].strip(), fg="green"))
                return parsed["final_answer"].strip()

            # 2) 模型请求调用工具
            if parsed["action"]:
                thought = parsed["thought"] or "无思考过程"
                print(_section(f"助手思考（第 {step} 步）", thought, fg="yellow"))
                action_info = f"工具：{parsed['action']}"
                if parsed["action_input"]:
                    action_info += f"\n参数：{parsed['action_input']}"
                print(_section("工具调用", action_info, fg="cyan"))
                repeat_hint = self._check_repeat(
                    parsed["action"], parsed["action_input"], recent_calls
                )
                if repeat_hint:
                    observation = repeat_hint
                else:
                    observation = self.tools.execute_text(
                        parsed["action"], parsed["action_input"]
                    )
                print(_section("工具返回", truncate(observation, 1000), fg="magenta"))
                messages.append({"role": "assistant", "content": last_output})
                messages.append(
                    {"role": "user", "content": f"Observation:\n{truncate(observation)}"}
                )
                continue

            # 3) 既无结束标记也无工具调用：按最终答案容错处理
            print(_section(
                "助手回答",
                f"（未识别到结束标记，按最终答案处理）\n{last_output.strip()}",
                fg="green",
            ))
            return last_output.strip()

        # 达到最大步数仍未完成：返回最后输出，并明确告知
        print(_section(
            "助手回答",
            f"（达到最大步数 {self.max_steps}，返回最后输出）\n{last_output.strip()}",
            fg="green",
        ))
        logger.warning("ReAct 达到最大步数 %d 未完成（文本协议）", self.max_steps)
        final = parse_react(last_output)["final_answer"]
        return (final or last_output).strip()

    # ----------------------------------------------------------
    # 模式三：Plan & Execute
    # ----------------------------------------------------------

    def _run_plan(self, task: str, context: str = "") -> str:
        """Plan & Execute：生成结构化计划 → 逐步执行 → 汇总最终答案"""

        print(_section("Plan & Execute", f"任务：{task}", fg="magenta"))

        # 1) 生成计划（结构化 JSON）
        plan = self._make_plan(task, context)
        if not plan:
            logger.info("Plan&Execute 计划解析失败，回退到 ReAct：%s", task[:80])
            print(_section("助手思考", "计划解析失败，回退到 ReAct", fg="yellow"))
            return self._run_react(task, context)

        plan_summary = "\n".join(
            f"  {i}. {step.get('goal') or step.get('step') or f'步骤 {i}'}"
            for i, step in enumerate(plan, 1)
        )
        print(_section(f"执行计划（共 {len(plan)} 步）", plan_summary, fg="magenta"))

        # 2) 逐步执行，前序结果作为下一步上下文
        results: list[str] = []
        plan_text = json.dumps(plan, ensure_ascii=False, indent=2)
        for i, step in enumerate(plan, 1):
            step_goal = step.get("goal") or step.get("step") or f"步骤 {i}"
            step_detail = step.get("detail") or ""
            step_text = f"执行计划步骤 {i}/{len(plan)}：{step_goal}"
            if step_detail:
                step_text += f"\n具体要求：{step_detail}"

            step_ctx = f"【执行计划】\n{plan_text}"
            if results:
                step_ctx += "\n\n【已完成步骤】\n" + "\n\n".join(
                    f"步骤 {j}: {truncate(r, 800)}" for j, r in enumerate(results, 1)
                )

            if self.tools and self.tools.tool_list:
                output = self._run_react(step_text, step_ctx)
            else:
                output = self._run_chat(step_text, step_ctx)
            results.append(output)

        # 3) 汇总各步骤结果为最终答案
        final = self._summarize(task, context, results)
        print(_section("助手回答", final, fg="green"))
        return final

    def _make_plan(self, task: str, context: str = "") -> list[dict]:
        """让 LLM 输出结构化 JSON 计划（步骤数组）"""
        system = (
            "你是任务规划专家，擅长把模糊任务分解为清晰、可执行、顺序合理的步骤。"
        )
        prompt = (
            f"为以下任务制定分步执行计划：\n任务：{task}\n\n"
            "输出 JSON 数组（不要输出任何其他内容），每个元素包含：\n"
            "  - step: 步骤短名称\n"
            "  - goal: 该步骤目标（一句话）\n"
            "  - detail: 具体执行内容（1-2 句话）\n"
            "要求：2-5 步，步骤间有依赖顺序，覆盖任务全部要求。"
        )
        if context:
            prompt += f"\n\n参考上下文：\n{truncate(context, 1500)}"
        raw = llm.call(prompt, system, temperature=0.2)
        return parse_plan(raw)

    def _summarize(self, task: str, context: str, results: list[str]) -> str:
        """汇总各步骤结果，生成最终完整答案"""
        system = self.build_system_prompt(include_tools=False)
        body = "\n\n".join(
            f"【步骤 {i}】\n{truncate(r, 2000)}" for i, r in enumerate(results, 1)
        )
        prompt = (
            f"以下是按计划执行「{task}」各步骤得到的结果：\n\n{body}\n\n"
            "请基于这些结果，输出任务的最终完整答案"
            "（结构清晰、直接可用，不要复述执行过程）。"
        )
        return llm.call(prompt, system)

    # ----------------------------------------------------------
    # 公共辅助
    # ----------------------------------------------------------

    def _build_user_message(self, task: str, context: str = "") -> str:
        """组装用户消息：任务 + 上下文 + 记忆"""
        user_msg = f"当前任务：{task}"
        if context:
            user_msg += f"\n\n可以参考以下上下文：\n{context}"
        if self.memory:
            mem_ctx = self.memory.get_context()
            if mem_ctx:
                user_msg = f"{mem_ctx}\n\n{user_msg}"
        return user_msg

    def _display_user_input(self, task: str, context: str = "") -> str:
        """观测显示用：只展示当前任务，上下文/记忆用摘要代替，避免刷屏

        注意：这只是观测层展示，发送给 LLM 的完整上下文不受影响。
        """
        text = f"当前任务：{task}"
        if context:
            text += f"\n（参考上下文 {len(context)} 字符，未展开显示）"
        if self.memory and self.memory.get_context():
            text += "\n（携带会话记忆，未展开显示）"
        return text

    def _check_repeat(
        self,
        tool_name: str,
        args: dict | str,
        recent_calls: list[tuple[str, str]],
    ) -> str | None:
        """重复调用检测：相同 (工具, 参数) 连续尝试 >= 2 次时返回提示并阻止执行

        args 支持 dict（FC 通道）或字符串（文本协议通道）。
        命中时返回提示文本（作为 Observation 回灌给模型），未命中返回 None。
        """
        if isinstance(args, dict):
            key = (tool_name, json.dumps(args, sort_keys=True, ensure_ascii=False))
        else:
            key = (tool_name, str(args or "").strip())
        recent_calls.append(key)

        count = 0
        for k in reversed(recent_calls):
            if k == key:
                count += 1
            else:
                break
        if count >= 2:
            logger.warning(
                "重复调用拦截 tool=%s args=%s count=%d",
                tool_name, key[1][:120], count,
            )
            return (
                f"[重复调用] 工具 {tool_name} 使用相同参数已连续尝试 {count} 次未成功。"
                "请勿重复此调用，改用其他参数或工具，或直接向用户说明无法完成。"
            )
        return None


# ============================================================
# 计划解析
# ============================================================

def parse_plan(text: str) -> list[dict]:
    """解析 LLM 输出的计划 JSON 数组，兼容 ```json 代码块包裹和前后杂文本"""
    if not text:
        return []

    # 去掉 ```json ... ``` 代码块包裹
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)

    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []

    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return data
    return []


# ============================================================
# Task
# ============================================================

class Task:
    """一个任务单元"""

    def __init__(
        self,
        description: str,
        agent: Agent,
        expected_output: str = "文本",
        mode: str = "react",
    ):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.mode = mode
        self.output: Optional[str] = None

    def execute(self, context: str = "", mode: str = "") -> str:
        mode = mode or self.mode
        self.output = self.agent.run(self.description, context, mode=mode)
        return self.output


# ============================================================
# Crew
# ============================================================

class Crew:
    """任务调度器"""

    def __init__(self, agents: list[Agent], tasks: list[Task], memory=None):
        self.agents = agents
        self.tasks = tasks
        self.memory = memory
        self._context = ""

    def kickoff(self) -> str:
        """顺序执行所有任务，前一个输出传给后一个，但规划结果一直保留"""
        print(f"\n  \U0001f680 启动 {len(self.agents)} 个 Agent，{len(self.tasks)} 个任务\n")

        for agent in self.agents:
            print(f"    \U0001f464 {agent.role}")
        print()

        plan_text = ""   # 规划结果，全局保留
        final = ""
        for i, task in enumerate(self.tasks):
            print(f"  \U0001f4cb 任务 {i+1}: {task.description[:50]}...")

            # 组装上下文：规划（如有）+ 前一个 Agent 的输出
            full_context = ""
            if plan_text:
                full_context += f"【执行计划】\n{plan_text}\n\n"
            if self._context:
                full_context += f"【已有成果】\n{self._context}"

            output = task.execute(context=full_context)
            print(f"  ✅ 完成 ({len(output)} 字)")

            # 第一个 Agent（Planner）的输出单独保留
            if i == 0:
                plan_text = output

            # 任务完成后，将关键成果提取到记忆中
            if self.memory and i >= 1:
                self._extract_findings(output)

            self._context = output   # 只保留上一个 Agent 的输出（精简）
            final = output

        return final

    def _extract_findings(self, text: str):
        """从执行结果中提取关键发现，存入记忆"""
        # 用 LLM 提取 3-5 条关键发现
        from . import llm
        prompt = (
            "从以下执行结果中提取 3-5 条最关键的发现，每条一句话，用 - 开头：\n\n"
            f"{text[:3000]}\n\n关键发现："
        )
        result = llm.call(prompt, "你是一个结构化提取专家。")
        count = 0
        for line in result.split("\n"):
            line = line.strip().strip("- *").strip()
            if line and len(line) > 5:
                self.memory.add_finding(line)
                count += 1
        print(f"    [记忆] 提取 {count} 条关键发现")
