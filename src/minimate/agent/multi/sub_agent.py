"""SubAgent —— 轻量子 Agent 实现

每个子 Agent 有独立的角色、系统提示词和对话历史；共享 LLM 客户端、
工具注册表和记忆管理器（避免每个子 Agent 都重新初始化一份）。

角色分工：
  - 规划者（PLANNER）  ：纯 LLM 分析，输出 JSON 执行计划，不调用工具
  - 执行者（WORKER）   ：唯一允许调用工具的角色，走 ReAct 循环
  - 检查者（REVIEWER） ：纯 LLM 审查，输出 JSON 审批结果，不调用工具
"""

from __future__ import annotations

from typing import Optional

from ... import llm
from ...tools import ToolExecutor, truncate
from ...memory import MemoryManager
from ..agent import Agent, _section
from .role import AgentRole
from .message import AgentMessage, AgentMessageType


PLANNER_PROMPT = (
    "你是一个任务规划专家。你的职责是分析用户的需求，将其拆解为清晰的执行步骤。\n"
    "请按以下 JSON 格式输出执行计划（不要输出任何其他内容）：\n"
    '{\n'
    '  "summary": "任务摘要",\n'
    '  "steps": [\n'
    '    {\n'
    '      "id": "step_1",\n'
    '      "description": "步骤描述，要具体明确",\n'
    '      "type": "FILE_READ | FILE_WRITE | COMMAND | ANALYSIS | VERIFICATION",\n'
    '      "dependencies": []\n'
    '    }\n'
    '  ]\n'
    '}\n'
    "要求：简单任务拆 1-3 步，复杂任务拆 5-10 步；\n"
    "type 必须是 FILE_READ / FILE_WRITE / COMMAND / ANALYSIS / VERIFICATION 之一；\n"
    "dependencies 填写依赖的步骤 id（无依赖则为空数组）。"
)

REVIEWER_PROMPT = (
    "你是一个质量检查专家。你的职责是检查执行结果是否正确、完整和高质量。\n"
    "请以 JSON 格式输出检查结果（不要输出任何其他内容）：\n"
    '{\n'
    '  "approved": true,\n'
    '  "summary": "检查摘要",\n'
    '  "issues": ["问题1", "问题2"],\n'
    '  "suggestions": ["建议1", "建议2"]\n'
    '}\n'
    "approved 为 true 表示放行；为 false 表示打回重做，并必须给出具体的 issues 和 suggestions。"
)

WORKER_BACKSTORY = (
    "你是一个任务执行专家。你的职责是根据给定的任务步骤，调用工具完成具体操作。\n"
    "可用工具见工具列表，使用优先级：\n"
    "1. 涉及理解代码库时，优先使用 search_code 工具检索；\n"
    "2. 文件读写用 read_file / write_file / list_files；\n"
    "3. 执行命令用 run_shell；\n"
    "4. 不要上来就用命令扫描文件系统。"
)


class SubAgent:
    """轻量子 Agent：独立角色 + 系统提示词；共享 LLM 客户端与工具注册表"""

    def __init__(
        self,
        name: str,
        role: AgentRole,
        tools: Optional[ToolExecutor] = None,
        memory: Optional[MemoryManager] = None,
        max_steps: int = 8,
    ):
        self.name = name
        self.role = role
        self.tools = tools
        self.memory = memory
        self.max_steps = max_steps
        self.history: list[dict] = []  # 独立对话历史（每步完成后清空）

        # 执行者复用 Agent 的 ReAct 循环（Function Calling + 文本协议降级）
        self._worker_agent: Optional[Agent] = None
        if role is AgentRole.WORKER:
            self._worker_agent = Agent(
                role=f"执行者（{name}）",
                goal="根据给定的任务步骤，调用工具完成具体操作。",
                backstory=WORKER_BACKSTORY,
                tools=tools,
                memory=memory,
                max_steps=max_steps,
            )

    # ----------------------------------------------------------
    # 角色行为
    # ----------------------------------------------------------

    def plan(self, task: str, context: str = "") -> AgentMessage:
        """规划者：把用户需求拆成 JSON 执行计划（纯 LLM，不调用工具）"""
        self._require_role(AgentRole.PLANNER)
        prompt = task
        if context:
            prompt += f"\n\n参考上下文：\n{truncate(context, 1500)}"
        content = llm.call(prompt, system=PLANNER_PROMPT, temperature=0.2)
        msg_type = (
            AgentMessageType.RESULT
            if content and not content.startswith("[API 错误]")
            else AgentMessageType.ERROR
        )
        return AgentMessage(self.name, self.role, content, msg_type)

    def review(self, task: str, result: str) -> AgentMessage:
        """检查者：审查执行结果，输出 JSON 审批（纯 LLM，不调用工具）"""
        self._require_role(AgentRole.REVIEWER)
        prompt = f"任务：\n{task}\n\n执行结果：\n{truncate(result, 3000)}"
        content = llm.call(prompt, system=REVIEWER_PROMPT, temperature=0.1)
        msg_type = (
            AgentMessageType.RESULT
            if content and not content.startswith("[API 错误]")
            else AgentMessageType.ERROR
        )
        return AgentMessage(self.name, self.role, content, msg_type)

    def execute(self, task: str, context: str = "") -> AgentMessage:
        """执行者：调用工具完成具体操作（唯一允许使用工具的角色）"""
        self._require_role(AgentRole.WORKER)
        if self._worker_agent is None:
            return AgentMessage(
                self.name, self.role, "执行者未初始化", AgentMessageType.ERROR
            )
        content = self._worker_agent.run(task, context=context, mode="react")
        msg_type = (
            AgentMessageType.RESULT
            if content and not content.startswith("[API 错误]")
            else AgentMessageType.ERROR
        )
        return AgentMessage(self.name, self.role, content, msg_type)

    def clear_history(self) -> None:
        """完成一个独立任务后清空对话历史（保留系统提示词）

        MiniMate 的 Agent.run 每次都会新建消息列表，天然满足"每步干净状态"，
        此处保留接口便于后续扩展持久对话历史。
        """
        self.history.clear()

    def _require_role(self, role: AgentRole):
        if self.role is not role:
            raise ValueError(
                f"{self.name} 是 {self.role.display}，不能执行 {role.display} 的任务"
            )
