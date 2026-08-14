"""Agent 包 —— 单 Agent 与多 Agent 扩展基础

结构：
  agent.py             - Agent（chat / react / plan 三种模式 + 交互观测）
  task.py              - PlanTask / 计划解析（DAG 拓扑排序）
  history_compactor.py - LLM 消息历史压缩（请求层，与记忆无关）
  multi/               - 多 Agent 编排（规划者 / 执行者 / 检查者 三角色协作）
"""

from .agent import Agent
from .task import PlanTask, parse_plan, topo_sort
from .history_compactor import ConversationHistoryCompactor
from .multi import (
    AgentMessage,
    AgentMessageType,
    AgentRole,
    ExecutionStep,
    MultiAgentOrchestrator,
    StepStatus,
    SubAgent,
)

__all__ = [
    "Agent",
    "PlanTask",
    "parse_plan",
    "topo_sort",
    "ConversationHistoryCompactor",
    "AgentRole",
    "AgentMessage",
    "AgentMessageType",
    "ExecutionStep",
    "StepStatus",
    "SubAgent",
    "MultiAgentOrchestrator",
]
