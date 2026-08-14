"""多 Agent 包 —— 主从架构 + 三角色分工（规划者 / 执行者 / 检查者）

设计参考 paicoding Multi-Agent：
  - AgentRole      角色枚举（PLANNER / WORKER / REVIEWER）
  - AgentMessage   子 Agent 间通信消息（六种类型，全部经编排器路由）
  - ExecutionStep  计划步骤建模（依赖 DAG + 状态机）
  - SubAgent       轻量子 Agent（独立角色/提示词，共享工具与记忆）
  - MultiAgentOrchestrator 编排器：规划→解析→执行→审查→残留→汇总
"""

from .role import AgentRole
from .message import AgentMessage, AgentMessageType
from .sub_agent import SubAgent
from .orchestrator import ExecutionStep, StepStatus, MultiAgentOrchestrator

__all__ = [
    "AgentRole",
    "AgentMessage",
    "AgentMessageType",
    "ExecutionStep",
    "StepStatus",
    "SubAgent",
    "MultiAgentOrchestrator",
]
