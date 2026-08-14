"""编排引擎 —— 兼容入口

Agent / 计划解析等实际代码已迁移至 minimate.agent 包，
后续多 Agent 编排在 agent/ 目录下扩展。
"""

from .agent import (
    Agent,
    ConversationHistoryCompactor,
    PlanTask,
    parse_plan,
    topo_sort,
)

__all__ = [
    "Agent",
    "PlanTask",
    "parse_plan",
    "topo_sort",
    "ConversationHistoryCompactor",
]
