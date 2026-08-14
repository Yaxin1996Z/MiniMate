"""Agent 角色枚举 —— 三个专职角色，各司其职"""

from __future__ import annotations

from enum import Enum


class AgentRole(Enum):
    """子 Agent 角色：规划者拆任务、执行者干活、检查者找茬"""

    PLANNER = ("planner", "规划者", "负责分析用户任务，制定执行计划，将复杂任务拆解为可执行的子任务")
    WORKER = ("worker", "执行者", "负责执行具体任务步骤，调用工具完成文件操作、命令执行等操作")
    REVIEWER = ("reviewer", "检查者", "负责检查执行结果的质量和正确性，提供改进建议")

    def __init__(self, code: str, display: str, description: str):
        self.code = code
        self.display = display
        self.description = description

    @classmethod
    def by_code(cls, code: str) -> "AgentRole":
        for role in cls:
            if role.code == code:
                return role
        raise ValueError(f"未知角色：{code}")

    def should_use_tools(self) -> bool:
        """只有执行者才允许调用工具，规划者/检查者只做分析与判断"""
        return self is AgentRole.WORKER
