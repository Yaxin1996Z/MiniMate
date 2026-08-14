"""Agent 通信消息 —— 六种消息类型，主从模式下全部经编排器路由"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .role import AgentRole


class AgentMessageType(Enum):
    """消息类型：TASK 是向下分发的指令，其余是向上返回的状态/结果"""

    TASK = "TASK"            # 编排器 → 子 Agent：派发任务
    RESULT = "RESULT"        # 子 Agent → 编排器：返回结果/计划/审查结论
    FEEDBACK = "FEEDBACK"    # 检查者 → 编排器 → 执行者：审查不通过的改进建议
    APPROVAL = "APPROVAL"    # 检查者 → 编排器：审查通过
    REJECTION = "REJECTION"  # 检查者 → 编排器：审查不通过，触发重试
    ERROR = "ERROR"          # 任意 Agent → 编排器：执行异常


@dataclass
class AgentMessage:
    """子 Agent 间通信消息"""

    from_agent: str
    from_role: AgentRole
    content: str
    type: AgentMessageType

    def __repr__(self) -> str:
        return f"<AgentMessage {self.type.value} from={self.from_agent} ({self.from_role.display})>"
