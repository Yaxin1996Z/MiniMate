"""
记忆管理 —— 基于之前的 HybridMemory，精简为核心能力

三层记忆：
  Entity   — 提取和跟踪关键实体（人名、术语、概念）
  Buffer   — 短期对话历史，自动丢弃最早的消息
  Summary  — 长对话自动压缩为摘要
"""

import re
import json
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================
# 消息模型
# ============================================================

@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def format(self) -> str:
        return f"{self.role}: {self.content}"


# ============================================================
# 环形缓冲区记忆
# ============================================================

class CircularBuffer:
    """保留最近 k 轮对话，自动丢弃最早的"""

    def __init__(self, k: int = 20):
        self.k = k
        self._messages: List[Message] = []

    def add(self, role: str, content: str):
        self._messages.append(Message(role, content))
        max_msgs = self.k * 2
        while len(self._messages) > max_msgs:
            self._messages.pop(0)

    def get_context(self) -> str:
        return "\n".join(m.format() for m in self._messages)

    @property
    def count(self) -> int:
        return len(self._messages)

    def clear(self):
        self._messages.clear()


# ============================================================
# 实体记忆
# ============================================================

class EntityMemory:
    """从对话中提取关键实体信息"""

    PATTERNS = {
        "name": [r"我叫(.{1,8}?)(?:[，。、,.!！]|$)"],
        "topic": [r"研究(.{1,30}?)(?:[，。、,.!！]|$)"],
    }

    def __init__(self, llm_extract: Optional[Callable] = None):
        self._entities: Dict[str, str] = {}
        self._llm_extract = llm_extract

    def extract(self, text: str):
        for key, patterns in self.PATTERNS.items():
            if key in self._entities:
                continue
            for pattern in patterns:
                m = re.search(pattern, text)
                if m:
                    self._entities[key] = m.group(1).strip()
                    break

    def get_context(self) -> str:
        if not self._entities:
            return ""
        parts = [f"{k}={v}" for k, v in self._entities.items()]
        return f"[已知信息] {'; '.join(parts)}"

    def get_all(self) -> dict:
        return self._entities.copy()

    def clear(self):
        self._entities.clear()


# ============================================================
# 混合记忆（整合）
# ============================================================

class ResearchMemory:
    """研究助手专用的混合记忆"""

    def __init__(self):
        self.buffer = CircularBuffer(k=20)
        self.entity = EntityMemory()
        self._findings: List[str] = []  # 研究中发现的关键事实

    def add_user_message(self, content: str):
        self.buffer.add("user", content)
        self.entity.extract(content)

    def add_ai_message(self, content: str):
        self.buffer.add("assistant", content)

    def add_finding(self, finding: str):
        """记录研究发现的关键事实"""
        self._findings.append(finding)
        self.entity.extract(finding)

    def get_context(self) -> str:
        parts = []

        entity_ctx = self.entity.get_context()
        if entity_ctx:
            parts.append(entity_ctx)

        if self._findings:
            parts.append("[研究发现]")
            parts.extend(f"  - {f}" for f in self._findings[-5:])

        history = self.buffer.get_context()
        if history:
            parts.append("[对话历史]")
            parts.append(history)

        return "\n".join(parts)

    def get_report_context(self) -> str:
        """生成报告用的上下文（不带对话历史，只带实体和发现）"""
        parts = []
        entity_ctx = self.entity.get_context()
        if entity_ctx:
            parts.append(entity_ctx)
        if self._findings:
            parts.append("研究发现：")
            parts.extend(f"- {f}" for f in self._findings)
        return "\n".join(parts)

    def clear(self):
        self.buffer.clear()
        self.entity.clear()
        self._findings.clear()
