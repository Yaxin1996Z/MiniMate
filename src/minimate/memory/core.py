"""Memory 核心：条目模型、记忆类型、Token 估算、统一抽象"""

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .tokenizer import tokenize


# 四种记忆类型
MEMORY_TYPES = ("conversation", "fact", "summary", "tool_result")

_TYPE_PREFIX = {
    "conversation": "conv",
    "fact": "fact",
    "summary": "summ",
    "tool_result": "tool",
}


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文按字计，英文按 4 字符/token"""
    if not text:
        return 0
    cn = sum(1 for c in text if ord(c) > 0x2E80)
    other = len(text) - cn
    return cn + other // 4 + 1


class Memory(ABC):
    """记忆统一抽象"""

    @abstractmethod
    def store(self, entry: "MemoryItem") -> None:
        """存储一条记忆"""

    @abstractmethod
    def retrieve(self, entry_id: str) -> "MemoryItem | None":
        """根据 ID 检索记忆"""

    @abstractmethod
    def search(self, query: str, limit: int) -> list["MemoryItem"]:
        """搜索相关记忆"""

    @abstractmethod
    def get_all(self, *args, **kwargs) -> list["MemoryItem"]:
        """获取所有记忆"""

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """删除指定记忆"""

    @abstractmethod
    def clear(self) -> None:
        """清空所有记忆"""

    @abstractmethod
    def get_token_count(self) -> int:
        """当前 token 总数"""

    @abstractmethod
    def size(self) -> int:
        """记忆条数"""


@dataclass
class MemoryItem:
    """记忆条目：ID + 类型 + 内容 + 时间 + token 数 + 关键词"""

    id: str = ""
    type: str = "conversation"      # conversation / fact / summary / tool_result
    role: str = ""                  # user / assistant / tool
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.id:
            self.id = f"{_TYPE_PREFIX.get(self.type, 'mem')}-{uuid.uuid4().hex[:8]}"
        if not self.tokens:
            self.tokens = estimate_tokens(self.content)

    def format(self) -> str:
        prefix = self.role or self.type
        return f"{prefix}: {self.content}"


def score_relevance(item: MemoryItem, query: str) -> float:
    """相关度评分：精确匹配 1.0 + 关键词命中比例 × 时间衰减（24h 内 1.0→0.5）

    中文按连续双字切分（见 tokenizer），避免贪心整段导致"什么"这类词匹配不上。
    """
    if not query:
        return 0.0
    content = (item.content or "").lower()
    q = query.strip().lower()
    if q and q in content:
        return 1.0

    q_words = tokenize(q)
    if not q_words:
        return 0.0
    matched = sum(1 for w in q_words if w in content)
    if matched == 0:
        return 0.0
    keyword_score = matched / len(q_words)

    age_hours = (time.time() - item.timestamp) / 3600.0
    time_decay = max(0.5, 1.0 - age_hours / 24.0)
    return keyword_score * time_decay
