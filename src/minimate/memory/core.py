"""Memory 核心：条目模型、记忆类型、Token 估算"""

import time
from dataclasses import dataclass, field


# 四种记忆类型
MEMORY_TYPES = ("conversation", "fact", "summary", "tool_result")


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文按字计，英文按 4 字符/token"""
    if not text:
        return 0
    cn = sum(1 for c in text if ord(c) > 0x2E80)
    other = len(text) - cn
    return cn + other // 4 + 1


@dataclass
class MemoryItem:
    """记忆条目：类型 + 内容 + 时间 + token 数 + 关键词"""

    type: str = "conversation"      # conversation / fact / summary / tool_result
    role: str = ""                  # user / assistant / tool
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    tokens: int = 0
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if not self.tokens:
            self.tokens = estimate_tokens(self.content)

    def format(self) -> str:
        prefix = self.role or self.type
        return f"{prefix}: {self.content}"


def score_relevance(item: "MemoryItem", query: str) -> float:
    """相关度评分：精确匹配 1.0 + 关键词命中比例 × 时间衰减（24h 内 1.0→0.5）"""
    if not query:
        return 0.0
    content = item.content or ""
    q = query.strip()
    if q and q in content:
        return 1.0

    import re

    q_words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,6}", q)
    if not q_words:
        return 0.0
    matched = sum(1 for w in q_words if w in content)
    if matched == 0:
        return 0.0
    keyword_score = matched / len(q_words)

    import time
    age_hours = (time.time() - item.timestamp) / 3600.0
    time_decay = max(0.5, 1.0 - age_hours / 24.0)
    return keyword_score * time_decay
