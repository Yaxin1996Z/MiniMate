"""Memory 包

结构：
  core.py             - MemoryItem（id/type/content/metadata） + Memory 接口 + Token 估算 + 相关度
  tokenizer.py        - 检索分词（英文整词 + 中文双字）
  short_term.py       - 短期记忆（id 索引 + 硬预算淘汰 + 摘要注入）
  long_term.py        - 长期记忆（SQLite + 归一化去重 + 项目作用域）
  tool_memory.py      - 工具调用记忆（会话内存，循环内重复检测）
  compressor.py       - Map-Reduce 上下文压缩器
  token_budget.py     - Token 预算 + ContextProfile
  manager.py          - MemoryManager 门面
"""

from .core import (
    Memory,
    MemoryItem,
    MEMORY_TYPES,
    estimate_tokens,
    score_relevance,
)
from .tokenizer import matches, tokenize
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .tool_memory import ToolCallRecord, ToolMemory
from .compressor import MapReduceCompressor
from .token_budget import ContextProfile, TokenBudget
from .manager import MemoryManager

__all__ = [
    "Memory",
    "MemoryItem",
    "MEMORY_TYPES",
    "estimate_tokens",
    "score_relevance",
    "tokenize",
    "matches",
    "ShortTermMemory",
    "LongTermMemory",
    "ToolCallRecord",
    "ToolMemory",
    "MapReduceCompressor",
    "ContextProfile",
    "TokenBudget",
    "MemoryManager",
]
