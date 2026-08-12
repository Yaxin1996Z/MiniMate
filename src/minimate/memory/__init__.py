"""Memory 包 —— 四种记忆 + Token 预算 + 压缩 + 持久化

结构：
  core.py       - MemoryItem / 记忆类型 / Token 估算
  short_term.py - 短期记忆（预算淘汰 + 摘要保留）
  long_term.py  - 长期记忆（JSON 持久化 + 去重 + 检索）
  compressor.py - Map-Reduce 上下文压缩器
  manager.py    - MemoryManager（整合）与 ResearchMemory 兼容接口
"""

from .core import MemoryItem, MEMORY_TYPES, estimate_tokens, score_relevance
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .compressor import MapReduceCompressor
from .manager import MemoryManager, ResearchMemory

__all__ = [
    "MemoryItem",
    "MEMORY_TYPES",
    "estimate_tokens",
    "score_relevance",
    "ShortTermMemory",
    "LongTermMemory",
    "MapReduceCompressor",
    "MemoryManager",
    "ResearchMemory",
]
