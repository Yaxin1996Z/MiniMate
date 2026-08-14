"""LLM 调用封装 —— 统一管理大模型调用入口

react / plan 模式的推理调用、memory 的摘要压缩与事实提取，全部经由本包：
- client.py  客户端与模型配置（密钥 / BaseURL / 模型名）
- chat.py    对话调用（call / chat / chat_tools）
- stats.py   Token 用量统计（进程级累计）
"""

from .client import get_client, get_model
from .stats import _record_usage, get_stats, reset_stats
from .chat import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    call,
    chat,
    chat_tools,
)

__all__ = [
    "get_client",
    "get_model",
    "call",
    "chat",
    "chat_tools",
    "get_stats",
    "reset_stats",
    "_record_usage",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
]
