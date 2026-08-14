"""LLM 用量统计（进程级累计）"""

_stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def get_stats() -> dict:
    """返回累计 Token 统计（调用次数 / prompt / completion）"""
    return dict(_stats)


def reset_stats() -> None:
    """重置 Token 统计"""
    _stats["calls"] = 0
    _stats["prompt_tokens"] = 0
    _stats["completion_tokens"] = 0


def _record_usage(resp) -> None:
    """记录一次响应的 token 用量"""
    global _stats
    _stats["calls"] += 1
    usage = getattr(resp, "usage", None)
    if usage is not None:
        _stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
