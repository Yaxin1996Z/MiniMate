"""短期记忆 —— Token 预算分级管理

机制（对照 Java TokenBudget + ConversationMemory）：
  1. 添加条目后若总 token 超过硬预算 → 立即淘汰最旧（滑动窗口）
  2. 达到 trigger_ratio（默认 0.9）时由 Manager 触发 compress_old：
     保留最近 N 轮完整消息，旧条目交给压缩器生成摘要并注入（injectSummary）
"""

from .core import MemoryItem


class ShortTermMemory:
    def __init__(self, token_budget: int = 8000, trigger_ratio: float = 0.9):
        self.token_budget = token_budget
        self.trigger_ratio = trigger_ratio
        self._items: list[MemoryItem] = []     # 对话 / 工具结果（短期窗口）
        self._summaries: list[str] = []        # 淘汰后保留的压缩摘要
        self._current_tokens = 0

    # ----------------------------------------------------------
    # 写入与淘汰
    # ----------------------------------------------------------

    def add(self, item: MemoryItem) -> list[MemoryItem]:
        """添加条目；超硬预算时立即淘汰最旧（滑动窗口）"""
        self._items.append(item)
        self._current_tokens += item.tokens
        evicted: list[MemoryItem] = []
        while self._current_tokens > self.token_budget and len(self._items) > 1:
            evicted.append(self._items.pop(0))
            self._current_tokens -= evicted[-1].tokens
        return evicted

    # ----------------------------------------------------------
    # 压缩（由 Manager 在达到 trigger_ratio 时触发）
    # ----------------------------------------------------------

    def needs_compression(self) -> bool:
        """是否达到压缩触发阈值（占用率 >= trigger_ratio 且有足够旧条目）"""
        return (
            self._current_tokens >= self.token_budget * self.trigger_ratio
            and len(self._items) > 8
        )

    def compress_old(self, compressor, keep_recent_rounds: int = 3) -> bool:
        """压缩旧条目（保留最近 N 轮完整消息），注入摘要"""
        keep_count = keep_recent_rounds * 2
        if len(self._items) <= keep_count + 2:
            return False
        old = self._items[:-keep_count]
        recent = self._items[-keep_count:]

        summary = compressor.compress(old)
        if not summary:
            return False

        # 重建：摘要条目 + 近期完整消息（injectSummary 模式）
        summary_item = MemoryItem(type="summary", role="", content=summary)
        self._items = [summary_item]
        self._current_tokens = summary_item.tokens
        for r in recent:
            self._items.append(r)
            self._current_tokens += r.tokens

        self._summaries.append(summary)
        if len(self._summaries) > 3:
            self._summaries.pop(0)
        return True

    def get_token_count(self) -> int:
        return self._current_tokens

    # ----------------------------------------------------------
    # 读取
    # ----------------------------------------------------------

    def get_context(self, max_tokens: int = 6000) -> str:
        """按 token 预算返回短期上下文（最近优先）"""
        parts: list[str] = []
        used = 0
        for item in reversed(self._items):
            if parts and used + item.tokens > max_tokens:
                break
            parts.append(item.format())
            used += item.tokens
        return "\n".join(reversed(parts))

    def get_summaries(self, limit: int = 3) -> str:
        """返回压缩摘要（旧历史）"""
        if not self._summaries:
            return ""
        return "\n".join(f"- {s[:300]}" for s in self._summaries[-limit:])

    @property
    def items(self) -> list[MemoryItem]:
        return list(self._items)

    @property
    def summaries(self) -> list[str]:
        return list(self._summaries)

    def clear(self):
        self._items.clear()
        self._summaries.clear()
        self._current_tokens = 0
