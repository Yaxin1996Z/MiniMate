"""短期记忆 —— 对话历史 + 硬预算淘汰 + 压缩摘要

职责：
1. 维护对话历史（用户消息、助手回复、工具结果），按 id 索引
2. token 硬预算超限自动淘汰最旧（滑动窗口），只记录淘汰条数
3. 占用率达 trigger_ratio 时触发压缩：旧条目移出 _entries，
   生成的摘要单独保存在 _summaries（最多 3 条），保留最近 N 轮完整消息
"""

import threading

from .core import Memory, MemoryItem
from .tokenizer import matches, tokenize


class ShortTermMemory(Memory):
    def __init__(self, token_budget: int = 8000, trigger_ratio: float = 0.9):
        self.token_budget = token_budget
        self.trigger_ratio = trigger_ratio
        self._entries: dict[str, MemoryItem] = {}
        self._current_tokens = 0
        self._summaries: list[str] = []  # 压缩摘要文本（最多 3 条，供上下文按时间序拼装）
        self._evicted_count = 0          # 硬预算淘汰条数（只计数，不保留内容）
        self._lock = threading.RLock()   # 多 Agent 并行 Worker 共享时的线程安全

    # ----------------------------------------------------------
    # Memory 接口
    # ----------------------------------------------------------

    def store(self, entry: MemoryItem) -> None:
        with self._lock:
            self._entries[entry.id] = entry
            self._current_tokens += entry.tokens
            while self._current_tokens > self.token_budget and len(self._entries) > 1:
                self.evict_oldest()

    def retrieve(self, entry_id: str) -> MemoryItem | None:
        return self._entries.get(entry_id)

    def search(self, query: str, limit: int) -> list[MemoryItem]:
        tokens = tokenize(query)
        return [e for e in self._entries.values() if matches(e.content, tokens)][:limit]

    def get_all(self) -> list[MemoryItem]:
        return list(self._entries.values())

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            removed = self._entries.pop(entry_id, None)
            if removed is not None:
                self._current_tokens -= removed.tokens
                return True
            return False

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._current_tokens = 0
            self._summaries.clear()
            self._evicted_count = 0

    def get_token_count(self) -> int:
        return self._current_tokens

    def size(self) -> int:
        return len(self._entries)

    # ----------------------------------------------------------
    # 预算与淘汰
    # ----------------------------------------------------------

    def get_max_tokens(self) -> int:
        return self.token_budget

    def set_max_tokens(self, max_tokens: int):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.token_budget = max_tokens
        while self._current_tokens > self.token_budget and len(self._entries) > 1:
            self.evict_oldest()

    def evict_oldest(self) -> MemoryItem:
        with self._lock:
            oldest_key = next(iter(self._entries))
            oldest = self._entries.pop(oldest_key)
            self._current_tokens -= oldest.tokens
            self._evicted_count += 1
            return oldest

    def get_usage_ratio(self) -> float:
        return self._current_tokens / self.token_budget if self.token_budget else 0.0

    def get_status_summary(self) -> str:
        return (
            f"短期记忆: {len(self._entries)}条 / {self._current_tokens} tokens "
            f"(预算: {self.token_budget}, 使用率: {self.get_usage_ratio() * 100:.0f}%, "
            f"已淘汰: {self._evicted_count}条)"
        )

    # ----------------------------------------------------------
    # 压缩（保留最近 N 轮完整消息，摘要单独存 _summaries）
    # ----------------------------------------------------------

    def needs_compression(self) -> bool:
        return (
            self._current_tokens >= self.token_budget * self.trigger_ratio
            and len(self._entries) > 8
        )

    def compress_old(
        self,
        compressor,
        keep_recent_rounds: int = 3,
        on_evict=None,
    ) -> bool:
        """压缩旧条目：从 _entries 移除旧条目，摘要存入 _summaries，保留最近 N 轮

        on_evict：可选回调，在旧条目被移除前收到待压缩列表（用于事实提取等）。
        """
        with self._lock:
            keep_count = keep_recent_rounds * 2
            entries = list(self._entries.values())
            if len(entries) <= keep_count + 2:
                return False
            old = entries[:-keep_count]
            recent = entries[-keep_count:]

            if on_evict:
                on_evict(old)

            summary = compressor.compress(old)
            if not summary:
                return False

            # 从 _entries 移除被压缩的旧条目（保留最近 N 轮）
            for o in old:
                self._entries.pop(o.id, None)
                self._current_tokens -= o.tokens

            # 摘要单独存文本（不进 _entries，保持条目时间序；最多 3 条）
            self._summaries.append(summary)
            if len(self._summaries) > 3:
                self._summaries.pop(0)
            return True

    # ----------------------------------------------------------
    # 读取
    # ----------------------------------------------------------

    def get_context(self, max_tokens: int = 6000) -> str:
        """按 token 预算返回最近对话（_entries 时间序；摘要由上层按序拼装）"""
        with self._lock:
            parts: list[str] = []
            used = 0
            for item in reversed(list(self._entries.values())):
                if parts and used + item.tokens > max_tokens:
                    break
                parts.append(item.format())
                used += item.tokens
            return "\n".join(reversed(parts))

    def get_summaries(self, limit: int = 3) -> str:
        """返回压缩摘要文本（旧历史，按时间序）"""
        if not self._summaries:
            return ""
        return "\n".join(f"- {s[:300]}" for s in self._summaries[-limit:])

    @property
    def items(self) -> list[MemoryItem]:
        return list(self._entries.values())

    @property
    def summaries(self) -> list[str]:
        return list(self._summaries)
