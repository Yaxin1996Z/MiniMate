"""Map-Reduce 上下文压缩器

策略：
  Map   - 把待压缩内容按字符分片，每片调用 LLM 生成局部摘要
  Reduce- 把各片摘要合并，再调用 LLM 生成综合摘要
保留最近 N 轮完整消息不参与压缩（keep_recent_rounds）。
"""

from .. import llm

from .core import MemoryItem


class MapReduceCompressor:
    def __init__(self, chunk_chars: int = 1200, keep_recent_rounds: int = 3):
        self.chunk_chars = chunk_chars
        self.keep_recent_rounds = keep_recent_rounds

    def compress(self, items: list[MemoryItem]) -> str:
        """把条目列表压缩为摘要；保留最近 keep_recent_rounds 轮完整消息"""
        if not items:
            return ""

        # 保留区：最近 N 轮（每轮按 2 条计：user + assistant）
        keep_count = self.keep_recent_rounds * 2
        compress_zone = items[:-keep_count] if len(items) > keep_count else []
        if not compress_zone:
            return ""

        text = "\n".join(i.format() for i in compress_zone)
        return self._map_reduce(text)

    def _map_reduce(self, text: str) -> str:
        # Map：分片摘要
        chunks = [
            text[i : i + self.chunk_chars]
            for i in range(0, len(text), self.chunk_chars)
        ]
        partials = [self._summarize_chunk(c) for c in chunks]
        # 过滤失败结果；失败的片降级为截取原文前 200 字
        partials = [
            p if p and not p.startswith("[API 错误]") else c[:200]
            for p, c in zip(partials, chunks)
        ]
        if not partials:
            return ""

        # Reduce：合并
        if len(partials) == 1:
            return partials[0]
        combined = "\n".join(f"- {p}" for p in partials)
        system = "你是一个高效的信息压缩器，摘要必须保留关键事实、结论与用户偏好。"
        final = llm.call(
            f"将以下多个摘要合并为一个不超过 200 字的综合摘要（保留所有关键信息）：\n\n{combined}",
            system,
            temperature=0.1,
        )
        if not final or final.startswith("[API 错误]"):
            # 降级：直接拼接各片摘要
            return "；".join(partials)[:1000]
        return final.strip()

    @staticmethod
    def _summarize_chunk(chunk: str) -> str:
        system = "你是一个高效的信息压缩器，摘要必须保留关键事实、结论与用户偏好。"
        return llm.call(
            f"将以下对话/记录压缩为不超过 80 字的要点摘要：\n\n{chunk}",
            system,
            temperature=0.1,
        ).strip()
