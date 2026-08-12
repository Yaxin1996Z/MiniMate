"""MemoryManager —— 整合四种记忆 + Token 预算 + 持久化

记忆分层：
  conversation - 短期对话（token 预算内，超限淘汰 + 摘要保留）
  fact         - 长期事实（用户偏好/项目信息，JSON 持久化 + 去重 + 检索）
  summary      - 压缩摘要（Map-Reduce 生成，跨窗口保留主线）
  tool_result  - 工具执行结果（短期）

上下文组装顺序：长期事实 → 历史摘要 → 短期对话（按预算截断）
"""

from .core import MemoryItem, estimate_tokens, score_relevance
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .compressor import MapReduceCompressor


class MemoryManager:
    def __init__(
        self,
        token_budget: int = 8000,
        context_budget: int = 6000,
        memory_path: str = "",
        compressor: MapReduceCompressor | None = None,
        project: str = "",
        trigger_ratio: float = 0.9,
    ):
        self.token_budget = token_budget
        self.context_budget = context_budget
        self.compressor = compressor or MapReduceCompressor()
        self.short_term = ShortTermMemory(
            token_budget=token_budget,
            trigger_ratio=trigger_ratio,
        )
        self.long_term = LongTermMemory(path=memory_path)
        self.project = project

    # ----------------------------------------------------------
    # 四种记忆写入
    # ----------------------------------------------------------

    def add_conversation(self, role: str, content: str):
        self.short_term.add(MemoryItem(type="conversation", role=role, content=content))
        # 从对话提取潜在事实（简单的用户偏好/项目信息句式）
        self._auto_extract_fact(content)
        self._extract_explicit_memory(content)
        self._compress_if_needed()

    def add_fact(
        self,
        content: str,
        keywords: list[str] | None = None,
        scope: str = "project",
    ):
        self.long_term.add_fact(content, keywords, scope=scope, project=self.project)

    def add_summary(self, content: str):
        self.short_term._summaries.append(content)

    def add_tool_result(self, content: str):
        self.short_term.add(MemoryItem(type="tool_result", role="tool", content=content))
        self._compress_if_needed()

    # ----------------------------------------------------------
    # 兼容接口（原 ResearchMemory）
    # ----------------------------------------------------------

    def add_user_message(self, content: str):
        self.add_conversation("user", content)

    def add_ai_message(self, content: str):
        self.add_conversation("assistant", content)

    def add_finding(self, content: str):
        self.add_fact(content)

    # ----------------------------------------------------------
    # 上下文组装（Token 预算控制）
    # ----------------------------------------------------------

    def get_context(self, max_tokens: int | None = None) -> str:
        return self.get_retrieved_context("", max_tokens)

    def get_retrieved_context(self, query: str = "", max_tokens: int | None = None) -> str:
        """按相关度组装上下文：长期事实（检索注入）→ 摘要 → 短期对话"""
        budget = max_tokens or self.context_budget
        parts: list[str] = []
        used = 0

        # 1) 长期事实：有 query 按相关度检索（时间衰减），否则取最近 10 条
        if query:
            facts = self.long_term.search(query, 5, project=self.project)
        else:
            facts = self.long_term.get_visible(self.project)[-10:]
        if facts:
            fact_text = "【长期记忆】\n" + "\n".join(
                f"- {f.content[:200]}" for f in facts
            )
            parts.append(fact_text)
            used += estimate_tokens(fact_text)

        # 2) 历史摘要（最近 3 条）
        summaries = self.short_term.get_summaries()
        if summaries:
            sum_text = "【历史摘要】\n" + summaries
            parts.append(sum_text)
            used += estimate_tokens(sum_text)

        # 3) 短期对话（按剩余预算截断）
        conv = self.short_term.get_context(max(1000, budget - used))
        if conv:
            parts.append("【最近对话】\n" + conv)

        return "\n\n".join(parts)

    def get_report_context(self) -> str:
        """生成报告用上下文（事实 + 摘要，不带对话）"""
        parts: list[str] = []
        facts = self.long_term.items
        if facts:
            parts.append("长期事实：\n" + "\n".join(f"- {f.content}" for f in facts))
        summaries = self.short_term.get_summaries()
        if summaries:
            parts.append("历史摘要：\n" + summaries)
        return "\n".join(parts)

    # ----------------------------------------------------------
    # 持久化 / 清理 / 统计
    # ----------------------------------------------------------

    def save(self):
        self.long_term.save()

    def load(self):
        self.long_term.load()

    def clear(self):
        self.short_term.clear()
        self.long_term.clear()

    def stats(self) -> dict:
        return {
            "short_term_items": len(self.short_term.items),
            "short_term_tokens": sum(i.tokens for i in self.short_term.items),
            "summaries": len(self.short_term.summaries),
            "long_term_facts": len(self.long_term.items),
        }

    def search_facts(self, query: str, limit: int = 5) -> list[MemoryItem]:
        """长期事实关键词检索"""
        return self.long_term.search(query, limit, project=self.project)

    def _compress_if_needed(self):
        """短期记忆达到 trigger_ratio 时触发压缩"""
        if self.short_term.needs_compression():
            self.short_term.compress_old(self.compressor, keep_recent_rounds=3)

    # ----------------------------------------------------------
    # 简单事实自动提取（用户偏好/项目信息）
    # ----------------------------------------------------------

    _FACT_PATTERNS = (
        (r"我(?:是|叫|喜欢|偏好|希望|想用)(.{1,40}?)(?:[。！!？?]|$)", "用户偏好"),
        (r"项目(?:是|使用|采用|基于)(.{1,50}?)(?:[。！!？?]|$)", "项目信息"),
    )

    _REMEMBER_PATTERNS = (
        r"(?:记住|记一下|记下来|以后记得|下次记得|保存(?:这个)?偏好)"
        r"(?:[:：]?\s*)(.{1,60}?)(?:[。！!？?]|$)",
    )

    def _auto_extract_fact(self, content: str):
        import re

        for pattern, tag in self._FACT_PATTERNS:
            m = re.search(pattern, content)
            if m and len(m.group(1).strip()) >= 4:
                self.long_term.add_fact(f"{tag}：{m.group(1).strip()}")

    def _extract_explicit_memory(self, content: str):
        """显式记忆指令：'记住/记一下/以后记得...' → 存入长期（global 跨项目）"""
        import re

        m = re.search(self._REMEMBER_PATTERNS[0], content)
        if m and len(m.group(1).strip()) >= 3:
            self.add_fact("用户偏好（显式）：" + m.group(1).strip(), scope="global")


# 兼容旧接口（原 ResearchMemory 由 MemoryManager 取代）
ResearchMemory = MemoryManager
