"""MemoryManager —— 整合四种记忆 + Token 预算 + 持久化

记忆分层：
  conversation - 短期对话（token 预算内，超限淘汰 + 摘要保留）
  fact         - 长期事实（用户偏好/项目信息，JSON 持久化 + 去重 + 检索）
  summary      - 压缩摘要（Map-Reduce 生成，跨窗口保留主线）
  tool_result  - 工具执行结果（短期）

上下文组装顺序：长期事实 → 历史摘要 → 短期对话（按预算截断）
"""

import os

from .core import MemoryItem, estimate_tokens, score_relevance
from .short_term import ShortTermMemory
from .long_term import LongTermMemory
from .tool_memory import ToolMemory
from .compressor import MapReduceCompressor
from .token_budget import ContextProfile, TokenBudget
from .. import llm


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
        self.context_profile = ContextProfile(
            short_term_budget=token_budget,
            compression_trigger_ratio=trigger_ratio,
        )
        self.token_budget_control = TokenBudget(self.context_profile.max_context_window)
        self.tool_memory = ToolMemory()

    # ----------------------------------------------------------
    # 四种记忆写入
    # ----------------------------------------------------------

    def add_conversation(self, role: str, content: str):
        self.short_term.store(MemoryItem(type="conversation", role=role, content=content))
        # 从对话提取潜在事实（仅用户消息，避免把助手复述/反问也当成事实）
        if role == "user":
            self._auto_extract_fact(content)
            self._extract_explicit_memory(content)
        self._compress_if_needed()

    def add_fact(
        self,
        content: str,
        keywords: list[str] | None = None,
        scope: str = "project",
        source: str = "manual",
    ):
        self.long_term.add_fact(
            content, keywords, scope=scope, project=self.project, source=source
        )

    def add_summary(self, content: str):
        self.short_term._summaries.append(content)

    # ----------------------------------------------------------
    # 常用写入接口
    # ----------------------------------------------------------

    def add_user_message(self, content: str):
        self.add_conversation("user", content)

    def add_ai_message(self, content: str):
        self.add_conversation("assistant", content)

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

    def _compress_if_needed(self) -> bool:
        """短期记忆达到 trigger_ratio 时触发压缩；淘汰前先用 LLM 提取可长期记忆的事实"""
        if self.short_term.needs_compression():
            return self.short_term.compress_old(
                self.compressor,
                keep_recent_rounds=3,
                on_evict=self._llm_extract_facts,
            )
        return False

    def _llm_extract_facts(self, items: list[MemoryItem]) -> None:
        """压缩淘汰前，用 LLM 从旧条目中提取稳定事实写入长期记忆

        结构化提示词 + 每行一条事实 + 启发式后过滤（临时请求/猜测拦截，持久线索放行）。
        """
        texts = [
            f"{i.role.upper()}({i.type}): {i.content}"
            for i in items
            if (i.content or "").strip()
        ]
        if not texts:
            return
        prompt = self._EXTRACT_FACTS_PROMPT % "\n\n".join(texts)
        raw = llm.call(
            prompt,
            system="你是一个信息提取助手，只输出关键事实，不输出其他内容。",
            temperature=0.1,
        )
        if not raw or raw.startswith("[API 错误]"):
            return
        for line in raw.splitlines():
            fact = self._normalize_fact_line(line)
            if self._is_persistent_fact(fact):
                self.long_term.add_fact(
                    fact, scope="project", project=self.project, source="llm"
                )

    _EXTRACT_FACTS_PROMPT = (
        "请从以下对话中提取\"跨会话仍然成立、未来复用仍有价值\"的稳定事实，"
        "格式为每行一条：\n"
        "- 用户偏好和习惯\n"
        "- 项目信息（名称、路径、技术栈）\n"
        "- 重要决策和约定\n\n"
        "只保留用户明确说明、或工具/代码库可验证的信息。\n"
        "绝对不要提取以下内容：\n"
        "- 当前这一轮让你执行的临时任务、步骤、todo\n"
        "- 一次性的文件名、目录名、输出要求\n"
        "- 模型自己的猜测、纠错、提醒、推断\n"
        "- \"用户想要/需要/让我/请你...\" 这类请求句\n\n"
        "对话内容：\n%s\n\n"
        "请每行一条事实，不要多余解释。"
    )

    _EPHEMERAL_FACT_PREFIXES = (
        "用户想", "用户要", "用户需要", "用户请求", "帮我", "让我",
        "新建", "创建", "删除", "修改", "生成", "补充要求", "当前这一轮", "本次任务",
    )

    _SPECULATION_CUES = ("可能", "应该", "猜测", "推测", "笔误", "提醒")

    _DURABLE_FACT_HINTS = (
        "用户偏好", "用户习惯", "喜欢", "倾向", "项目", "仓库", "路径", "技术栈",
        "版本", "模型", "接口", "配置", "环境变量", "命令", "约定", "规则", "默认",
    )

    @staticmethod
    def _normalize_fact_line(line: str) -> str:
        """去掉行首 "- " / "• " 项目符号并去除首尾空白"""
        fact = (line or "").strip()
        if fact.startswith("- "):
            fact = fact[2:]
        elif fact.startswith("• "):
            fact = fact[2:]
        return fact.strip()

    @staticmethod
    def _is_persistent_fact(fact: str) -> bool:
        """启发式过滤：太短/临时请求/猜测句不存；带标签或含持久线索的才存"""
        if len(fact) <= 5:
            return False
        normalized = fact.lower()
        if any(
            normalized.startswith(p)
            for p in MemoryManager._EPHEMERAL_FACT_PREFIXES
        ):
            return False
        if any(c in normalized for c in MemoryManager._SPECULATION_CUES):
            return False
        if "：" in normalized or ":" in normalized:
            return True
        return any(h in normalized for h in MemoryManager._DURABLE_FACT_HINTS)

    # ----------------------------------------------------------
    # 简单事实自动提取（用户偏好/项目信息）
    # ----------------------------------------------------------

    _FACT_PATTERNS = (
        (r"我(?:的|家)?((?:[^，。！!？?]{0,10}?)(?:叫|是|喜欢|偏好|希望|想用)(.{1,40}?))(?:[，。！!？?]|$)", "用户偏好"),
        (r"项目(?:是|叫|使用|采用|基于)(.{1,50}?)(?:[。！!？?]|$)", "项目信息"),
    )

    _REMEMBER_PATTERNS = (
        r"(?:记住了|记住|记一下|记下来|以后记得|下次记得|保存(?:这个)?偏好)"
        r"(?:[:：！!？?，,\s]*)(.{1,60}?)(?:[。！!？?]|$)",
    )

    def _auto_extract_fact(self, content: str):
        import re

        # 疑问句不提取（什么/怎么/吗/呢/哪/谁/几/多少 或带问号）
        if re.search(r"[？?]|什么|怎么|哪|谁|吗|呢|几|多少", content):
            return

        for pattern, tag in self._FACT_PATTERNS:
            m = re.search(pattern, content)
            if not m:
                continue
            fact = re.sub(r"[，。！!？?\s]+$", "", m.group(0)).strip()
            if len(fact) >= 4:
                self.long_term.add_fact(f"{tag}：{fact}", source="agent")

    def _extract_explicit_memory(self, content: str):
        """显式记忆指令：'记住/记一下/以后记得...' → 存入长期（global 跨项目）"""
        import re

        m = re.search(self._REMEMBER_PATTERNS[0], content)
        if m and len(m.group(1).strip()) >= 3:
            self.add_fact("用户偏好（显式）：" + m.group(1).strip(), scope="global")

    # ----------------------------------------------------------
    # 门面接口：统一记忆存取
    # ----------------------------------------------------------

    def set_project_path(self, project_path: str):
        """切换当前项目（决定长期记忆的 project 作用域）"""
        if project_path:
            self.project = os.path.abspath(project_path)

    def add_tool_result(
        self,
        content: str,
        result: str | None = None,
        max_chars: int = 500,
    ):
        """工具结果进短期记忆（截断过长结果）

        兼容两种调用：add_tool_result("内容") 旧接口；add_tool_result("工具名", "结果") 新接口。
        """
        if result is None:
            item_content = content
            metadata = {"source": "tool"}
        else:
            truncated = (
                result if len(result) <= max_chars else result[:max_chars] + "...(已截断)"
            )
            item_content = f"[{content}] {truncated}"
            metadata = {"source": "tool", "toolName": content}
        item = MemoryItem(
            type="tool_result",
            role="tool",
            content=item_content,
            metadata=metadata,
        )
        self.short_term.store(item)
        self._compress_if_needed()

    def store_fact(self, fact: str, scope: str = "project", source: str = "manual"):
        """存储关键事实到长期记忆"""
        self.add_fact(fact, scope=scope, source=source)

    def list_long_term(self) -> list[MemoryItem]:
        return self.long_term.get_all(None)

    def search_long_term(self, query: str, limit: int) -> list[MemoryItem]:
        return self.long_term.search(query, limit, project=self.project)

    def delete_long_term(self, entry_id: str) -> bool:
        """按 id 删除一条长期事实"""
        return self.long_term.delete(entry_id)

    def record_token_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ):
        self.token_budget_control.record_usage(
            input_tokens, output_tokens, cached_input_tokens
        )

    def compress_if_needed(self) -> bool:
        """检查并触发压缩（基于 TokenBudget 阈值）"""
        if not self.token_budget_control.needs_compression(
            self.short_term, self.context_profile.compression_trigger_ratio
        ):
            return False
        return self._compress_if_needed()

    def clear_short_term(self):
        self.short_term.clear()

    def clear_long_term(self):
        self.long_term.clear()

    def get_system_status(self) -> str:
        """记忆系统整体状态"""
        return (
            f"上下文策略: {self.context_profile.summary()}\n"
            + self.short_term.get_status_summary()
            + "\n"
            + self.long_term.get_status_summary()
            + "\n"
            + self.token_budget_control.usage_report()
        )
