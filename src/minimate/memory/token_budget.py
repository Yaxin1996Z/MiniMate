"""Token 预算管理器

策略：
1. 模型上下文窗口减去（系统提示 + 工具定义 + 回复）预留 = 对话可用预算
2. 每次调用 LLM 前检查消息列表是否在预算内
3. 超阈值触发压缩（短期记忆 / 对话历史两层）
"""

from dataclasses import dataclass

from .core import estimate_tokens


@dataclass
class ContextProfile:
    """上下文策略配置"""

    max_context_window: int = 131072          # 模型上下文窗口（DeepSeek 128K 量级）
    short_term_budget: int = 8000             # 短期记忆 token 预算
    compression_trigger_ratio: float = 0.9    # 短期记忆压缩触发占用率
    compression_trigger_tokens: int = 60000   # 对话历史（LLM messages）压缩触发阈值

    def summary(self) -> str:
        return (
            f"窗口 {self.max_context_window} / 短期预算 {self.short_term_budget} / "
            f"压缩阈值 {int(self.compression_trigger_ratio * 100)}%"
        )


class TokenBudget:
    """预算计算 + 累计用量统计"""

    def __init__(
        self,
        context_window: int = 131072,
        reserved_for_system: int = 500,
        reserved_for_tools: int = 800,
        reserved_for_response: int = 2000,
    ):
        self.context_window = context_window
        self.reserved_for_system = reserved_for_system
        self.reserved_for_tools = reserved_for_tools
        self.reserved_for_response = reserved_for_response
        self._total_input = 0
        self._total_output = 0
        self._total_cached = 0
        self._call_count = 0

    def available_for_conversation(self) -> int:
        """对话历史可用的 token 预算"""
        return (
            self.context_window
            - self.reserved_for_system
            - self.reserved_for_tools
            - self.reserved_for_response
        )

    def is_within_budget(self, messages: list[dict]) -> bool:
        return self.estimate_messages_tokens(messages) <= self.available_for_conversation()

    def needs_compression(self, memory, trigger_ratio: float = 0.9) -> bool:
        """短期记忆占用是否达到压缩阈值（取 min(记忆预算, 对话可用)）"""
        budget = min(memory.get_max_tokens(), self.available_for_conversation())
        return memory.get_token_count() >= budget * trigger_ratio

    def record_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ):
        self._total_input += input_tokens
        self._total_output += output_tokens
        self._total_cached += max(0, cached_input_tokens)
        self._call_count += 1

    def usage_report(self) -> str:
        avg = (
            self._total_input / self._call_count
            if self._call_count
            else 0
        )
        return (
            f"Token 统计: 调用 {self._call_count} 次 | 总输入: {self._total_input} | "
            f"总输出: {self._total_output} | cached: {self._total_cached} | "
            f"平均输入: {avg:.0f} | 窗口: {self.context_window} "
            f"(可用: {self.available_for_conversation()})"
        )

    @property
    def total_input_tokens(self) -> int:
        return self._total_input

    @property
    def total_output_tokens(self) -> int:
        return self._total_output

    @property
    def total_cached_input_tokens(self) -> int:
        return self._total_cached

    @property
    def llm_call_count(self) -> int:
        return self._call_count

    @staticmethod
    def estimate_messages_tokens(messages: list[dict] | None) -> int:
        """估算 LLM 消息列表 token 数（内容 + 工具参数 + 每条消息 4 token 开销）"""
        if not messages:
            return 0
        total = 0
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, str):
                total += estimate_tokens(content)
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                total += estimate_tokens(fn.get("arguments") or "")
        total += len(messages) * 4
        return total
