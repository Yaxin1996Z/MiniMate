"""对话历史压缩器 —— 压缩 Agent 实际发给 LLM 的消息列表

与 ContextCompressor 的区别：
- ContextCompressor 压的是短期记忆条目（MemoryItem）
- 本类压的是 Agent 实际发给 LLM 的 messages 列表（dict 消息），
  在 ReAct 主循环"调 LLM 前"评估并压缩，真正缩短发送给模型的 token。

关键约束：分割点必须落在 user 消息边界，避免切断 tool_call / tool_result 成对协议。
"""

from .. import llm
from ..memory.token_budget import TokenBudget


SUMMARY_PROMPT = """请把下面的对话历史压缩成简明摘要，保留：
1. 用户提出的关键诉求与目标
2. Agent 已经完成的关键操作（哪些工具调用了什么、返回了什么核心结果）
3. 已经达成的共识或结论
4. 仍未解决的问题或待办

不要复述每条原文，不要列举所有工具调用，不要保留无关闲聊。
输出 1-3 段中文，不要用列表，不要加任何前缀或元描述。

=== 待压缩的对话 ===
%s
=== 待压缩的对话（结束）===
"""


class ConversationHistoryCompactor:
    def __init__(
        self,
        retain_recent_rounds: int = 3,
        max_summary_input_chars: int = 60_000,
        trigger_tokens: int = 60_000,
    ):
        self.retain_recent_rounds = max(1, retain_recent_rounds)
        self.max_summary_input_chars = max_summary_input_chars
        self.trigger_tokens = trigger_tokens

    def compact_if_needed(self, history: list[dict]) -> bool:
        return self._compact(history, self.trigger_tokens, force=False)

    def compact_now(self, history: list[dict]) -> bool:
        return self._compact(history, 0, force=True)

    def _compact(self, history: list[dict], trigger_tokens: int, force: bool) -> bool:
        if not history:
            return False
        current_tokens = TokenBudget.estimate_messages_tokens(history)
        if not force and current_tokens < trigger_tokens:
            return False

        system_end = 1 if history[0].get("role") == "system" else 0
        user_indices = [
            i
            for i in range(system_end, len(history))
            if history[i].get("role") == "user"
        ]
        if len(user_indices) <= self.retain_recent_rounds:
            return False
        split_idx = user_indices[-self.retain_recent_rounds]
        if split_idx <= system_end:
            return False

        old_msgs = history[system_end:split_idx]
        if not old_msgs:
            return False
        summary = self._summarize(old_msgs)
        if not summary:
            return False

        rebuilt = history[:system_end] + [
            {"role": "user", "content": "[已压缩的历史对话摘要]\n" + summary.strip()},
            {"role": "assistant", "content": "好的，我已了解之前的上下文，请继续。"},
        ] + history[split_idx:]
        after_tokens = TokenBudget.estimate_messages_tokens(rebuilt)
        history.clear()
        history.extend(rebuilt)
        print(f"  [压缩对话历史] tokens {current_tokens} -> {after_tokens}，"
              f"消息 {len(old_msgs) + system_end} -> {len(rebuilt)}")
        return True

    def _summarize(self, messages: list[dict]) -> str:
        sb: list[str] = []
        total = 0
        for m in messages:
            role = m.get("role", "unknown")
            content = m.get("content") or ""
            chunk = f"{role.upper()}: {content}"
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                chunk += f"\n  TOOL_CALL {fn.get('name', '')}: {fn.get('arguments', '')}"
            chunk += "\n\n"
            sb.append(chunk)
            total += len(chunk)
            if total > self.max_summary_input_chars:
                sb.append("...(超长内容已截断)\n")
                break
        prompt = SUMMARY_PROMPT % "".join(sb)
        result = llm.call(
            prompt,
            system="你是一个对话摘要助手，只输出摘要本身，不输出元描述。",
            temperature=0.1,
        )
        if not result or result.startswith("[API 错误]"):
            return ""
        return result.strip()
