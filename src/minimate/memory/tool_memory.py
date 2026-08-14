"""工具调用记忆 —— 记录每次 react 循环中的工具调用（会话内存，不落盘）

三层记忆：短期（对话）→ 长期（事实）→ 工具调用记忆（工具级调用日志）
用途：
- 循环内重复调用检测（按 loop_id 隔离，避免把新任务复用工具误判为重复）
- 会话内结果复用 / 审计 / 失败规避（跨循环可查历史）
"""

import json
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ToolCallRecord:
    tool_name: str
    args: dict | str
    result: str
    ok: bool = True
    step: int = 0
    loop_id: str = ""
    timestamp: float = field(default_factory=time.time)
    duration: float | None = None

    def args_key(self) -> str:
        if isinstance(self.args, dict):
            return json.dumps(self.args, sort_keys=True, ensure_ascii=False)
        return str(self.args or "").strip()


class ToolMemory:
    """会话级工具调用记忆（不落盘）"""

    def __init__(self, max_records: int = 500):
        self._records: list[ToolCallRecord] = []
        self._max_records = max_records

    def new_loop(self) -> str:
        """开启一个 react 循环，返回 loop_id（循环内重复检测按此隔离）"""
        return f"loop-{uuid.uuid4().hex[:8]}"

    def record(
        self,
        tool_name: str,
        args: dict | str,
        result: str,
        ok: bool = True,
        step: int = 0,
        loop_id: str = "",
        duration: float | None = None,
    ) -> ToolCallRecord:
        rec = ToolCallRecord(
            tool_name=tool_name,
            args=args,
            result=result,
            ok=ok,
            step=step,
            loop_id=loop_id,
            duration=duration,
        )
        self._records.append(rec)
        if len(self._records) > self._max_records:
            self._records.pop(0)
        return rec

    def repeated_count(self, tool_name: str, args: dict | str, loop_id: str) -> int:
        """当前循环内相同 (工具, 参数) 的连续尝试次数（不含本次）"""
        if isinstance(args, dict):
            key = json.dumps(args, sort_keys=True, ensure_ascii=False)
        else:
            key = str(args or "").strip()
        count = 0
        for r in reversed(self._records):
            if r.loop_id != loop_id:
                break
            if r.tool_name == tool_name and r.args_key() == key:
                count += 1
            else:
                break
        return count

    def history(self, limit: int = 20) -> list[ToolCallRecord]:
        """会话内工具调用历史（最新在前）"""
        return list(reversed(self._records[-limit:]))

    def failed(self, limit: int = 10) -> list[ToolCallRecord]:
        return [r for r in reversed(self._records) if not r.ok][:limit]

    def clear(self):
        self._records.clear()

    @property
    def records(self) -> list[ToolCallRecord]:
        return list(self._records)

    def __len__(self) -> int:
        return len(self._records)
