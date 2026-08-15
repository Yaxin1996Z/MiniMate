"""HITL —— 人工审批（Human-in-the-Loop）

设计参考 minimate-HITL.md：
  - ApprovalPolicy：静态规则判断危险工具，不引入 LLM 动态判断
  - ApprovalRequest / ApprovalResult：审批数据载体（五种决策）
  - HitlHandler / TerminalHitlHandler：终端审批交互（线程安全）

默认关闭（/hitl on 开启），危险操作名单按工具名静态维护；
拒绝原因会作为工具返回回灌给 Agent，让其调整规划。
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# 审批策略（静态规则）
# ============================================================

class ApprovalPolicy:
    """危险工具静态名单：写磁盘 / 执行命令需要人工确认"""

    # 工具名 -> (危险等级, 风险描述)
    DANGEROUS_TOOLS: dict[str, tuple[str, str]] = {
        "run_shell": (
            "高",
            "将在系统上执行 Shell 命令，可能修改文件、安装软件或影响系统状态",
        ),
        "write_file": (
            "中",
            "将写入或覆盖文件内容，原有内容将丢失",
        ),
        "save_file": (
            "中",
            "将保存内容到文件，可能覆盖同名文件",
        ),
    }

    SAFE_TOOLS_HINT = "read_file / list_files / find_files / grep_files / search_code / web_search / query_knowledge"

    @classmethod
    def requires_approval(cls, tool_name: str) -> bool:
        return tool_name in cls.DANGEROUS_TOOLS

    @classmethod
    def danger_level(cls, tool_name: str) -> str:
        return cls.DANGEROUS_TOOLS.get(tool_name, ("低", ""))[0]

    @classmethod
    def risk_description(cls, tool_name: str) -> str:
        return cls.DANGEROUS_TOOLS.get(tool_name, ("低", "安全的只读操作"))[1]


# ============================================================
# 审批数据载体
# ============================================================

class ApprovalDecision(Enum):
    """用户决策：批准 / 全部放行 / 拒绝 / 修改参数 / 跳过"""

    APPROVED = "approved"
    APPROVED_ALL = "approved_all"
    REJECTED = "rejected"
    MODIFIED = "modified"
    SKIPPED = "skipped"


def display_width(text: str) -> int:
    """终端显示列宽：CJK / 全角 / emoji 占 2 列，其余占 1 列"""
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


def _box_width(text: str) -> int:
    """计算单行内容（含两端空格）需要的边框内宽"""
    return display_width(text) + 2


def _pad(text: str, width: int) -> str:
    """按终端显示列宽补齐空格（CJK/emoji 占 2 列，避免右边框错位）"""
    pad = max(0, width - display_width(text))
    return text + " " * pad


@dataclass
class ApprovalRequest:
    """审批请求：工具名 + 参数 + 危险等级 + 风险描述"""

    tool_name: str
    arguments: str = ""
    suggestion: str = ""
    caller_context: str = ""

    @property
    def danger_level(self) -> str:
        return ApprovalPolicy.danger_level(self.tool_name)

    @property
    def risk_description(self) -> str:
        return ApprovalPolicy.risk_description(self.tool_name)

    def _parse_arguments(self) -> dict:
        """参数 JSON 解析：展示用，解析失败返回原始文本"""
        try:
            data = json.loads(self.arguments or "{}")
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}

    def to_display_text(self, max_value_chars: int = 120) -> str:
        """格式化为终端审批框（对齐 CJK/emoji 宽度，长参数截断）"""
        level_icon = {"高": "高", "中": "中", "低": "低"}.get(self.danger_level, "?")
        header = f"需要审批 ｜ 工具: {self.tool_name} ｜ 等级: {level_icon}"
        risk = f"风险: {self.risk_description}"
        width = max(_box_width(header), _box_width(risk), 40)

        lines = [
            "┌" + "─" * (width - 2) + "┐",
            "│ " + _pad(header, width - 3) + "│",
            "│ " + _pad(risk, width - 3) + "│",
            "├" + "─" * (width - 2) + "┤",
        ]
        args = self._parse_arguments()
        if args:
            for key, value in args.items():
                text = str(value)
                if len(text) > max_value_chars:
                    text = text[:max_value_chars] + f"...({len(str(value))} 字符)"
                line = f"{key}: {text}"
                lines.append("│ " + _pad(line, width - 3) + "│")
        else:
            body = (self.arguments or "(无参数)")[:max_value_chars]
            lines.append("│ " + _pad(body, width - 3) + "│")
        lines.append("└" + "─" * (width - 2) + "┘")
        return "\n".join(lines)


@dataclass
class ApprovalResult:
    """审批结果：决策 + 修改后的参数 + 拒绝原因"""

    decision: ApprovalDecision
    modified_arguments: str = ""
    reason: str = ""

    @classmethod
    def approve(cls) -> "ApprovalResult":
        return cls(ApprovalDecision.APPROVED)

    @classmethod
    def approve_all(cls) -> "ApprovalResult":
        return cls(ApprovalDecision.APPROVED_ALL)

    @classmethod
    def reject(cls, reason: str = "") -> "ApprovalResult":
        return cls(ApprovalDecision.REJECTED, reason=reason)

    @classmethod
    def skip(cls) -> "ApprovalResult":
        return cls(ApprovalDecision.SKIPPED)

    @classmethod
    def modified(cls, arguments: str) -> "ApprovalResult":
        return cls(ApprovalDecision.MODIFIED, modified_arguments=arguments)

    def effective_arguments(self, original: str) -> str:
        """生效参数：仅 MODIFIED 使用修改值，其余一律用原始参数"""
        if (
            self.decision is ApprovalDecision.MODIFIED
            and self.modified_arguments
            and self.modified_arguments.strip()
        ):
            return self.modified_arguments
        return original

    @property
    def is_rejected(self) -> bool:
        return self.decision is ApprovalDecision.REJECTED

    @property
    def is_skipped(self) -> bool:
        return self.decision is ApprovalDecision.SKIPPED


# ============================================================
# 审批处理接口 + 终端实现
# ============================================================

class HitlHandler:
    """HITL 处理接口：启用开关 + 发起审批"""

    def is_enabled(self) -> bool:
        raise NotImplementedError

    def set_enabled(self, enabled: bool):
        raise NotImplementedError

    def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        raise NotImplementedError

    def clear_approved_all(self):
        raise NotImplementedError


class TerminalHitlHandler(HitlHandler):
    """终端审批实现：synchronized 串行 + 五种决策 + 连续无效输入保守拒绝"""

    MAX_INVALID_ATTEMPTS = 5

    def __init__(self, enabled: bool = False, out=None, inp=None, interactive=None):
        self._enabled = enabled
        # 固定使用真实 stdout/stdin，避免被多 Agent 的 stdout 捕获重定向
        self.out = out or sys.__stdout__
        self.inp = inp or sys.__stdin__
        self.interactive = interactive  # None 时按输入流是否为 tty 判断
        self._lock = threading.RLock()          # 多 Worker 并发审批串行化
        self.approved_all: set[str] = set()     # 工具级"全部放行"

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def clear_approved_all(self):
        with self._lock:
            self.approved_all.clear()

    def request_approval(self, request: ApprovalRequest) -> ApprovalResult:
        with self._lock:
            if request.tool_name in self.approved_all:
                self.out.write(
                    f"[HITL] {request.tool_name} 已在本次会话中全部放行，自动通过\n"
                )
                self.out.flush()
                return ApprovalResult.approve_all()
            if not self._can_interact():
                # 非交互终端（管道/重定向）下无法人工确认，保守拒绝
                self.out.write(
                    "[HITL] 当前输入流非交互终端，无法人工确认，保守拒绝\n"
                )
                self.out.flush()
                return ApprovalResult.reject("非交互终端，保守拒绝")
            self.out.write("\n────────── 需要审批 ──────────\n")
            self.out.write(request.to_display_text() + "\n")
            self.out.flush()
            return self._prompt_until_decision(request)

    def _can_interact(self) -> bool:
        if self.interactive is not None:
            return self.interactive
        return bool(getattr(self.inp, "isatty", lambda: False)())

    def _prompt_until_decision(self, request: ApprovalRequest) -> ApprovalResult:
        for _ in range(self.MAX_INVALID_ATTEMPTS):
            self.out.write(
                "请选择操作：[y/Enter] 批准 [a] 全部放行 [n] 拒绝 [s] 跳过 [m] 修改参数\n> "
            )
            self.out.flush()
            try:
                raw = self.inp.readline()
            except (EOFError, OSError):
                return ApprovalResult.reject("输入流不可用，保守拒绝")
            choice = (raw or "").strip().lower()

            if choice in ("", "y"):
                return ApprovalResult.approve()
            if choice == "a":
                self.approved_all.add(request.tool_name)
                return ApprovalResult.approve_all()
            if choice == "s":
                return ApprovalResult.skip()
            if choice == "n":
                self.out.write("拒绝原因（回车表示不说明）：")
                self.out.flush()
                try:
                    reason = self.inp.readline().strip()
                except (EOFError, OSError):
                    reason = ""
                return ApprovalResult.reject(reason or "用户拒绝了此操作")
            if choice == "m":
                self.out.write("请输入修改后的完整参数（JSON 格式）：\n> ")
                self.out.flush()
                try:
                    modified = self.inp.readline().strip()
                except (EOFError, OSError):
                    modified = ""
                if not modified:
                    return ApprovalResult.reject("修改参数为空")
                return ApprovalResult.modified(modified)
            self.out.write(" 无法识别的选项，请输入 y/a/n/s/m 之一\n")
        # 连续多次无效输入：保守拒绝（用户没有明确同意就不该执行）
        return ApprovalResult.reject("连续多次无效输入，已保守拒绝")
