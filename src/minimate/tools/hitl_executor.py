"""HITL 拦截层 —— 继承 ToolExecutor，只覆写执行入口

未启用 HITL 或工具不在危险名单时，行为与普通 ToolExecutor 完全一致；
启用且危险时，先发起人工审批，按结果决定执行 / 拒绝 / 跳过 / 修改参数。
"""

from __future__ import annotations

import json
from typing import Optional

from ..hitl import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResult,
    HitlHandler,
)
from .core import ToolExecutor


class HitlToolExecutor(ToolExecutor):
    """带人工审批的工具执行器"""

    def __init__(self, handler: Optional[HitlHandler] = None):
        super().__init__()
        self.handler = handler

    def _should_intercept(self, tool_name: str) -> bool:
        return bool(
            self.handler
            and self.handler.is_enabled()
            and ApprovalPolicy.requires_approval(tool_name)
        )

    def _apply_result(
        self,
        tool_name: str,
        result: ApprovalResult,
        original_args: str,
        effective_args: str,
    ) -> tuple[str, bool] | None:
        """审批结果处理：返回 (是否继续执行, 拒绝/跳过消息 或 None)"""
        if result.is_rejected:
            reason = result.reason or "用户拒绝了此操作"
            return (f"[HITL] 操作已被拒绝：{reason}", False)
        if result.is_skipped:
            return ("[HITL] 操作已被跳过", False)
        return None

    def execute(self, tool_name: str, **kwargs) -> str:
        """FC 通道：危险工具先审批"""
        if not self._should_intercept(tool_name):
            return super().execute(tool_name, **kwargs)

        original = json.dumps(kwargs, ensure_ascii=False)
        request = ApprovalRequest(tool_name=tool_name, arguments=original)
        result = self.handler.request_approval(request)
        handled = self._apply_result(tool_name, result, original, result.effective_arguments(original))
        if handled is not None:
            return handled[0]

        if result.decision is ApprovalDecision.MODIFIED:
            try:
                kwargs = json.loads(result.effective_arguments(original))
                if not isinstance(kwargs, dict):
                    return "[HITL] 修改后的参数不是 JSON 对象，已取消执行"
            except json.JSONDecodeError:
                return "[HITL] 修改后的参数不是合法 JSON，已取消执行"
        return super().execute(tool_name, **kwargs)

    def execute_text(self, tool_name: str, text: str = "") -> str:
        """文本协议通道：危险工具先审批"""
        if not self._should_intercept(tool_name):
            return super().execute_text(tool_name, text)

        request = ApprovalRequest(tool_name=tool_name, arguments=text)
        result = self.handler.request_approval(request)
        handled = self._apply_result(
            tool_name, result, text, result.effective_arguments(text)
        )
        if handled is not None:
            return handled[0]

        effective = result.effective_arguments(text)
        return super().execute_text(tool_name, effective)
