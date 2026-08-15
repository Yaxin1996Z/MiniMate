"""HITL 人工审批单元测试 —— 策略 / 数据载体 / 终端交互 / 拦截层"""

import io
import unittest

from minimate.hitl import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResult,
    TerminalHitlHandler,
)
from minimate.tools import Tool
from minimate.tools.hitl_executor import HitlToolExecutor


class ApprovalPolicyTest(unittest.TestCase):
    """危险工具静态名单"""

    def test_dangerous_tools(self):
        for t in ("run_shell", "write_file", "save_file"):
            self.assertTrue(ApprovalPolicy.requires_approval(t))

    def test_safe_tools(self):
        for t in ("read_file", "list_files", "find_files", "grep_files",
                  "search_code", "web_search", "query_knowledge"):
            self.assertFalse(ApprovalPolicy.requires_approval(t))

    def test_danger_level(self):
        self.assertEqual(ApprovalPolicy.danger_level("run_shell"), "高")
        self.assertEqual(ApprovalPolicy.danger_level("write_file"), "中")


class ApprovalRequestTest(unittest.TestCase):
    """审批请求展示"""

    def test_display_text_contains_info(self):
        req = ApprovalRequest(
            tool_name="write_file",
            arguments='{"path": "a.txt", "content": "你好"}',
        )
        text = req.to_display_text()
        self.assertIn("write_file", text)
        self.assertIn("path: a.txt", text)
        self.assertIn("将写入或覆盖文件内容", text)

    def test_display_text_aligns_cjk(self):
        """中文字符按 2 列宽对齐，边框闭合"""
        req = ApprovalRequest(
            tool_name="write_file",
            arguments='{"路径": "中文目录/文件.txt", "内容": "中文内容"}',
        )
        text = req.to_display_text()
        for line in text.splitlines():
            self.assertTrue(line.startswith(("┌", "│", "├", "└")))
            self.assertTrue(line.endswith(("┐", "│", "┤", "┘")))


class ApprovalResultTest(unittest.TestCase):
    """审批结果参数选择"""

    def test_effective_arguments(self):
        r = ApprovalResult.modified('{"path": "b.txt"}')
        self.assertEqual(r.effective_arguments('{"path": "a.txt"}'), '{"path": "b.txt"}')
        r2 = ApprovalResult.approve()
        self.assertEqual(r2.effective_arguments('{"path": "a.txt"}'), '{"path": "a.txt"}')
        r3 = ApprovalResult.reject("路径有误")
        self.assertTrue(r3.is_rejected)
        r4 = ApprovalResult.skip()
        self.assertTrue(r4.is_skipped)


class TerminalHitlHandlerTest(unittest.TestCase):
    """终端交互：五种决策 + 保守拒绝"""

    def _handler(self, inputs: str) -> tuple[TerminalHitlHandler, io.StringIO]:
        out = io.StringIO()
        handler = TerminalHitlHandler(
            enabled=True,
            out=out,
            inp=io.StringIO(inputs),
            interactive=True,
        )
        return handler, out

    def _request(self):
        return ApprovalRequest(tool_name="write_file", arguments='{"path": "a.txt"}')

    def test_approve(self):
        h, _ = self._handler("y\n")
        result = h.request_approval(self._request())
        self.assertEqual(result.decision, ApprovalDecision.APPROVED)

    def test_enter_approves(self):
        h, _ = self._handler("\n")
        self.assertEqual(
            h.request_approval(self._request()).decision,
            ApprovalDecision.APPROVED,
        )

    def test_approve_all_then_auto(self):
        h, out = self._handler("a\n")
        r1 = h.request_approval(self._request())
        self.assertEqual(r1.decision, ApprovalDecision.APPROVED_ALL)
        # 同一工具再次请求：自动放行，不再弹框
        r2 = h.request_approval(self._request())
        self.assertEqual(r2.decision, ApprovalDecision.APPROVED_ALL)
        self.assertIn("自动通过", out.getvalue())
        # 不同工具仍需审批
        h2, _ = self._handler("y\n")
        h2.approved_all = set(h.approved_all)
        r3 = h2.request_approval(ApprovalRequest(tool_name="run_shell", arguments="ls"))
        self.assertEqual(r3.decision, ApprovalDecision.APPROVED)

    def test_reject_with_reason(self):
        h, _ = self._handler("n\n路径有误\n")
        result = h.request_approval(self._request())
        self.assertEqual(result.decision, ApprovalDecision.REJECTED)
        self.assertEqual(result.reason, "路径有误")

    def test_skip(self):
        h, _ = self._handler("s\n")
        self.assertEqual(
            h.request_approval(self._request()).decision,
            ApprovalDecision.SKIPPED,
        )

    def test_modified(self):
        h, _ = self._handler("m\n{\"path\": \"b.txt\"}\n")
        result = h.request_approval(self._request())
        self.assertEqual(result.decision, ApprovalDecision.MODIFIED)
        self.assertEqual(result.modified_arguments, '{"path": "b.txt"}')

    def test_invalid_inputs_rejected_conservatively(self):
        h, _ = self._handler("x\nx\nx\nx\nx\n")
        result = h.request_approval(self._request())
        self.assertEqual(result.decision, ApprovalDecision.REJECTED)
        self.assertIn("连续多次无效输入", result.reason)

    def test_clear_approved_all(self):
        h, _ = self._handler("a\n")
        h.request_approval(self._request())
        self.assertIn("write_file", h.approved_all)
        h.clear_approved_all()
        self.assertEqual(h.approved_all, set())


class HitlToolExecutorTest(unittest.TestCase):
    """拦截层：启用 + 危险工具才审批"""

    def _executor(self, handler=None) -> HitlToolExecutor:
        ex = HitlToolExecutor(handler)
        ex.register(
            Tool(
                name="write_file",
                description="测试写入",
                func=lambda path, content: f"已写入 {path}",
            )
        )
        ex.register(
            Tool(
                name="read_file",
                description="测试读取",
                func=lambda path: f"内容[{path}]",
            )
        )
        return ex

    def test_disabled_passes_through(self):
        out = io.StringIO()
        ex = self._executor(TerminalHitlHandler(enabled=False, out=out, inp=io.StringIO(""), interactive=True))
        self.assertEqual(ex.execute("write_file", path="a.txt", content="x"), "已写入 a.txt")

    def test_safe_tool_no_intercept(self):
        out = io.StringIO()
        ex = self._executor(TerminalHitlHandler(enabled=True, out=out, inp=io.StringIO(""), interactive=True))
        self.assertEqual(ex.execute("read_file", path="a.txt"), "内容[a.txt]")

    def test_approved_executes(self):
        out = io.StringIO()
        ex = self._executor(TerminalHitlHandler(enabled=True, out=out, inp=io.StringIO("y\n"), interactive=True))
        self.assertEqual(ex.execute("write_file", path="a.txt", content="x"), "已写入 a.txt")

    def test_rejected_returns_hitl_message(self):
        out = io.StringIO()
        ex = self._executor(TerminalHitlHandler(enabled=True, out=out, inp=io.StringIO("n\n原因\n"), interactive=True))
        result = ex.execute("write_file", path="a.txt", content="x")
        self.assertTrue(result.startswith("[HITL] 操作已被拒绝"))
        self.assertIn("原因", result)

    def test_skipped_returns_hitl_message(self):
        out = io.StringIO()
        ex = self._executor(TerminalHitlHandler(enabled=True, out=out, inp=io.StringIO("s\n"), interactive=True))
        self.assertEqual(ex.execute("write_file", path="a.txt", content="x"), "[HITL] 操作已被跳过")

    def test_modified_arguments_executed(self):
        out = io.StringIO()
        ex = self._executor(
            TerminalHitlHandler(
                enabled=True,
                out=out,
                inp=io.StringIO('m\n{"path": "b.txt", "content": "x"}\n'),
                interactive=True,
            )
        )
        self.assertEqual(ex.execute("write_file", path="a.txt", content="x"), "已写入 b.txt")

    def test_approve_all_scope(self):
        out = io.StringIO()
        ex = self._executor(TerminalHitlHandler(enabled=True, out=out, inp=io.StringIO("a\n"), interactive=True))
        self.assertEqual(ex.execute("write_file", path="a.txt", content="x"), "已写入 a.txt")
        # 同工具自动放行（无输入也不会阻塞）
        self.assertEqual(ex.execute("write_file", path="b.txt", content="y"), "已写入 b.txt")


if __name__ == "__main__":
    unittest.main()
