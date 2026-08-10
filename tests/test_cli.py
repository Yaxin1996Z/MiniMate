"""CLI 交互模式单元测试 —— mock 输入，验证命令与退出流程"""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from cli import interactive


class InteractiveTest(unittest.TestCase):
    """交互模式：命令处理、退出、Ctrl+C"""

    def setUp(self):
        # 测试中不真实连接 MCP Server
        self.mcp_patcher = patch("cli._load_mcp_tools")
        self.mcp_patcher.start()

    def tearDown(self):
        self.mcp_patcher.stop()

    @patch("builtins.input", side_effect=["/quit"])
    def test_quit_clears_session(self, mock_input):
        buf = io.StringIO()
        with redirect_stdout(buf):
            interactive()
        out = buf.getvalue()
        self.assertIn("MiniMate v", out)          # logo + 版本
        self.assertIn("再见！会话记忆已清除", out)   # 退出提示

    @patch("builtins.input", side_effect=KeyboardInterrupt())
    def test_ctrl_c_exits_cleanly(self, mock_input):
        """Ctrl+C 不抛异常，正常退出并提示记忆清除"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            interactive()
        self.assertIn("再见！会话记忆已清除", buf.getvalue())

    @patch("builtins.input", side_effect=["/mode bad", "/mode plan", "/quit"])
    def test_mode_command(self, mock_input):
        buf = io.StringIO()
        with redirect_stdout(buf):
            interactive()
        out = buf.getvalue()
        self.assertIn("用法：/mode chat|react|plan", out)
        self.assertIn("已切换到 plan 模式", out)

    @patch("builtins.input", side_effect=["/clear", "/memory", "/quit"])
    def test_clear_and_memory(self, mock_input):
        buf = io.StringIO()
        with redirect_stdout(buf):
            interactive()
        out = buf.getvalue()
        self.assertIn("会话记忆已清空", out)
        self.assertIn("记忆为空", out)


if __name__ == "__main__":
    unittest.main()
