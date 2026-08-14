"""CLI 交互模式单元测试 —— mock 输入，验证命令与退出流程"""

import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from cli import _display_width, _input_windows, _repos_command, interactive
from minimate.memory import MemoryManager


class InteractiveTest(unittest.TestCase):
    """交互模式：命令处理、退出、Ctrl+C"""

    def setUp(self):
        # 测试中不真实连接 MCP Server
        self.mcp_patcher = patch("cli._load_mcp_tools")
        self.mcp_patcher.start()

    def tearDown(self):
        self.mcp_patcher.stop()

    @patch("cli._input_line", side_effect=["/quit"])
    def test_quit_clears_session(self, mock_input):
        buf = io.StringIO()
        with redirect_stdout(buf):
            interactive()
        out = buf.getvalue()
        self.assertIn("MiniMate v", out)          # logo + 版本
        self.assertIn("再见！会话记忆已清除", out)   # 退出提示

    @patch("cli._input_line", side_effect=KeyboardInterrupt())
    def test_ctrl_c_exits_cleanly(self, mock_input):
        """Ctrl+C 不抛异常，正常退出并提示记忆清除"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            interactive()
        self.assertIn("再见！会话记忆已清除", buf.getvalue())

    @patch("cli._input_line", side_effect=["/mode bad", "/mode plan", "/quit"])
    def test_mode_command(self, mock_input):
        buf = io.StringIO()
        with redirect_stdout(buf):
            interactive()
        out = buf.getvalue()
        self.assertIn("用法：/mode chat|react|plan", out)
        self.assertIn("已切换到 plan 模式", out)

    @patch("cli._input_line", side_effect=["/mode multi", "写一个文件", "/quit"])
    @patch("cli.MultiAgentOrchestrator")
    def test_mode_multi_uses_orchestrator(self, mock_orch, mock_input):
        """交互模式 /mode multi 后，提问必须走多 Agent 编排器而非 agent.run"""
        instance = mock_orch.return_value
        instance.run.return_value = "多 Agent 汇总结果"
        buf = io.StringIO()
        with redirect_stdout(buf):
            interactive()
        out = buf.getvalue()
        self.assertIn("已切换到 multi 模式", out)
        mock_orch.assert_called_once()
        instance.run.assert_called_once_with("写一个文件")

    @patch("cli._input_line", side_effect=["/clear", "/memory", "/quit"])
    def test_clear_and_memory(self, mock_input):
        buf = io.StringIO()
        with redirect_stdout(buf):
            interactive()
        out = buf.getvalue()
        self.assertIn("会话记忆已清空", out)
        self.assertIn("记忆统计", out)  # /memory 显示统计报告

    @patch("cli._input_line", side_effect=EOFError())
    def test_eof_exits_cleanly(self, mock_input):
        """Ctrl+Z（EOF）正常退出并提示记忆清除"""
        buf = io.StringIO()
        with redirect_stdout(buf):
            interactive()
        self.assertIn("再见！会话记忆已清除", buf.getvalue())


class InputHistoryTest(unittest.TestCase):
    """Windows 自绘输入的 ↑/↓ 历史导航与基础编辑"""

    def _run(self, keys, history=None):
        import msvcrt

        with patch.object(msvcrt, "getwch", side_effect=keys), patch(
            "sys.stdout", new_callable=io.StringIO
        ):
            return _input_windows(history or [], "> ")

    def test_arrow_up_loads_history(self):
        self.assertEqual(self._run(["\x00", "H", "\r"], ["hello"]), "hello")

    def test_arrow_down_restores_draft(self):
        result = self._run(["x", "\x00", "H", "\x00", "P", "\r"], ["hello"])
        self.assertEqual(result, "x")

    def test_backspace_edits_line(self):
        self.assertEqual(self._run(["a", "b", "\x08", "\r"]), "a")

    def test_left_right_moves_cursor(self):
        # 输入 ab → ← → 插入 c → Enter
        self.assertEqual(self._run(["a", "b", "\x00", "K", "c", "\r"]), "acb")

    def test_display_width_counts_cjk_double(self):
        self.assertEqual(_display_width("我家猫咪叫五月"), 14)
        self.assertEqual(_display_width("/memory"), 7)
        self.assertEqual(_display_width("abc"), 3)

    def test_down_clears_wide_char_leftover(self):
        """从中文行切到 ASCII 行时按终端列宽补空格，避免残留字符（如 '/memory 叫五月'）"""
        import msvcrt

        keys = ["\x00", "H", "\x00", "H", "\x00", "P", "\x00", "P", "\r"]
        out = io.StringIO()
        with patch.object(msvcrt, "getwch", side_effect=keys), patch("sys.stdout", out):
            result = _input_windows(["/memory", "我家猫咪叫五月"], "> ")
        self.assertEqual(result, "")
        text = out.getvalue()
        self.assertIn("> /memory" + " " * 7 + "\r> /memory", text)
        self.assertIn("> " + " " * 14 + "\r> ", text)


class ReposCommandTest(unittest.TestCase):
    """/repos 命令：配置/索引后写入长期记忆，引导 LLM 用 search_code"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "memory.db")
        self.mem = MemoryManager(memory_path=self.path)

    def tearDown(self):
        self.tmp.cleanup()

    @patch("minimate.coderag.CodeRAGManager")
    def test_add_writes_memory_fact(self, mock_cls):
        mgr = mock_cls.return_value
        mgr.list_repos.return_value = {}
        result = _repos_command("add demo D:/tmp/proj", memory=self.mem)
        self.assertIn("已配置仓库", result)
        facts = self.mem.list_long_term()
        repo_facts = [
            f
            for f in facts
            if "代码仓库 demo" in f.content and "search_code" in f.content
        ]
        self.assertTrue(repo_facts)
        self.assertEqual(repo_facts[0].metadata.get("project"), "demo")

    @patch("minimate.coderag.CodeRAGManager")
    def test_index_replaces_memory_fact(self, mock_cls):
        mgr = mock_cls.return_value
        mgr.index.return_value = {"chunks": 10, "relations": 3, "db": "x"}
        _repos_command("index demo", memory=self.mem)
        _repos_command("index demo", memory=self.mem)
        facts = [
            f for f in self.mem.list_long_term() if "代码仓库 demo" in f.content
        ]
        self.assertEqual(len(facts), 1)
        self.assertIn("10 个代码块", facts[0].content)


if __name__ == "__main__":
    unittest.main()
