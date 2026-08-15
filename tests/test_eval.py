"""Agent 评测系统单元测试 —— 确定性 checker + 运行器流程（mock LLM）"""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from minimate.eval import EvalCase, EvalRunner
from minimate.eval.checkers import run_checker
from minimate.eval.suites_ai import get_suite, list_suites
from minimate.eval.suites_real import REAL_SUITE, get_suite as get_real_suite, list_suites as list_real_suites


class CheckerTest(unittest.TestCase):
    """确定性断言检查器"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="eval_checker_")
        with open(os.path.join(self.tmp, "hello.txt"), "w", encoding="utf-8") as f:
            f.write("你好，MiniMate！")
        os.makedirs(os.path.join(self.tmp, "project"), exist_ok=True)
        with open(os.path.join(self.tmp, "project", "main.py"), "w", encoding="utf-8") as f:
            f.write("print('main')")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_file_exists(self):
        ok, reason = run_checker("file_exists", "hello.txt", cwd=self.tmp)
        self.assertTrue(ok)
        ok, _ = run_checker("file_exists", "missing.txt", cwd=self.tmp)
        self.assertFalse(ok)

    def test_file_contains(self):
        ok, _ = run_checker("file_contains", "hello.txt|你好", cwd=self.tmp)
        self.assertTrue(ok)
        ok, _ = run_checker("file_contains", "hello.txt|再见", cwd=self.tmp)
        self.assertFalse(ok)

    def test_dir_contains(self):
        ok, _ = run_checker("dir_contains", "project|main.py", cwd=self.tmp)
        self.assertTrue(ok)
        ok, _ = run_checker("dir_contains", "project|missing.py", cwd=self.tmp)
        self.assertFalse(ok)

    def test_output_contains(self):
        ok, _ = run_checker("output_contains", "MiniMate", output="这是 MiniMate 的评测")
        self.assertTrue(ok)
        ok, _ = run_checker("output_contains", "不存在的词", output="测试")
        self.assertFalse(ok)

    def test_math_answer(self):
        ok, reason = run_checker("math_answer", "57", output="计算结果是：57")
        self.assertTrue(ok)
        ok, _ = run_checker("math_answer", "57", output="计算结果是：42")
        self.assertFalse(ok)

    def test_command_exit0(self):
        ok, _ = run_checker("command_exit0", "python -c \"print('ok')\"", cwd=self.tmp)
        self.assertTrue(ok)
        ok, _ = run_checker("command_exit0", "python -c \"raise SystemExit(1)\"", cwd=self.tmp)
        self.assertFalse(ok)

    def test_grep_finds(self):
        ok, reason = run_checker("grep_finds", "project|print", cwd=self.tmp)
        self.assertTrue(ok)
        ok, _ = run_checker("grep_finds", "project|不存在的关键字", cwd=self.tmp)
        self.assertFalse(ok)

    def test_unknown_checker(self):
        ok, _ = run_checker("no_such_checker", "x")
        self.assertFalse(ok)


class SuiteTest(unittest.TestCase):
    """内置评测集"""

    def test_list_and_get(self):
        self.assertIn("basic", list_suites())
        cases = get_suite("basic")
        self.assertGreaterEqual(len(cases), 12)
        modes = {c.mode for c in cases}
        self.assertEqual(modes, {"chat", "react", "plan", "multi"})

    def test_case_ids_unique(self):
        cases = get_suite("basic")
        ids = [c.id for c in cases]
        self.assertEqual(len(ids), len(set(ids)))


class RealSuiteTest(unittest.TestCase):
    """真实用户用例集"""

    def test_real_suite_structure(self):
        self.assertIn("real", list_real_suites())
        cases = get_real_suite("real")
        self.assertEqual(len(cases), 1)
        case = cases[0]
        self.assertEqual(case.id, "real_multi_001")
        self.assertEqual(case.mode, "multi")
        self.assertEqual(case.checker, "command_exit0")
        self.assertIn("结构", case.prompt)
        self.assertIn("minimate.md", case.prompt)

    def test_runner_loads_real_suite(self):
        with tempfile.TemporaryDirectory(prefix="eval_out_") as out_dir:
            with patch("minimate.eval.runner.Agent") as mock_agent_cls, patch(
                "minimate.eval.runner.MultiAgentOrchestrator"
            ) as mock_orch:
                mock_orch.return_value.run.return_value = "已生成文档"
                runner = EvalRunner(results_dir=out_dir)
                results, summary, _ = runner.run_suite("real")
            self.assertEqual(summary.total, 1)
            self.assertEqual(results[0].case.id, "real_multi_001")


class RunnerTest(unittest.TestCase):
    """运行器流程：沙箱隔离 + 判定 + 统计 + 报告"""

    def test_runner_with_mocked_agent(self):
        with tempfile.TemporaryDirectory(prefix="eval_out_") as out_dir:
            mock_agent = MagicMock()
            mock_agent.run.return_value = "答案是 42"
            mock_orch = MagicMock()
            mock_orch.run.return_value = "mock multi output"
            with patch("minimate.eval.runner.Agent", return_value=mock_agent), patch(
                "minimate.eval.runner.MultiAgentOrchestrator", return_value=mock_orch
            ):
                runner = EvalRunner(results_dir=out_dir)
                results, summary, paths = runner.run_suite("basic")

            self.assertEqual(summary.total, len(get_suite("basic")))
            self.assertEqual(summary.by_mode["chat"]["total"], 4)
            self.assertEqual(summary.by_mode["multi"]["total"], 3)
            # mock 输出 "答案是 42"：仅 react_003（输出含 42）通过
            self.assertEqual(summary.passed, 1)
            self.assertAlmostEqual(summary.pass_rate, 1 / summary.total)
            # 报告文件已生成
            self.assertTrue(os.path.isfile(paths["markdown"]))
            self.assertTrue(os.path.isfile(paths["json"]))
            with open(paths["markdown"], encoding="utf-8") as f:
                md = f.read()
            self.assertIn("逐条明细", md)
            self.assertIn("react_003", md)

    def test_mode_filter_and_max_cases(self):
        with tempfile.TemporaryDirectory(prefix="eval_out_") as out_dir:
            with patch("minimate.eval.runner.Agent") as mock_agent_cls:
                mock_agent_cls.return_value.run.return_value = "x"
                runner = EvalRunner(
                    mode_filter={"chat"},
                    max_cases=2,
                    results_dir=out_dir,
                )
                results, summary, _ = runner.run_suite("basic")
            self.assertEqual(summary.total, 2)
            self.assertTrue(all(r.case.mode == "chat" for r in results))

    def test_case_id_filter(self):
        """case_ids 只运行指定用例"""
        with tempfile.TemporaryDirectory(prefix="eval_out_") as out_dir:
            with patch("minimate.eval.runner.Agent") as mock_agent_cls:
                mock_agent_cls.return_value.run.return_value = "x"
                runner = EvalRunner(
                    case_ids={"react_006", "multi_002"},
                    results_dir=out_dir,
                )
                results, summary, _ = runner.run_suite("basic")
            self.assertEqual(summary.total, 2)
            self.assertEqual(
                {r.case.id for r in results}, {"react_006", "multi_002"}
            )

    def test_setup_writes_files(self):
        """setup 的内置 write 操作应真实落盘到沙箱"""
        case = EvalCase(
            id="t_001",
            name="setup 测试",
            mode="chat",
            prompt="忽略",
            checker="file_contains",
            expected="data.txt|秘密",
            setup=[{"write": {"path": "data.txt", "content": "秘密数字是 42"}}],
        )
        with tempfile.TemporaryDirectory(prefix="eval_sb_") as sb:
            old = os.getcwd()
            try:
                os.chdir(sb)
                from minimate.eval.runner import EvalRunner

                runner = EvalRunner(results_dir=sb)
                runner._prepare(case)
                with open("data.txt", encoding="utf-8") as f:
                    self.assertIn("秘密", f.read())
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
