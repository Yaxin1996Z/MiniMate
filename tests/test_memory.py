"""Memory 系统单元测试：淘汰/压缩/去重/检索/持久化"""

import os
import tempfile
import unittest
from unittest.mock import patch

from minimate.memory import (
    LongTermMemory,
    MapReduceCompressor,
    MemoryItem,
    MemoryManager,
    ShortTermMemory,
    estimate_tokens,
)


class EstimateTokensTest(unittest.TestCase):
    def test_chinese_counts_per_char(self):
        self.assertGreater(estimate_tokens("你好世界"), 3)

    def test_english_four_chars(self):
        self.assertEqual(estimate_tokens("abcdefgh"), 2 + 1)


class ShortTermMemoryTest(unittest.TestCase):
    def test_eviction_over_budget(self):
        """超 token 硬预算自动淘汰最旧条目（滑动窗口）"""
        mem = ShortTermMemory(token_budget=30)
        for i in range(10):
            mem.add(MemoryItem(type="conversation", role="user", content=f"消息内容{i} 一些字数"))
        self.assertLessEqual(mem.get_token_count(), 30)
        self.assertGreater(len(mem.items), 1)

    def test_get_context_recent_first(self):
        mem = ShortTermMemory(token_budget=1000)
        mem.add(MemoryItem(type="conversation", role="user", content="第一轮"))
        mem.add(MemoryItem(type="conversation", role="assistant", content="回答"))
        ctx = mem.get_context()
        self.assertIn("第一轮", ctx)
        self.assertIn("回答", ctx)


class LongTermMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "memory.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_deduplicate(self):
        mem = LongTermMemory(path=self.path)
        self.assertTrue(mem.add_fact("用户偏好：喜欢 Python"))
        self.assertFalse(mem.add_fact("用户偏好：喜欢 Python"))  # 重复跳过
        self.assertEqual(len(mem.items), 1)

    def test_keyword_search(self):
        mem = LongTermMemory(path=self.path)
        mem.add_fact("用户偏好：喜欢 Python 和 FastAPI", keywords=["Python", "FastAPI"])
        mem.add_fact("项目信息：MiniMate 使用 DeepSeek", keywords=["MiniMate", "DeepSeek"])
        results = mem.search("Python")
        self.assertEqual(len(results), 1)
        self.assertIn("Python", results[0].content)

    def test_persist_and_load(self):
        mem = LongTermMemory(path=self.path)
        mem.add_fact("跨会话事实：目标岗位 AI Agent 开发")
        mem.save()

        mem2 = LongTermMemory(path=self.path)
        self.assertEqual(len(mem2.items), 1)
        self.assertIn("AI Agent", mem2.items[0].content)


class CompressorTest(unittest.TestCase):
    @patch("minimate.memory.compressor.llm.call")
    def test_map_reduce_keeps_recent(self, mock_call):
        """Map-Reduce 压缩：分片摘要 + 合并；保留最近 3 轮不压缩"""
        mock_call.return_value = "局部摘要"
        comp = MapReduceCompressor(chunk_chars=100, keep_recent_rounds=1)
        items = [
            MemoryItem(type="conversation", role="user", content=f"旧消息内容 {i} 很长" * 3)
            for i in range(8)
        ]
        summary = comp.compress(items)
        self.assertEqual(summary, "局部摘要")
        self.assertGreaterEqual(mock_call.call_count, 1)

    def test_compress_empty(self):
        comp = MapReduceCompressor()
        self.assertEqual(comp.compress([]), "")

    def test_all_recent_kept(self):
        """条目少于保留轮数时不压缩"""
        comp = MapReduceCompressor(keep_recent_rounds=3)
        items = [MemoryItem(type="conversation", role="user", content=f"m{i}") for i in range(3)]
        with patch("minimate.memory.compressor.llm.call") as mock_call:
            self.assertEqual(comp.compress(items), "")
        mock_call.assert_not_called()


class MemoryManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "memory.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_four_types_and_compat(self):
        mem = MemoryManager(memory_path=self.path)
        mem.add_user_message("我叫小明，喜欢Python")
        mem.add_ai_message("好的")
        mem.add_fact("项目信息：MiniMate")
        mem.add_tool_result("工具执行成功")
        mem.add_summary("历史摘要内容")
        self.assertGreater(mem.stats()["short_term_items"], 0)
        self.assertGreaterEqual(mem.stats()["long_term_facts"], 1)

    def test_auto_extract_fact(self):
        mem = MemoryManager(memory_path=self.path)
        mem.add_user_message("我叫小明，喜欢Python开发")
        facts = mem.long_term.items
        self.assertTrue(any("用户偏好" in f.content for f in facts))

    def test_save_load_facts(self):
        mem = MemoryManager(memory_path=self.path)
        mem.add_fact("用户偏好：工作地点上海")
        mem.save()

        mem2 = MemoryManager(memory_path=self.path)
        self.assertTrue(any("上海" in f.content for f in mem2.long_term.items))

    def test_context_budget(self):
        mem = MemoryManager(memory_path=self.path, context_budget=100)
        for i in range(5):
            mem.add_user_message(f"第 {i} 轮对话内容信息")
            mem.add_ai_message(f"第 {i} 轮回答")
        ctx = mem.get_context()
        self.assertLessEqual(estimate_tokens(ctx), 100 * 2)  # 有节流余量


class ProjectScopeTest(unittest.TestCase):
    """P0：项目作用域隔离 + 相关度评分"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "memory.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_project_scope_isolation(self):
        mem = LongTermMemory(path=self.path)
        mem.add_fact("项目A事实：使用Python", project="projA")
        mem.add_fact("项目B事实：使用Java", project="projB")
        mem.add_fact("全局事实：用户在上海", scope="global")
        self.assertEqual(len(mem.get_visible("projA")), 2)  # A + global
        self.assertEqual(len(mem.get_visible("projB")), 2)  # B + global

    def test_search_filter_by_project(self):
        mem = LongTermMemory(path=self.path)
        mem.add_fact("项目A：FastAPI 服务", project="projA")
        mem.add_fact("项目B：Spring Cloud", project="projB")
        results = mem.search("FastAPI", 5, project="projA")
        self.assertEqual(len(results), 1)
        self.assertIn("FastAPI", results[0].content)

    def test_time_decay_scoring(self):
        from minimate.memory import MemoryItem, score_relevance
        import time
        # 用非精确匹配（多词查询部分命中）触发时间衰减路径
        old = MemoryItem(type="fact", content="旧事实：Python 语言", timestamp=time.time() - 48 * 3600)
        new = MemoryItem(type="fact", content="新事实：Python 语言", timestamp=time.time())
        self.assertGreater(score_relevance(new, "Python 开发"), score_relevance(old, "Python 开发"))


class CompressionTriggerTest(unittest.TestCase):
    """P1：trigger_ratio 触发压缩 + 降级"""

    def test_compress_old_injects_summary(self):
        class FakeCompressor:
            def compress(self, items):
                return "历史摘要内容"

        mem = ShortTermMemory(token_budget=300, trigger_ratio=0.5)
        for i in range(10):
            mem.add(MemoryItem(type="conversation", role="user", content=f"第{i}轮长内容信息" * 3))
        self.assertTrue(mem.needs_compression())
        self.assertTrue(mem.compress_old(FakeCompressor(), keep_recent_rounds=1))
        # 注入摘要 + 保留最近 1 轮（2 条）
        self.assertLessEqual(len(mem.items), 3)
        self.assertTrue(any(i.type == "summary" for i in mem.items))

    def test_compressor_fallback_on_llm_failure(self):
        from minimate.memory.compressor import MapReduceCompressor
        with patch("minimate.memory.compressor.llm.call", return_value="[API 错误] 网络失败"):
            comp = MapReduceCompressor(chunk_chars=50)
            items = [MemoryItem(type="conversation", role="user", content=f"内容{i}很多字" * 5) for i in range(12)]
            summary = comp.compress(items)
        self.assertTrue(summary)  # 降级仍返回摘要（截取/拼接）


class ExplicitMemoryTest(unittest.TestCase):
    """P1：显式记忆指令"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "memory.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_explicit_remember_global(self):
        mem = MemoryManager(memory_path=self.path)
        mem.add_user_message("记住我的工作地点在上海")
        facts = mem.long_term.items
        self.assertTrue(any("上海" in f.content and f.metadata.get("scope") == "global" for f in facts))


if __name__ == "__main__":
    unittest.main()
