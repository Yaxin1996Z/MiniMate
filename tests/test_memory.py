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

    def test_deduplicate_normalized(self):
        """标点/大小写差异视为重复：'喜欢 Python。' 与 '喜欢Python' 只存一条"""
        mem = LongTermMemory(path=self.path)
        self.assertTrue(mem.add_fact("用户偏好：喜欢 Python。"))
        self.assertFalse(mem.add_fact("用户偏好：喜欢Python"))
        self.assertEqual(len(mem.items), 1)

    def test_deduplicate_cross_prefix(self):
        """不同前缀但归一化后相同视为重复：显式/自动提取的同一条事实只存一条"""
        mem = LongTermMemory(path=self.path)
        self.assertTrue(mem.add_fact("用户偏好：五月"))
        self.assertFalse(mem.add_fact("用户偏好（显式）：五月"))
        self.assertEqual(len(mem.items), 1)

    def test_existing_rows_normalized_on_load(self):
        """存量数据按原始哈希入库后，新实例启动时重哈希并合并重复"""
        import hashlib
        import sqlite3

        mem = LongTermMemory(path=self.path)
        conn = sqlite3.connect(self.path)
        for content in ("用户偏好：五月", "用户偏好（显式）：五月"):
            raw_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            conn.execute(
                """
                INSERT INTO memories
                    (scope, content, created_at, kind, source, updated_at, content_hash)
                VALUES ('', ?, '2026-01-01T00:00:00+00:00', 'fact', 'manual', '', ?)
                """,
                (content, raw_hash),
            )
        conn.commit()
        conn.close()

        mem2 = LongTermMemory(path=self.path)  # 触发归一化重哈希 + 去重
        self.assertEqual(len(mem2.items), 1)

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

    def test_sqlite_schema_and_access_count(self):
        """SQLite 表结构对齐 PaiCLI memories 设计；检索命中递增 access_count"""
        import sqlite3

        mem = LongTermMemory(path=self.path)
        self.assertTrue(mem.add_fact("用户家的猫咪名叫五月", source="agent"))

        conn = sqlite3.connect(self.path)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
            expected = {
                "id", "scope", "content", "created_at", "kind", "source",
                "importance", "confidence", "updated_at", "expires_at",
                "access_count", "content_hash",
            }
            self.assertTrue(expected.issubset(cols))
            row = conn.execute(
                "SELECT scope, source, content_hash FROM memories"
            ).fetchone()
            self.assertEqual(row[0], "")      # 无项目时为 project 作用域（空串）
            self.assertEqual(row[1], "agent")
            self.assertEqual(len(row[2]), 64)  # sha256 hex
        finally:
            conn.close()

        mem.search("五月")  # 命中一次
        conn = sqlite3.connect(self.path)
        try:
            self.assertEqual(
                conn.execute("SELECT access_count FROM memories").fetchone()[0], 1
            )
        finally:
            conn.close()


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

    def test_auto_extract_fact_with_owner(self):
        """带所有格/主语的句子也能提取，且保留上下文（小猫咪叫五月）"""
        mem = MemoryManager(memory_path=self.path)
        mem.add_user_message("我家小猫咪叫五月")
        facts = mem.long_term.items
        self.assertTrue(any("小猫咪叫五月" in f.content for f in facts))

    def test_question_not_extracted_as_fact(self):
        """疑问句不提取：'我家猫咪叫什么' 不应入库成为事实"""
        mem = MemoryManager(memory_path=self.path)
        mem.add_user_message("我家猫咪叫什么")
        self.assertFalse(any("什么" in f.content for f in mem.long_term.items))

    def test_ai_reply_not_auto_extracted(self):
        """助手复述/反问不自动提取事实"""
        mem = MemoryManager(memory_path=self.path)
        mem.add_ai_message("好的，我喜欢Python和FastAPI。")
        self.assertEqual(len(mem.long_term.items), 0)

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

    def test_llm_fact_extraction_on_compress(self):
        """压缩触发时，用 LLM 从旧条目提取事实写入长期记忆（正则之外的补充通道）"""
        mem = MemoryManager(memory_path=self.path, token_budget=600, trigger_ratio=0.5)

        def fake_llm_call(prompt, system="", temperature=0):
            return (
                "用户偏好：我每天工作到十点\n"
                "帮我写一个二分查找脚本\n"      # 临时请求，应被启发式过滤
                "项目技术栈：FastAPI"
            )

        with patch("minimate.memory.manager.llm.call", side_effect=fake_llm_call):
            for i in range(10):
                mem.add_user_message(f"第{i}轮：讨论技术方案细节内容信息" * 3)
                mem.add_ai_message(f"第{i}轮回答：确认方案可行" * 3)

        self.assertTrue(
            any("工作到十点" in f.content for f in mem.long_term.items)
        )
        self.assertTrue(
            any("FastAPI" in f.content for f in mem.long_term.items)
        )
        self.assertFalse(
            any("二分查找" in f.content for f in mem.long_term.items)
        )


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

    def test_chinese_bigram_scoring(self):
        """中文双字切分：'我家猫咪叫什么' 能命中 '我家猫咪叫五月'"""
        from minimate.memory import MemoryItem, score_relevance

        item = MemoryItem(type="fact", content="用户偏好：我家猫咪叫五月")
        self.assertGreater(score_relevance(item, "我家猫咪叫什么"), 0)


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

    def test_explicit_remember_with_le(self):
        """"记住了X"也能正确提取，不把"了"吞进事实内容"""
        mem = MemoryManager(memory_path=self.path)
        mem.add_user_message("记住了MiniMate项目的完整信息")
        facts = mem.long_term.items
        self.assertTrue(
            any(
                "MiniMate项目的完整信息" in f.content
                and "了MiniMate" not in f.content
                for f in facts
            )
        )

    def test_ai_reply_not_treated_as_remember_command(self):
        """助手复述"我记住了"不应被当成显式记忆指令"""
        mem = MemoryManager(memory_path=self.path)
        mem.add_user_message("我家猫咪叫五月")
        mem.add_ai_message("好的，我记住了！你家猫咪叫五月，名字真好听。")
        facts = mem.long_term.items
        self.assertFalse(any("（显式）" in f.content for f in facts))


if __name__ == "__main__":
    unittest.main()
