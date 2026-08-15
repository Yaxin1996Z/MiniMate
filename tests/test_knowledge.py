"""Knowledge RAG（SQLite + 混合检索）单元测试 —— mock embedding provider"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from minimate.coderag.embeddings import HashEmbeddingProvider
from minimate.knowledgerag.knowledge import KnowledgeBase
from minimate.knowledgerag.storage import KnowledgeStorage


class KnowledgeStorageTest(unittest.TestCase):
    """SQLite 文档存储"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="kb_storage_")
        self.db = os.path.join(self.tmp, "knowledge.db")
        self.storage = KnowledgeStorage(self.db)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_save_load_count(self):
        docs = [
            {"id": "a.md#0", "source": "a.md", "content": "你好 Java", "vector": [1.0, 0.0]},
            {"id": "b.md#0", "source": "b.md", "content": "hello python", "vector": [0.0, 1.0]},
        ]
        self.storage.save_documents(docs, embed_model="test-model")
        self.assertEqual(self.storage.count(), 2)
        loaded = self.storage.load_documents()
        self.assertEqual(loaded[0]["vector"], [1.0, 0.0])
        self.assertEqual(self.storage.get_embed_model(), "test-model")
        self.storage.clear()
        self.assertEqual(self.storage.count(), 0)
        self.assertEqual(self.storage.get_embed_model(), "")


class KnowledgeBaseTest(unittest.TestCase):
    """文档加载 + 混合检索"""

    def setUp(self):
        # 重置单例，避免测试间复用旧实例（临时目录已被删除）
        KnowledgeBase._instance = None
        KnowledgeBase._current_repo = ""
        self.tmp = tempfile.mkdtemp(prefix="kb_base_")
        self.docs = os.path.join(self.tmp, "docs")
        os.makedirs(self.docs, exist_ok=True)
        with open(os.path.join(self.docs, "java.md"), "w", encoding="utf-8") as f:
            f.write("# Java 规范\nJava 缩进使用 2 个空格。\n" * 20)
        with open(os.path.join(self.docs, "python.md"), "w", encoding="utf-8") as f:
            f.write("# Python 规范\nPython 变量使用小写加下划线。\n" * 20)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmp, ignore_errors=True)

    def _kb(self):
        with patch.dict(
            os.environ,
            {"MINIMATE_KNOWLEDGE_DIR": self.tmp},
            clear=False,
        ), patch(
            "minimate.knowledgerag.knowledge.get_provider",
            return_value=HashEmbeddingProvider(),
        ):
            return KnowledgeBase()

    def test_auto_load_and_count(self):
        kb = self._kb()
        self.assertGreater(kb.count(), 0)

    def test_query_returns_source(self):
        kb = self._kb()
        result = kb.query("Java 缩进", top_k=2)
        self.assertIn("java.md", result)
        self.assertIn("相似度", result)

    def test_rebuild(self):
        kb = self._kb()
        before = kb.count()
        kb.rebuild()
        self.assertEqual(kb.count(), before)

    def test_chunk_overlap(self):
        kb = self._kb()
        text = "a" * 1000 + "\n" + "b" * 1000
        chunks = kb._chunk(text, chunk_size=500, overlap=50)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(c) <= 550 for c in chunks))


if __name__ == "__main__":
    unittest.main()
