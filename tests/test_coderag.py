"""Code RAG 单元测试：向量/分块/关系/存储/检索/调用链"""

import os
import tempfile
import unittest
from unittest.mock import patch

from minimate.coderag import (
    CodeRAGManager,
    cosine,
    embed,
    index_directory,
    index_file,
)
from minimate.coderag.embeddings import HashEmbeddingProvider, tokenize
from minimate.coderag.bm25 import BM25
from minimate.coderag.storage import SQLiteCodeStorage
from minimate.coderag.retriever import CodeRetriever


def _sample_project(root: str):
    os.makedirs(os.path.join(root, "pkg"), exist_ok=True)
    with open(os.path.join(root, "pkg", "base.py"), "w", encoding="utf-8") as f:
        f.write(
            '"""基础服务模块"""\n'
            "class BaseService:\n"
            "    def run(self):\n"
            "        return 'running'\n"
            "\n"
            "class Service(BaseService):\n"
            "    def process(self, data):\n"
            "        return self.run()\n"
        )
    with open(os.path.join(root, "pkg", "app.py"), "w", encoding="utf-8") as f:
        f.write(
            "from pkg.base import Service\n"
            "\n"
            "def main():\n"
            "    s = Service()\n"
            "    return s.process('x')\n"
        )


class EmbeddingsTest(unittest.TestCase):
    def test_embed_stable(self):
        self.assertEqual(embed("hello world"), embed("hello world"))

    def test_cosine_similar(self):
        a = embed("class Service process data")
        b = embed("class Service process data")
        c = embed("def unrelated_function")
        self.assertGreater(cosine(a, b), 0.9)
        self.assertLess(cosine(a, c), 0.3)

    def test_tokenize_camel_case(self):
        tokens = tokenize("myFunctionName")
        self.assertIn("function", tokens)
        self.assertIn("name", tokens)


class IndexerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _sample_project(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_chunks_and_relations(self):
        chunks, relations = index_directory(self.tmp.name)
        # 文件块 + 类块 + 方法块
        granularities = {c.granularity for c in chunks}
        self.assertEqual(granularities, {"file", "class", "method"})
        names = {c.name for c in chunks}
        self.assertIn("Service", names)
        self.assertIn("Service.process", names)

        rel_types = {r.relation for r in relations}
        self.assertTrue({"extends", "imports", "calls", "contains"} <= rel_types)

        # extends: Service -> BaseService
        extends = [r for r in relations if r.relation == "extends"]
        self.assertTrue(any(r.from_name == "Service" and r.to_name == "BaseService" for r in extends))


class StorageRetrieverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _sample_project(os.path.join(self.tmp.name, "proj"))
        self.db = os.path.join(self.tmp.name, "rag", "demo.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_store_load_and_search(self):
        from minimate.coderag.embeddings import embed as do_embed
        chunks, relations = index_directory(os.path.join(self.tmp.name, "proj"))
        for c in chunks:
            c.vector = do_embed(c.search_text)
        storage = SQLiteCodeStorage(self.db)
        storage.save_repo("demo", chunks, relations, {"source": "x"})

        retriever = CodeRetriever(storage)
        results = retriever.search("demo", "Service 处理数据", top_k=3)
        self.assertTrue(results)
        self.assertIn("score", results[0])

        # 调用链：Service.process 调用了 run
        chain = retriever.call_chain("demo", "Service.process")
        callees = chain.get("callees", [])
        self.assertTrue(any("run" in p for p in callees))


class ManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        _sample_project(os.path.join(self.tmp.name, "proj"))
        self.rag = os.path.join(self.tmp.name, "rag")

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_index_search(self):
        mgr = CodeRAGManager(
            rag_dir=self.rag,
            repos_dir=os.path.join(self.tmp.name, "repos"),
            provider=HashEmbeddingProvider(),
        )
        mgr.add_repo("demo", os.path.join(self.tmp.name, "proj"))
        info = mgr.index("demo")
        self.assertGreater(info["chunks"], 0)
        self.assertGreater(info["relations"], 0)
        self.assertTrue(os.path.exists(info["db"]))

        results = mgr.search("demo", "Service 处理数据")
        self.assertTrue(results)

        # 持久化后重新加载（跨进程向量稳定性）
        mgr2 = CodeRAGManager(
            rag_dir=self.rag,
            repos_dir=os.path.join(self.tmp.name, "repos"),
            provider=HashEmbeddingProvider(),
        )
        results2 = mgr2.search("demo", "Service 处理数据")
        self.assertTrue(results2)

        # 混合检索字段
        self.assertIn("bm25", results[0])
        self.assertIn("cosine", results[0])

    def test_bm25_scores_identifier_docs(self):
        """BM25 对包含查询标识符的文档给更高分"""
        docs = [
            "class BaseService: def run(self): pass",
            "class Service(BaseService): def process(self, data): return self.run()",
            "def main(): print('hello')",
        ]
        bm = BM25(docs)
        scores = bm.score("Service process")
        self.assertGreater(scores[1], scores[0])
        self.assertGreater(scores[1], scores[2])

    def test_update_reindex_local(self):
        """update：本地目录重新扫描并重建索引，新增代码可被检索"""
        mgr = CodeRAGManager(
            rag_dir=self.rag,
            repos_dir=os.path.join(self.tmp.name, "repos"),
            provider=HashEmbeddingProvider(),
        )
        mgr.add_repo("demo", os.path.join(self.tmp.name, "proj"))
        info1 = mgr.index("demo")

        with open(
            os.path.join(self.tmp.name, "proj", "pkg", "extra.py"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write("def helper():\n    return 'extra'\n")

        info2 = mgr.update("demo")
        self.assertGreaterEqual(info2["chunks"], info1["chunks"])
        self.assertTrue(mgr.search("demo", "helper"))

    @patch("minimate.coderag.manager.subprocess.run", side_effect=FileNotFoundError)
    def test_search_keyword_fallback_scan(self, mock_run):
        """rg 不可用时回退逐行扫描，仍能精确关键词定位"""
        mgr = CodeRAGManager(
            rag_dir=self.rag,
            repos_dir=os.path.join(self.tmp.name, "repos"),
            provider=HashEmbeddingProvider(),
        )
        mgr.add_repo("demo", os.path.join(self.tmp.name, "proj"))
        hits = mgr.search_keyword("demo", "process")
        self.assertTrue(hits)
        self.assertEqual(hits[0]["granularity"], "keyword")

    @patch("minimate.coderag.manager.subprocess.run", side_effect=FileNotFoundError)
    def test_search_two_stage_identifier_first(self, mock_run):
        """标识符查询：rg 关键词命中排在最前，再补语义召回"""
        mgr = CodeRAGManager(
            rag_dir=self.rag,
            repos_dir=os.path.join(self.tmp.name, "repos"),
            provider=HashEmbeddingProvider(),
        )
        mgr.add_repo("demo", os.path.join(self.tmp.name, "proj"))
        mgr.index("demo")
        results = mgr.search("demo", "process")
        self.assertTrue(results)
        self.assertEqual(results[0]["granularity"], "keyword")

    @patch("minimate.coderag.manager.subprocess.run")
    def test_update_pull_failure_still_reindexes(self, mock_run):
        """git pull 失败（网络不可达）不中断，降级为使用现有代码重建索引"""
        repo_dir = os.path.join(self.tmp.name, "repos", "demo")
        _sample_project(repo_dir)  # 模拟已 clone 的本地目录
        fake_result = type(
            "R", (), {"returncode": 1, "stderr": "Failed to connect", "stdout": ""}
        )()
        mock_run.return_value = fake_result
        mgr = CodeRAGManager(
            rag_dir=self.rag,
            repos_dir=os.path.join(self.tmp.name, "repos"),
            provider=HashEmbeddingProvider(),
        )
        mgr.add_repo("demo", "https://github.com/example/demo.git")
        info = mgr.update("demo")
        self.assertGreater(info["chunks"], 0)


if __name__ == "__main__":
    unittest.main()
