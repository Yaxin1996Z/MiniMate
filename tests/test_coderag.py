"""Code RAG 单元测试：向量/分块/关系/存储/检索/调用链"""

import os
import tempfile
import unittest

from minimate.coderag import (
    CodeRAGManager,
    cosine,
    embed,
    index_directory,
    index_file,
)
from minimate.coderag.embeddings import tokenize
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
        mgr = CodeRAGManager(rag_dir=self.rag, repos_dir=os.path.join(self.tmp.name, "repos"))
        mgr.add_repo("demo", os.path.join(self.tmp.name, "proj"))
        info = mgr.index("demo")
        self.assertGreater(info["chunks"], 0)
        self.assertGreater(info["relations"], 0)
        self.assertTrue(os.path.exists(info["db"]))

        results = mgr.search("demo", "Service 处理数据")
        self.assertTrue(results)

        # 持久化后重新加载（跨进程向量稳定性）
        mgr2 = CodeRAGManager(rag_dir=self.rag, repos_dir=os.path.join(self.tmp.name, "repos"))
        results2 = mgr2.search("demo", "Service 处理数据")
        self.assertTrue(results2)


if __name__ == "__main__":
    unittest.main()
