"""Knowledge RAG —— 文档知识库（SQLite + BM25/余弦混合检索）

与 Code RAG 统一存储与检索方案：
  - 存储：SQLite 持久化文档块与向量（JSON 数组），零外部向量库依赖
  - 检索：BM25 关键词召回 + 余弦向量召回，RRF 排名倒数融合

文档源：~/.minimate/knowledge/docs/（.md / .txt，可用 MINIMATE_KNOWLEDGE_DIR 覆盖根目录）
向量库：~/.minimate/knowledge/rag_db/knowledge.db
"""

from __future__ import annotations

import glob
import os
import time
from typing import Optional

from ..coderag.bm25 import BM25
from ..coderag.embeddings import EmbeddingProvider, cosine, get_provider
from .storage import KnowledgeStorage


class KnowledgeBase:
    """文档知识库：加载 docs 目录 → 分块 → embedding → SQLite 持久化 → 混合检索"""

    _instance: Optional["KnowledgeBase"] = None
    _current_repo: str = ""

    def __new__(cls, repo_dir: str = ""):
        if repo_dir and repo_dir != cls._current_repo:
            cls._instance = None
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, repo_dir: str = ""):
        if self._initialized:
            return
        self._initialized = True

        # 知识库根目录：默认 ~/.minimate/knowledge，可用 MINIMATE_KNOWLEDGE_DIR 覆盖
        knowledge_dir = os.getenv(
            "MINIMATE_KNOWLEDGE_DIR",
            os.path.join(os.path.expanduser("~"), ".minimate", "knowledge"),
        )
        self._repo_dir = repo_dir or os.path.join(knowledge_dir, "docs")
        db_dir = os.path.join(knowledge_dir, "rag_db")
        os.makedirs(self._repo_dir, exist_ok=True)
        os.makedirs(db_dir, exist_ok=True)
        type(self)._current_repo = self._repo_dir

        self._storage = KnowledgeStorage(os.path.join(db_dir, "knowledge.db"))
        self._provider = self._resolve_provider()
        self._doc_count = self._storage.count()

        # 空库且 docs 目录有文档时自动加载
        if self._doc_count == 0 and self._list_docs():
            self._load_repo()

    def _resolve_provider(self):
        """选择 embedding Provider：环境变量 > 索引模型标识 > Code RAG 默认

        查询与索引必须使用同一模型，否则余弦相似度失效。
        """
        kb_model = os.getenv("MINIMATE_KNOWLEDGE_EMBED_MODEL")
        if kb_model:
            return EmbeddingProvider(model_path=kb_model)
        stored = self._storage.get_embed_model()
        if stored:
            return EmbeddingProvider(model_path=stored)
        return get_provider()

    # ================================================================
    # 文档加载
    # ================================================================

    def _list_docs(self) -> list[str]:
        return sorted(
            glob.glob(os.path.join(self._repo_dir, "*.md"))
            + glob.glob(os.path.join(self._repo_dir, "*.txt"))
        )

    def _load_repo(self):
        """扫描 docs 目录，分块 + 向量化 + 写入 SQLite"""
        files = self._list_docs()
        if not files:
            return

        total_chunks = 0
        total_start = time.time()
        documents: list[dict] = []
        for fpath in files:
            fname = os.path.basename(fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                continue

            chunk_size = 1000 if len(content) > 100000 else 500
            chunks = self._chunk(content, chunk_size=chunk_size)
            if not chunks:
                continue
            for i, chunk in enumerate(chunks):
                documents.append({
                    "id": f"{fname}#{i}",
                    "source": fname,
                    "content": chunk,
                })
            total_chunks += len(chunks)

        if not documents:
            return

        print(f"  📄 正在 embedding {len(documents)} 个文档片段...")
        vec_start = time.time()
        vectors = self._provider.encode([d["content"] for d in documents])
        for doc, vec in zip(documents, vectors):
            doc["vector"] = vec
        self._storage.save_documents(
            documents,
            embed_model=getattr(self._provider, "model_path", ""),
        )
        self._doc_count = len(documents)
        elapsed = time.time() - vec_start
        print(f"     ✅ {elapsed:.0f} 秒")
        total_elapsed = time.time() - total_start
        print(f"  📚 知识库加载完成：{total_chunks} 个片段（共 {total_elapsed:.0f} 秒）")

    def _chunk(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        """按段落边界分块（保留前后重叠）"""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                nl = text.rfind("\n", start, end)
                if nl > start + chunk_size // 2:
                    end = nl
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    # ================================================================
    # 检索（BM25 + 余弦 RRF 混合）
    # ================================================================

    def query(self, question: str, top_k: int = 3) -> str:
        """混合检索，返回格式化结果（与旧 Chroma 接口一致）"""
        if self._doc_count == 0:
            return ""

        documents = self._storage.load_documents()
        if not documents:
            return ""

        texts = [d["content"] for d in documents]
        q_vec = self._provider.encode_one(question)
        bm25 = BM25(texts)
        bm_scores = bm25.score(question)
        cos_scores = [
            cosine(q_vec, d["vector"]) if d["vector"] else 0.0 for d in documents
        ]

        # RRF：BM25 与余弦两路召回按排名倒数融合（k=60）
        rrf: dict[int, float] = {}
        for scores in (bm_scores, cos_scores):
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, i in enumerate(order):
                if scores[i] > 0:
                    rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)

        top = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        output = []
        for i, rrf_score in top:
            source = documents[i]["source"]
            output.append(f"[{len(output) + 1}] 来源：{source}（相似度：{round(rrf_score, 3)}）")
            output.append(documents[i]["content"])
        return "\n\n".join(output)

    def count(self) -> int:
        return self._doc_count

    def rebuild(self):
        """强制重建索引：清空 SQLite 并重新加载 docs 目录"""
        self._storage.clear()
        self._doc_count = 0
        self._load_repo()


# ============================================================
# 全局实例
# ============================================================

_kb: Optional[KnowledgeBase] = None


def get_knowledge_base(repo_dir: str = "") -> KnowledgeBase:
    global _kb
    _kb = KnowledgeBase(repo_dir=repo_dir)
    return _kb
