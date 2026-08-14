"""
知识库 —— 基于 Chroma + bge 本地 embedding 的 RAG 检索系统

启动时自动加载 repo/ 下文档 → 分块 → embedding → 存入 Chroma（持久化）。
运行时只做检索，不写库。Chroma 数据落盘到 rag_db/，重复启动直接复用。
"""

import os
import glob
import tomllib
from typing import Optional

import chromadb


# ============================================================
# 配置加载
# ============================================================

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")


def _load_config() -> dict:
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)["rag"]


# ============================================================
# Embedding 函数：统一用 bge 模型（本地和容器一致）
# ============================================================

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ============================================================
# 知识库
# ============================================================

class KnowledgeBase:
    """本地文档知识库，基于 Chroma + ONNX embedding"""

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

        cfg = _load_config()
        self._repo_dir = repo_dir or os.path.join(
            os.path.dirname(__file__), cfg["repo_dir"]
        )
        db_dir = os.path.join(os.path.dirname(__file__), cfg["db_dir"])
        embed_path = cfg["embed_model_path"]

        # 环境变量可覆盖模型路径（便于切换模型 / 容器挂载）
        embed_path = os.getenv("MINIMATE_EMBED_MODEL", embed_path)
        if not os.path.exists(embed_path):
            print(f"  ⚠️ 模型路径不存在：{embed_path}")

        os.makedirs(self._repo_dir, exist_ok=True)
        os.makedirs(db_dir, exist_ok=True)

        type(self)._current_repo = self._repo_dir

        # 快速路径：repo 目录无文档时不加载 embedding 模型与 Chroma，
        # 避免空知识库场景下每次启动都要等待模型加载（几十秒）
        self._embed_func = None
        self._client = None
        self._collection = None
        self._doc_count = 0
        if not glob.glob(os.path.join(self._repo_dir, "*.md")) and \
           not glob.glob(os.path.join(self._repo_dir, "*.txt")):
            return

        self._embed_func = SentenceTransformerEmbeddingFunction(model_name=embed_path)

        # 初始化 Chroma
        self._client = chromadb.PersistentClient(
            path=db_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name="research_knowledge",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed_func,
        )

        # 如果是空库，自动加载 repo 文档
        self._doc_count = self._collection.count()
        if self._doc_count == 0:
            self._load_repo()

    # ================================================================
    # 文档加载（首次启动时自动执行）
    # ================================================================

    def _load_repo(self):
        """扫描 repo 目录，加载所有 .md / .txt 文件"""
        import time
        files = glob.glob(os.path.join(self._repo_dir, "*.md")) + \
                glob.glob(os.path.join(self._repo_dir, "*.txt"))
        if not files:
            return

        total_chunks = 0
        total_start = time.time()
        for fpath in files:
            fname = os.path.basename(fpath)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            if not content.strip():
                continue

            # 大文件用更大的 chunk 减少数量
            chunk_size = 1000 if len(content) > 100000 else 500
            chunks = self._chunk(content, chunk_size=chunk_size)
            if not chunks:
                continue

            print(f"  📄 {fname}: {len(chunks)} 个片段，正在 embedding...")
            chunk_start = time.time()

            ids = [f"doc_{self._doc_count + i}" for i in range(len(chunks))]
            metadatas = [{"source": fname, "chunk": i} for i in range(len(chunks))]

            self._collection.add(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )
            elapsed = time.time() - chunk_start
            print(f"     ✅ {elapsed:.0f} 秒")
            self._doc_count += len(chunks)
            total_chunks += len(chunks)

        total_elapsed = time.time() - total_start
        print(f"  📚 知识库加载完成：{total_chunks} 个片段（共 {total_elapsed:.0f} 秒）")

    # ================================================================
    # 分块
    # ================================================================

    def _chunk(self, text: str, chunk_size: int = 500, overlap: int = 50):
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
            # 已覆盖到末尾则退出
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    # ================================================================
    # 检索
    # ================================================================

    def query(self, question: str, top_k: int = 3) -> str:
        """语义检索，返回格式化结果"""
        if self._doc_count == 0:
            return ""

        results = self._collection.query(
            query_texts=[question],
            n_results=min(top_k, self._doc_count),
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        output = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            source = meta.get("source", "未知")
            score = round(1 - dist, 3)
            output.append(f"[{i+1}] 来源：{source}（相似度：{score}）")
            output.append(doc)

        return "\n\n".join(output)

    def count(self) -> int:
        return self._doc_count

    def rebuild(self):
        """强制重建索引：清空数据库，重新加载 repo 目录"""
        # 之前以空库初始化（未加载引擎），现在有文档了，重新完整初始化
        if self._client is None:
            self._initialized = False
            self.__init__(self._repo_dir)
            return
        self._client.delete_collection("research_knowledge")
        self._collection = self._client.get_or_create_collection(
            name="research_knowledge",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed_func,
        )
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
