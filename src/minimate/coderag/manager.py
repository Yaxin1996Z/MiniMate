"""Code RAG 管理器 —— 仓库配置 / 索引构建 / 检索入口"""

import json
import os
import re
import subprocess

from .embeddings import embed
from .indexer import index_directory
from .storage import SQLiteCodeStorage
from .retriever import CodeRetriever


DEFAULT_RAG_DIR = os.path.join(os.path.expanduser("~"), ".minimate", "rag")
DEFAULT_REPOS_DIR = os.path.join(os.path.expanduser("~"), ".minimate", "repos")


def safe_repo_name(source: str) -> str:
    """从 URL/路径推导安全的仓库名"""
    s = source.strip().rstrip("/")
    if s.startswith(("http://", "https://", "git@")):
        name = s.split("/")[-1]
        name = re.sub(r"\.git$", "", name)
    else:
        name = os.path.basename(s) or "repo"
    return re.sub(r"[^A-Za-z0-9_\-]", "_", name) or "repo"


class CodeRAGManager:
    def __init__(self, rag_dir: str = DEFAULT_RAG_DIR, repos_dir: str = DEFAULT_REPOS_DIR):
        self.rag_dir = rag_dir
        self.repos_dir = repos_dir
        os.makedirs(rag_dir, exist_ok=True)
        os.makedirs(repos_dir, exist_ok=True)
        self._config_path = os.path.join(rag_dir, "repos.json")
        self._config: dict[str, str] = self._load_config()

    # ----------------------------------------------------------
    # 仓库配置
    # ----------------------------------------------------------

    def add_repo(self, name: str, source: str):
        self._config[name] = source
        self._save_config()

    def list_repos(self) -> dict[str, str]:
        return dict(self._config)

    def resolve_source(self, name: str) -> str | None:
        """返回可索引的本地目录：本地路径直接用，git URL clone 到 repos_dir"""
        source = self._config.get(name)
        if not source:
            return None
        if source.startswith(("http://", "https://", "git@")):
            local = os.path.join(self.repos_dir, name)
            if not os.path.isdir(local):
                result = subprocess.run(
                    ["git", "clone", source, local],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    raise ValueError(f"git clone 失败：{result.stderr[:200]}")
            return local
        return source

    def db_path(self, name: str) -> str:
        return os.path.join(self.rag_dir, f"{safe_repo_name(name)}.db")

    # ----------------------------------------------------------
    # 索引
    # ----------------------------------------------------------

    def index(self, name: str) -> dict:
        """AST 分块 + 关系提取 + 向量化 + SQLite 持久化"""
        root = self.resolve_source(name)
        if not root or not os.path.isdir(root):
            raise ValueError(f"仓库不可用：{name}")
        chunks, relations = index_directory(root)
        for c in chunks:
            c.vector = embed(c.search_text)
        storage = SQLiteCodeStorage(self.db_path(name))
        storage.save_repo(name, chunks, relations, {"source": self._config[name]})
        return {
            "chunks": len(chunks),
            "relations": len(relations),
            "db": self.db_path(name),
        }

    # ----------------------------------------------------------
    # 检索
    # ----------------------------------------------------------

    def _retriever(self, name: str) -> CodeRetriever:
        return CodeRetriever(SQLiteCodeStorage(self.db_path(name)))

    def search(self, name: str, query: str, top_k: int = 5) -> list[dict]:
        return self._retriever(name).search(name, query, top_k)

    def call_chain(self, name: str, target: str, depth: int = 5):
        return self._retriever(name).call_chain(name, target, depth)

    def relations_of(self, name: str, symbol: str) -> list[dict]:
        return self._retriever(name).relations_of(name, symbol)

    # ----------------------------------------------------------
    # 配置持久化
    # ----------------------------------------------------------

    def _load_config(self) -> dict:
        if not os.path.exists(self._config_path):
            return {}
        try:
            with open(self._config_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_config(self):
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, ensure_ascii=False, indent=2)
