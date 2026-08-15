"""Code RAG 管理器 —— 仓库配置 / 索引构建 / 检索入口"""

import json
import os
import re
import subprocess

from .embeddings import HashEmbeddingProvider, get_provider
from .indexer import index_directory
from .storage import SQLiteCodeStorage
from .retriever import CodeRetriever


DEFAULT_RAG_DIR = os.path.join(
    os.path.expanduser("~"), ".minimate", "coderepos", "rag_db"
)
DEFAULT_REPOS_DIR = os.path.join(
    os.path.expanduser("~"), ".minimate", "coderepos", "repos"
)


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
    def __init__(
        self,
        rag_dir: str = DEFAULT_RAG_DIR,
        repos_dir: str = DEFAULT_REPOS_DIR,
        provider=None,
    ):
        self.rag_dir = rag_dir
        self.repos_dir = repos_dir
        self._provider = provider or get_provider()
        os.makedirs(rag_dir, exist_ok=True)
        os.makedirs(repos_dir, exist_ok=True)
        # 仓库配置放 coderepos 根目录（与 rag_db 数据目录分离）
        self._config_path = os.path.join(os.path.dirname(rag_dir), "repos.json")
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
        vectors = self._provider.encode([c.search_text for c in chunks])
        for c, v in zip(chunks, vectors):
            c.vector = v
        storage = SQLiteCodeStorage(self.db_path(name))
        storage.save_repo(name, chunks, relations, {"source": self._config[name]})
        return {
            "chunks": len(chunks),
            "relations": len(relations),
            "db": self.db_path(name),
        }

    def update(self, name: str) -> dict:
        """更新仓库：git 仓库先 pull 最新代码，再重建索引（本地目录直接重扫）

        git pull 失败（如网络不可达）不中断：降级为使用现有代码重建索引，并给出警告。
        """
        source = self._config.get(name)
        if not source:
            raise ValueError(f"仓库未配置：{name}")
        root = self.resolve_source(name)
        if not root or not os.path.isdir(root):
            raise ValueError(f"仓库不可用：{name}")
        if source.startswith(("http://", "https://", "git@")):
            try:
                result = subprocess.run(
                    ["git", "pull"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if result.returncode != 0:
                    print(
                        f"  [警告] git pull 失败（{result.stderr.strip()[:150]}），"
                        "继续使用现有代码重建索引"
                    )
            except (OSError, subprocess.TimeoutExpired) as e:
                print(f"  [警告] git pull 不可用（{e}），继续使用现有代码重建索引")
        return self.index(name)

    # ----------------------------------------------------------
    # 检索
    # ----------------------------------------------------------

    def _retriever(self, name: str) -> CodeRetriever:
        return CodeRetriever(
            SQLiteCodeStorage(self.db_path(name)), provider=self._provider
        )

    _IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]{1,}$")

    def search(self, name: str, query: str, top_k: int = 5) -> list[dict]:
        """两级检索：标识符查询先 rg 精确关键词定位，再混合语义召回"""
        semantic = self._retriever(name).search(name, query, top_k)
        q = (query or "").strip()
        keyword = (
            self.search_keyword(name, q, limit=top_k)
            if self._IDENT_RE.match(q)
            else []
        )
        if not keyword:
            return semantic
        merged: list[dict] = []
        seen: set[tuple] = set()
        for c in keyword + semantic:
            key = (c.get("file_path"), c.get("name"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)
            if len(merged) >= top_k:
                break
        return merged

    def search_keyword(self, name: str, keyword: str, limit: int = 10) -> list[dict]:
        """rg 精确关键词定位（rg 不可用时回退逐行扫描）"""
        root = self.resolve_source(name)
        if not root or not os.path.isdir(root):
            return []
        keyword = (keyword or "").strip()
        if not keyword:
            return []
        hits = self._rg_search(root, keyword, limit)
        if hits is None:
            hits = self._scan_search(root, keyword, limit)
        return hits

    def _rg_search(self, root: str, keyword: str, limit: int) -> list[dict] | None:
        """rg -n 定位；rg 不可用（未安装/超时）时返回 None 交给回退"""
        try:
            result = subprocess.run(
                ["rg", "-n", "-i", "--no-heading", "-m", str(limit), "--", keyword, root],
                capture_output=True, text=True, timeout=15,
            )
        except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode not in (0, 1):
            return None
        hits: list[dict] = []
        for line in result.stdout.splitlines():
            parts = line.split(":", 2)
            if len(parts) < 2 or not parts[1].isdigit():
                continue
            path, lineno, content = parts[0], parts[1], (parts[2] if len(parts) > 2 else "")
            hits.append({
                "granularity": "keyword",
                "name": keyword,
                "file_path": path,
                "start_line": int(lineno),
                "code": content.strip()[:200],
                "score": 1.0,
            })
            if len(hits) >= limit:
                break
        return hits

    def _scan_search(self, root: str, keyword: str, limit: int) -> list[dict]:
        """无 rg 时的回退：递归扫描文本文件逐行匹配（大小写不敏感）"""
        hits: list[dict] = []
        needle = keyword.lower()
        skip = {".git", ".venv", "node_modules", "__pycache__", "rag_db", "output", "dist", "build"}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in skip]
            for fn in filenames:
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for ln, text in enumerate(f, 1):
                            if needle in text.lower():
                                hits.append({
                                    "granularity": "keyword",
                                    "name": keyword,
                                    "file_path": path,
                                    "start_line": ln,
                                    "code": text.strip()[:200],
                                    "score": 1.0,
                                })
                                if len(hits) >= limit:
                                    return hits
                except OSError:
                    continue
        return hits

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
