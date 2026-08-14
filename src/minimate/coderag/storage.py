"""SQLite 轻量级向量存储 —— 向量以 JSON 数组持久化

表结构：
  chunks(id, repo, file_path, granularity, name, code, start_line, end_line, doc, vector_json)
  relations(repo, from_id, from_name, relation, to_id, to_name)
  repo_meta(repo, config_json, indexed_at)
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager


class SQLiteCodeStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        """带自动提交与关闭的连接上下文（sqlite3 的 with 只提交不关闭）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    file_path TEXT,
                    granularity TEXT,
                    name TEXT,
                    code TEXT,
                    start_line INTEGER,
                    end_line INTEGER,
                    doc TEXT,
                    vector_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relations (
                    repo TEXT,
                    from_id TEXT,
                    from_name TEXT,
                    relation TEXT,
                    to_id TEXT,
                    to_name TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS repo_meta (
                    repo TEXT PRIMARY KEY,
                    config_json TEXT,
                    indexed_at REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_repo ON relations(repo)")

    # ----------------------------------------------------------
    # 写入
    # ----------------------------------------------------------

    def save_repo(self, repo: str, chunks: list, relations: list, config: dict):
        """重建指定仓库索引（清空旧数据后写入）"""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE repo = ?", (repo,))
            conn.execute("DELETE FROM relations WHERE repo = ?", (repo,))
            for c in chunks:
                conn.execute(
                    "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        c.chunk_id, repo, c.file_path, c.granularity,
                        c.name, c.code, c.start_line, c.end_line, c.doc,
                        json.dumps(getattr(c, "vector", []), ensure_ascii=False),
                    ),
                )
            for r in relations:
                conn.execute(
                    "INSERT INTO relations VALUES (?,?,?,?,?,?)",
                    (repo,) + r.to_tuple(),
                )
            conn.execute(
                "INSERT OR REPLACE INTO repo_meta VALUES (?,?,?)",
                (repo, json.dumps(config, ensure_ascii=False), time.time()),
            )

    # ----------------------------------------------------------
    # 读取
    # ----------------------------------------------------------

    def load_chunks(self, repo: str) -> list[dict]:
        """加载仓库所有分块（vector 解析为 list）"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE repo = ?", (repo,)
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["vector"] = json.loads(d.pop("vector_json") or "[]")
            result.append(d)
        return result

    def load_relations(self, repo: str) -> list[tuple]:
        with self._connect() as conn:
            return [
                tuple(r) for r in conn.execute(
                    "SELECT from_id, from_name, relation, to_id, to_name "
                    "FROM relations WHERE repo = ?", (repo,)
                ).fetchall()
            ]

    def list_repos(self) -> list[str]:
        with self._connect() as conn:
            return [r[0] for r in conn.execute(
                "SELECT repo FROM repo_meta ORDER BY indexed_at DESC"
            ).fetchall()]

    def get_meta(self, repo: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT config_json FROM repo_meta WHERE repo = ?", (repo,)
            ).fetchone()
        return json.loads(row["config_json"]) if row else None

    def clear_repo(self, repo: str):
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE repo = ?", (repo,))
            conn.execute("DELETE FROM relations WHERE repo = ?", (repo,))
            conn.execute("DELETE FROM repo_meta WHERE repo = ?", (repo,))
