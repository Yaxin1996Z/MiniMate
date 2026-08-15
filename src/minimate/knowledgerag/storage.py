"""Knowledge RAG 存储 —— SQLite 持久化文档块与向量

与 Code RAG 统一存储方案：向量以 JSON 数组持久化，检索时内存计算余弦，
零外部向量库依赖。表结构：
  documents(id, source, content, vector_json, indexed_at)
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager


class KnowledgeStorage:
    """文档块 SQLite 存储"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
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
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    content TEXT NOT NULL,
                    vector_json TEXT,
                    indexed_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source)"
            )

    def save_documents(self, documents: list[dict], embed_model: str = "") -> None:
        """重建全部文档索引（清空旧数据后写入）"""
        with self._connect() as conn:
            conn.execute("DELETE FROM documents")
            conn.execute(
                "INSERT OR REPLACE INTO meta VALUES ('embed_model', ?)",
                (embed_model,),
            )
            now = time.time()
            for doc in documents:
                conn.execute(
                    "INSERT INTO documents VALUES (?,?,?,?,?)",
                    (
                        doc["id"],
                        doc["source"],
                        doc["content"],
                        json.dumps(doc.get("vector") or [], ensure_ascii=False),
                        now,
                    ),
                )

    def get_embed_model(self) -> str:
        """返回索引时使用的 embedding 模型标识（空表示未记录）"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM meta WHERE key = 'embed_model'"
            ).fetchone()
        return row["value"] if row else ""

    def load_documents(self) -> list[dict]:
        """加载全部文档块（vector 解析为 list）"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, source, content, vector_json FROM documents"
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["vector"] = json.loads(d.pop("vector_json") or "[]")
            result.append(d)
        return result

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()
        return row["n"] if row else 0

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM meta")
