"""长期记忆 —— SQLite 持久化（memories 表）+ 归一化去重 + 关键词检索

表结构对齐 PaiCLI memory.db 设计：scope / content / created_at / kind /
source / importance / confidence / updated_at / expires_at / access_count /
content_hash。跨会话保留用户偏好、项目信息等事实，Agent 启动时自动加载。
"""

import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone

from .core import MemoryItem, score_relevance


def _default_db_path() -> str:
    """默认数据库：~/.minimate/memory.db"""
    return os.path.join(os.path.expanduser("~"), ".minimate", "memory.db")


def _iso_to_epoch(iso: str) -> float:
    """ISO 时间字符串 → epoch 秒（解析失败返回 0）"""
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


class LongTermMemory:
    def __init__(self, path: str = "", max_items: int = 500):
        self.path = path or _default_db_path()
        self.max_items = max_items
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._init_schema()
        self._items: list[MemoryItem] = []
        self._normalize_existing()
        self._migrate_legacy_json()
        self.load()

    # ----------------------------------------------------------
    # 存储层
    # ----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    kind TEXT NOT NULL DEFAULT 'fact',
                    source TEXT NOT NULL DEFAULT 'manual',
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    updated_at TEXT NOT NULL DEFAULT '',
                    expires_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    content_hash TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
                CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope);
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _legacy_json_path(self) -> str:
        """旧版 JSON 持久化路径（项目 .cache/memory.json）"""
        return os.path.join(
            os.path.dirname(__file__), "..", "..", "..", ".cache", "memory.json"
        )

    def _migrate_legacy_json(self):
        """首次使用默认库时，把旧 JSON 长期记忆导入 SQLite（旧文件保留作备份）"""
        if os.path.normpath(self.path) != os.path.normpath(_default_db_path()):
            return
        legacy = self._legacy_json_path()
        if not os.path.exists(legacy):
            return
        conn = self._connect()
        try:
            count = conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()["c"]
        finally:
            conn.close()
        if count:
            return
        try:
            with open(legacy, encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                content = (d.get("content") or "").strip()
                if not content:
                    continue
                meta = d.get("metadata") or {}
                self.add_fact(
                    content,
                    keywords=d.get("keywords") or None,
                    scope="global" if meta.get("scope") == "global" else "project",
                    project=meta.get("project", ""),
                    source="migrated",
                )
        except (OSError, ValueError):
            pass

    # ----------------------------------------------------------
    # 写入（含去重）
    # ----------------------------------------------------------

    def add_fact(
        self,
        content: str,
        keywords: list[str] | None = None,
        scope: str = "project",
        project: str = "",
        source: str = "manual",
    ) -> bool:
        """添加事实；按归一化内容哈希去重，重复则跳过。

        scope：global（跨项目可见）或 project（仅当前项目可见，默认）
        source：manual / agent / migrated 等，记录事实来源
        """
        content = (content or "").strip()
        if not content:
            return False
        digest = hashlib.sha256(self._normalize(content).encode("utf-8")).hexdigest()
        if any(getattr(i, "_digest", "") == digest for i in self._items):
            return False

        now = datetime.now(timezone.utc)
        created_iso = now.isoformat()
        db_scope = "global" if scope == "global" else (project or "")
        conn = self._connect()
        try:
            if self.max_items:
                row = conn.execute(
                    "SELECT COUNT(*) AS c, MIN(id) AS mid FROM memories"
                ).fetchone()
                if row["c"] >= self.max_items and row["mid"] is not None:
                    # 超容量时淘汰最旧记录（最小 id）
                    conn.execute("DELETE FROM memories WHERE id = ?", (row["mid"],))
            try:
                conn.execute(
                    """
                    INSERT INTO memories
                        (scope, content, created_at, kind, source, importance,
                         confidence, updated_at, expires_at, access_count, content_hash)
                    VALUES (?, ?, ?, 'fact', ?, 0.5, 1.0, ?, NULL, 0, ?)
                    """,
                    (db_scope, content, created_iso, source, created_iso, digest),
                )
            except sqlite3.IntegrityError:
                return False  # 并发插入重复时的兜底
            conn.commit()
        finally:
            conn.close()

        metadata = (
            {"scope": "global"}
            if scope == "global"
            else {"scope": "project", "project": project}
        )
        item = MemoryItem(
            type="fact",
            role="",
            content=content,
            keywords=keywords or self._extract_keywords(content),
            metadata=metadata,
            timestamp=now.timestamp(),
        )
        item._digest = digest
        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items.pop(0)
        return True

    @staticmethod
    def _normalize(content: str) -> str:
        """归一化内容用于去重哈希：小写 + 去常见前缀 + 去空白/标点。

        例如"用户偏好：喜欢 Python。"与"用户偏好（显式）：喜欢Python"归一化后
        都得到"喜欢python"，视为同一条事实。
        """
        text = (content or "").strip().lower()
        for prefix in ("用户偏好（显式）：", "用户偏好：", "项目信息："):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)

    def _normalize_existing(self):
        """存量数据归一化：重算 content_hash 并按归一化内容合并重复（保留最早一条）"""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, content, content_hash FROM memories ORDER BY id"
            ).fetchall()
            seen: set[str] = set()
            for row in rows:
                digest = hashlib.sha256(
                    self._normalize(row["content"]).encode("utf-8")
                ).hexdigest()
                if digest in seen:
                    conn.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
                else:
                    seen.add(digest)
                    if digest != row["content_hash"]:
                        conn.execute(
                            "UPDATE memories SET content_hash = ? WHERE id = ?",
                            (digest, row["id"]),
                        )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """简单关键词提取：英文词（>=3 字符）与中文连续词（2-6 字）"""
        words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,6}", text)
        return list(dict.fromkeys(words))[:10]

    # ----------------------------------------------------------
    # 检索
    # ----------------------------------------------------------

    def search(self, query: str, limit: int = 5, project: str = "") -> list[MemoryItem]:
        """相关度检索：精确匹配 + 关键词比例 + 时间衰减；按项目可见性过滤"""
        scored: list[tuple[float, MemoryItem]] = []
        for it in self._items:
            if not self._is_visible(it, project):
                continue
            score = score_relevance(it, query)
            if score > 0:
                # 长期记忆更精炼，加权
                scored.append((score * 1.2, it))
        scored.sort(key=lambda x: -x[0])
        result = [it for _, it in scored[:limit]]
        if result:
            self._bump_access(result)
        return result

    def get_visible(self, project: str = "") -> list[MemoryItem]:
        """返回当前项目可见的全部记忆"""
        return [it for it in self._items if self._is_visible(it, project)]

    @staticmethod
    def _is_visible(item: MemoryItem, project: str) -> bool:
        scope = item.metadata.get("scope", "project")
        if scope == "global":
            return True
        if not project:
            return True
        return item.metadata.get("project", "") == project

    def _bump_access(self, items: list[MemoryItem]):
        """检索命中的事实 access_count +1"""
        digests = [getattr(i, "_digest", "") for i in items if getattr(i, "_digest", "")]
        if not digests:
            return
        conn = self._connect()
        try:
            for d in digests:
                conn.execute(
                    "UPDATE memories SET access_count = access_count + 1 WHERE content_hash = ?",
                    (d,),
                )
            conn.commit()
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 持久化（SQLite 实时落盘，save/load 保持兼容）
    # ----------------------------------------------------------

    def save(self):
        """SQLite 每次写入即提交，此方法仅为兼容旧接口保留"""
        return None

    def load(self):
        conn = self._connect()
        try:
            rows = conn.execute("SELECT * FROM memories ORDER BY id").fetchall()
        finally:
            conn.close()
        self._items = [self._row_to_item(r) for r in rows]

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        scope = row["scope"]
        if scope == "global":
            metadata = {"scope": "global"}
        else:
            metadata = {"scope": "project", "project": scope}
        item = MemoryItem(
            type=row["kind"] or "fact",
            role="",
            content=row["content"],
            timestamp=_iso_to_epoch(row["created_at"]),
            keywords=self._extract_keywords(row["content"]),
            metadata=metadata,
        )
        item._digest = row["content_hash"]
        return item

    @property
    def items(self) -> list[MemoryItem]:
        return list(self._items)

    def clear(self):
        conn = self._connect()
        try:
            conn.execute("DELETE FROM memories")
            conn.commit()
        finally:
            conn.close()
        self._items.clear()
