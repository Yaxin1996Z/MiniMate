"""长期记忆 —— JSON 持久化 + 内容去重 + 关键词检索

跨会话保留用户偏好、项目信息等事实；Agent 启动时自动加载。
"""

import hashlib
import json
import os
import re

from .core import MemoryItem, score_relevance


class LongTermMemory:
    def __init__(self, path: str = "", max_items: int = 500):
        self.path = path or os.path.join(
            os.path.dirname(__file__), "..", "..", "..", ".cache", "memory.json"
        )
        self.max_items = max_items
        self._items: list[MemoryItem] = []
        self.load()

    # ----------------------------------------------------------
    # 写入（含去重）
    # ----------------------------------------------------------

    def add_fact(
        self,
        content: str,
        keywords: list[str] | None = None,
        scope: str = "project",
        project: str = "",
    ) -> bool:
        """添加事实；内容哈希去重，重复则跳过。

        scope：global（跨项目可见）或 project（仅当前项目可见，默认）
        """
        content = (content or "").strip()
        if not content:
            return False
        digest = hashlib.md5(content.encode("utf-8")).hexdigest()
        if any(getattr(i, "_digest", "") == digest for i in self._items):
            return False
        item = MemoryItem(
            type="fact",
            role="",
            content=content,
            keywords=keywords or self._extract_keywords(content),
            metadata={"scope": scope, "project": project} if scope == "project" else {"scope": "global"},
        )
        item._digest = digest
        self._items.append(item)
        if len(self._items) > self.max_items:
            self._items.pop(0)
        return True

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
        return [it for _, it in scored[:limit]]

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

    # ----------------------------------------------------------
    # 持久化
    # ----------------------------------------------------------

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = [
            {
                "type": i.type,
                "role": i.role,
                "content": i.content,
                "keywords": i.keywords,
                "metadata": i.metadata,
                "timestamp": i.timestamp,
            }
            for i in self._items
        ]
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            for d in data:
                item = MemoryItem(
                    type=d.get("type", "fact"),
                    role=d.get("role", ""),
                    content=d.get("content", ""),
                    keywords=d.get("keywords", []),
                    metadata=d.get("metadata", {}),
                    timestamp=d.get("timestamp", 0),
                )
                item._digest = hashlib.md5(item.content.encode("utf-8")).hexdigest()
                self._items.append(item)
        except (OSError, ValueError):
            pass

    @property
    def items(self) -> list[MemoryItem]:
        return list(self._items)

    def clear(self):
        self._items.clear()
