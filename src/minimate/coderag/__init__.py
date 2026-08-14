"""Code RAG —— 代码仓库索引与语义检索

结构：
  embeddings.py - hashing TF 向量 + 余弦相似度（零模型依赖）
  indexer.py    - AST 多粒度分块（文件/类/方法）+ 关系图谱
  storage.py    - SQLite 存储（向量 JSON 数组持久化）
  retriever.py  - 余弦检索 + 调用链查询
  manager.py    - CodeRAGManager（配置/索引/检索）
  tools.py      - search_code 工具（注册给 Agent）
"""

from .embeddings import cosine, embed
from .indexer import GRANULARITY_FILE, GRANULARITY_CLASS, GRANULARITY_METHOD, index_directory, index_file
from .manager import CodeRAGManager, safe_repo_name
from .retriever import CodeRetriever
from .storage import SQLiteCodeStorage
from .tools import search_code

__all__ = [
    "cosine",
    "embed",
    "GRANULARITY_FILE",
    "GRANULARITY_CLASS",
    "GRANULARITY_METHOD",
    "index_directory",
    "index_file",
    "CodeRAGManager",
    "safe_repo_name",
    "CodeRetriever",
    "SQLiteCodeStorage",
    "search_code",
]
