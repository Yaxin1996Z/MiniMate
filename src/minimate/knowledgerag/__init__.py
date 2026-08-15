"""Knowledge RAG —— 文档知识库（与 Code RAG 区分）

文档源：~/.minimate/knowledge/docs/（.md / .txt）
向量库：~/.minimate/knowledge/rag_db/（Chroma + bge）
"""
from .knowledge import KnowledgeBase, get_knowledge_base
