"""RAG 知识库工具 —— 对应 ToolRegistry.registerRagTools"""

from .core import ToolExecutor, _first_line, tool


@tool(name="query_knowledge", description="从本地知识库中搜索相关内容。知识库已预加载了本地文档，输入问题返回最相关的文档片段，可作为调研参考")
def query_knowledge(question: str) -> str:
    """从已加载的本地知识库中检索相关内容"""
    question = _first_line(question)
    from ..knowledgerag import get_knowledge_base
    kb = get_knowledge_base()
    return kb.query(question) or "知识库中没有相关内容。"


def register_rag_tools(executor: ToolExecutor) -> None:
    """注册 RAG 知识库工具（对应 ToolRegistry.registerRagTools）"""
    executor.register(query_knowledge)
