"""互联网工具 —— 对应 ToolRegistry.registerWebTools"""

from .core import ToolExecutor, _first_line, tool


@tool(name="web_search", description="搜索互联网，输入搜索关键词，返回搜索结果标题和摘要")
def web_search(query: str) -> str:
    """搜索互联网，返回最新信息"""
    query = _first_line(query)
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                results.append(f"- {title}\n  {body}\n  {href}")
        if not results:
            return "没有找到相关结果。"
        return "搜索结果：\n" + "\n\n".join(results)
    except Exception as e:
        return f"[搜索出错] {e}"


def register_web_tools(executor: ToolExecutor) -> None:
    """注册互联网工具（对应 ToolRegistry.registerWebTools）"""
    executor.register(web_search)
