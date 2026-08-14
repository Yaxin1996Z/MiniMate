"""Code RAG 工具 —— search_code 注册给 Agent"""

from ..tools import tool


@tool(
    name="search_code",
    description=(
        "从已索引的代码仓库中检索相关代码，参数 query 为自然语言问题，"
        "返回相关代码块（文件/类/方法）与相似度。"
        "当任务需要理解、查找、修改项目代码时，请优先调用本工具。"
    ),
)
def search_code(query: str, repo: str = "", top_k: int = 3) -> str:
    """检索代码仓库（自然语言 → 相关代码块）"""
    from .manager import CodeRAGManager

    mgr = CodeRAGManager()
    repos = mgr.list_repos()
    if not repos:
        return "未配置代码仓库。请先使用 /repos add <名称> <路径或URL> 配置并 index。"
    name = repo if repo in repos else next(iter(repos))
    try:
        results = mgr.search(name, query, max(1, min(top_k, 10)))
    except Exception as e:
        return f"[检索错误] {e}"
    if not results:
        return f"未找到相关代码（仓库：{name}）"

    lines = [f"相关代码（仓库 {name}）："]
    for c in results:
        lines.append(
            f"- [{c['granularity']}] {c['name']}  "
            f"{c['file_path']}:{c['start_line']}（相似度 {c.get('score', 0)}）"
        )
        code_preview = (c.get("code") or "").replace("\n", " ")[:180]
        if code_preview:
            lines.append(f"  {code_preview}")
    return "\n".join(lines)
