"""示例远程 MCP Server（FastMCP, Streamable HTTP 传输）

模拟"远程" MCP Server：通过 HTTP 暴露工具。
运行：uv run python examples/mcp_http_server.py
然后在 config.json 配 transport=http + url=http://127.0.0.1:9100/mcp
"""

from fastmcp import FastMCP


mcp = FastMCP("mini-http-demo")


@mcp.tool()
def greeting(name: str) -> str:
    """给指定名字发送问候语"""
    return f"你好，{name}！这是来自远程 MCP Server 的问候。"


@mcp.tool()
def reverse(text: str) -> str:
    """反转字符串"""
    return text[::-1]


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="127.0.0.1", port=9100)
