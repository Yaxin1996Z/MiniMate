"""示例 MCP Server（FastMCP）—— 提供计算与时间工具

运行：uv run python examples/mcp_demo_server.py
通过 config.json 的 mcp.servers 配置后，MiniMate 启动时自动加载其工具。
"""

from datetime import datetime

from fastmcp import FastMCP


mcp = FastMCP("mini-demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """计算两个整数之和"""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """计算两个整数之积"""
    return a * b


@mcp.tool()
def current_time() -> str:
    """返回当前日期与时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    mcp.run()
