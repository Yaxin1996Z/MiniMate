"""stdio 传输 —— 与本地子进程（MCP Server）的进程间连接

本质：MiniMate 启动一个子进程，通过 stdin/stdout 传输 JSON-RPC 消息。
"""

from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client


def create_client_ctx(config: dict):
    """创建 stdio 客户端上下文（async context manager）

    进入时启动子进程并返回 (read, write) 传输流。
    config 需含 command，可选 args / env。
    """
    command = config.get("command")
    if not command:
        raise ValueError("stdio 传输需要 command 配置")
    params = StdioServerParameters(
        command=command,
        args=config.get("args") or [],
        env=config.get("env") or None,
    )
    return stdio_client(params), None
