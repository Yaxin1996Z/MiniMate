"""MCP 适配器 —— 把 MCP Server 暴露的工具包装成本地 Tool

Agent 的 Function Calling 循环对工具来源无感知：
  - 本地工具：ToolExecutor 直接执行
  - MCP 工具：通过本适配器连接 MCP Server，schema 来自 tools/list，
              执行走 tools/call（JSON-RPC）
"""

import asyncio
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .tools import Tool


class McpToolAdapter:
    """连接单个 MCP Server（stdio），把远程工具包装为统一 Tool 接口"""

    def __init__(
        self,
        server_name: str,
        command: str,
        args: list[str] | None = None,
        env: dict | None = None,
    ):
        self.server_name = server_name
        self.params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env or None,
        )
        self._tools: list[Tool] = []
        self._session = None
        self._session_ctx = None
        self._client_ctx = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    def connect(self, timeout: float = 30.0) -> list[Tool]:
        """启动后台事件循环并连接 MCP Server，返回包装后的工具列表"""
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"mcp-{self.server_name}",
            daemon=True,
        )
        self._thread.start()

        future = asyncio.run_coroutine_threadsafe(self._setup(), self._loop)
        future.result(timeout=timeout)
        return list(self._tools)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _setup(self):
        """建立 stdio 连接 + 会话，拉取工具列表并包装"""
        self._client_ctx = stdio_client(self.params)
        self._read, self._write = await self._client_ctx.__aenter__()
        self._session_ctx = ClientSession(self._read, self._write)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()

        result = await self._session.list_tools()
        for t in result.tools:
            self._tools.append(Tool(
                name=t.name,
                description=f"[MCP:{self.server_name}] {t.description or t.name}",
                parameters=t.inputSchema or {"type": "object", "properties": {}},
                func=self._make_caller(t.name),
            ))

    def _make_caller(self, tool_name: str):
        """构造同步执行函数：跨线程把调用投递到后台事件循环"""
        def caller(**kwargs):
            future = asyncio.run_coroutine_threadsafe(
                self._call(tool_name, kwargs), self._loop
            )
            return future.result(timeout=60)
        return caller

    async def _call(self, tool_name: str, arguments: dict) -> str:
        result = await self._session.call_tool(tool_name, arguments=arguments)
        parts = []
        for c in result.content or []:
            if hasattr(c, "text"):
                parts.append(c.text)
            else:
                parts.append(str(c))
        text = "\n".join(parts)
        if getattr(result, "isError", False):
            return f"[MCP 工具错误] {text}"
        return text

    def close(self, timeout: float = 10.0):
        """关闭会话与事件循环"""
        if self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._cleanup(), self._loop).result(timeout=timeout)
            except Exception:
                pass
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    async def _cleanup(self):
        for ctx in (self._session_ctx, self._client_ctx):
            if ctx is not None:
                try:
                    await ctx.__aexit__(None, None, None)
                except Exception:
                    pass
