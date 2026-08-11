"""MCP 适配器 —— 把 MCP Server 暴露的工具包装成本地 Tool

支持两种传输（transport）：
  stdio - 本地子进程连接（通过 stdin/stdout 通信）
  http  - 远程 URL 连接（Streamable HTTP，支持认证 headers）

Agent 的 Function Calling 循环对工具来源无感知：
  - 本地工具：ToolExecutor 直接执行
  - MCP 工具：通过本适配器连接 MCP Server，schema 来自 tools/list，
              执行走 tools/call（JSON-RPC）
"""

import asyncio
import threading
from datetime import datetime

from mcp import ClientSession

from ..tools import Tool
from ..logging import logger
from .stdio import create_client_ctx as _stdio_ctx
from .http import create_client_ctx as _http_ctx


class McpStatus:
    """MCP 服务器连接状态"""

    PENDING = "pending"        # 未连接
    CONNECTING = "connecting"  # 连接中
    CONNECTED = "connected"    # 已连接
    FAILED = "failed"          # 连接失败
    CLOSED = "closed"          # 已关闭


class McpToolAdapter:
    """连接单个 MCP Server（stdio 或 http），把远程工具包装为统一 Tool 接口"""

    def __init__(
        self,
        server_name: str,
        transport: str = "stdio",
        command: str | None = None,
        args: list[str] | None = None,
        env: dict | None = None,
        url: str | None = None,
        headers: dict | None = None,
        oauth: dict | None = None,
    ):
        self.server_name = server_name
        self.transport = transport
        self.config = {
            "command": command,
            "args": args or [],
            "env": env,
            "url": url,
            "headers": headers,
            "oauth": oauth,
        }
        # 连接状态
        self.status = McpStatus.PENDING
        self.error = ""
        self.connected_at = ""
        self.tool_count = 0
        self._tools: list[Tool] = []
        self._session = None
        self._session_ctx = None
        self._client_ctx = None
        self._http_client = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    @property
    def tools(self) -> list[Tool]:
        return list(self._tools)

    def connect(self, timeout: float = 30.0) -> list[Tool]:
        """启动后台事件循环并连接 MCP Server，返回包装后的工具列表"""
        self.status = McpStatus.CONNECTING
        self.error = ""
        try:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._run_loop,
                name=f"mcp-{self.server_name}",
                daemon=True,
            )
            self._thread.start()

            future = asyncio.run_coroutine_threadsafe(self._setup(), self._loop)
            future.result(timeout=timeout)
            self.status = McpStatus.CONNECTED
            self.tool_count = len(self._tools)
            self.connected_at = datetime.now().strftime("%H:%M:%S")
            logger.info(
                "MCP 连接成功 server=%s transport=%s tools=%d",
                self.server_name, self.transport, self.tool_count,
            )
        except Exception as e:
            self.status = McpStatus.FAILED
            self.error = str(e)
            logger.error(
                "MCP 连接失败 server=%s transport=%s error=%s",
                self.server_name, self.transport, e,
            )
            # 清理失败时启动的事件循环线程
            if self._loop is not None:
                try:
                    self._loop.call_soon_threadsafe(self._loop.stop)
                except Exception:
                    pass
            if self._thread is not None:
                self._thread.join(timeout=5)
            raise
        return list(self._tools)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _setup(self):
        """建立连接（按 transport 选择）+ 会话，拉取工具列表并包装"""
        if self.transport == "http":
            self._client_ctx, self._http_client = _http_ctx(self.config)
        elif self.transport == "stdio":
            self._client_ctx, self._http_client = _stdio_ctx(self.config)
        else:
            raise ValueError(f"未知 MCP transport：{self.transport}（支持 stdio / http）")
        entered = await self._client_ctx.__aenter__()
        if self.transport == "http":
            # streamable_http_client 返回 3 元组：(read, write, get_session_id)
            self._read, self._write, _ = entered
        else:
            self._read, self._write = entered
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
        self.status = McpStatus.CLOSED

    async def _cleanup(self):
        for ctx in (self._session_ctx, self._client_ctx):
            if ctx is not None:
                try:
                    await ctx.__aexit__(None, None, None)
                except Exception:
                    pass
        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception:
                pass
