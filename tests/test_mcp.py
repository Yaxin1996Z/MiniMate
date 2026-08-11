"""MCP 适配器单元测试 —— mock MCP 客户端，不启动真实 Server"""

import unittest
from unittest.mock import patch

from minimate.mcp import McpToolAdapter


class FakeTool:
    name = "add"
    description = "计算两个整数之和"
    inputSchema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }


class FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def initialize(self):
        pass

    async def list_tools(self):
        return type("R", (), {"tools": [FakeTool()]})()

    async def call_tool(self, name, arguments):
        content = type("C", (), {"text": str(arguments["a"] + arguments["b"])})()
        result = type("R", (), {"content": [content], "isError": False})()
        return result


class FakeStdioClient:
    async def __aenter__(self):
        return ("read", "write")

    async def __aexit__(self, *args):
        return False


class FakeHttpClient:
    """streamable_http_client 的 __aenter__ 返回 3 元组（含 get_session_id）"""

    async def __aenter__(self):
        return ("read", "write", lambda: None)

    async def __aexit__(self, *args):
        return False


class McpAdapterTest(unittest.TestCase):
    @patch("minimate.mcp.adapter.ClientSession", return_value=FakeSession())
    @patch("minimate.mcp.stdio.stdio_client", return_value=FakeStdioClient())
    def test_connect_wraps_tools(self, mock_stdio, mock_session):
        adapter = McpToolAdapter(server_name="demo", transport="stdio", command="python", args=["x.py"])
        tools = adapter.connect()

        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertEqual(tool.name, "add")
        self.assertIn("[MCP:demo]", tool.description)
        self.assertEqual(tool.parameters["required"], ["a", "b"])

        # 执行包装后的工具 → 走 MCP call_tool
        result = tool.run_kwargs({"a": 3, "b": 4})
        self.assertEqual(result, "7")

    @patch("minimate.mcp.adapter.ClientSession", return_value=FakeSession())
    @patch("minimate.mcp.stdio.stdio_client", return_value=FakeStdioClient())
    def test_close_is_safe(self, mock_stdio, mock_session):
        adapter = McpToolAdapter(server_name="demo", transport="stdio", command="python")
        adapter.connect()
        adapter.close()  # 不应抛异常

    @patch("minimate.mcp.adapter.ClientSession", return_value=FakeSession())
    @patch("minimate.mcp.http.streamable_http_client", return_value=FakeHttpClient())
    def test_http_transport_connect(self, mock_http, mock_session):
        """http 传输：连接远程 URL 并包装工具"""
        adapter = McpToolAdapter(
            server_name="remote",
            transport="http",
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer x"},
        )
        tools = adapter.connect()
        self.assertEqual(len(tools), 1)
        mock_http.assert_called_once()
        args, kwargs = mock_http.call_args
        self.assertIn("https://example.com/mcp", args)

    @patch("minimate.mcp.adapter.ClientSession", return_value=FakeSession())
    @patch("minimate.mcp.stdio.stdio_client", return_value=FakeStdioClient())
    def test_status_connected(self, mock_stdio, mock_session):
        adapter = McpToolAdapter(server_name="demo", transport="stdio", command="python")
        adapter.connect()
        self.assertEqual(adapter.status, "connected")
        self.assertEqual(adapter.tool_count, 1)
        self.assertEqual(adapter.error, "")

    @patch("minimate.mcp.stdio.stdio_client", side_effect=RuntimeError("连接失败"))
    def test_status_failed(self, mock_stdio):
        adapter = McpToolAdapter(server_name="demo", transport="stdio", command="python")
        with self.assertRaises(RuntimeError):
            adapter.connect()
        self.assertEqual(adapter.status, "failed")
        self.assertIn("连接失败", adapter.error)

    def test_unknown_transport_raises(self):
        adapter = McpToolAdapter(server_name="x", transport="websocket", command="python")
        with self.assertRaises(ValueError):
            adapter.connect()


if __name__ == "__main__":
    unittest.main()
