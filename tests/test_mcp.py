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


class McpAdapterTest(unittest.TestCase):
    @patch("minimate.mcp.ClientSession", return_value=FakeSession())
    @patch("minimate.mcp.stdio_client", return_value=FakeStdioClient())
    def test_connect_wraps_tools(self, mock_stdio, mock_session):
        adapter = McpToolAdapter(server_name="demo", command="python", args=["x.py"])
        tools = adapter.connect()

        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertEqual(tool.name, "add")
        self.assertIn("[MCP:demo]", tool.description)
        self.assertEqual(tool.parameters["required"], ["a", "b"])

        # 执行包装后的工具 → 走 MCP call_tool
        result = tool.run_kwargs({"a": 3, "b": 4})
        self.assertEqual(result, "7")

    @patch("minimate.mcp.ClientSession", return_value=FakeSession())
    @patch("minimate.mcp.stdio_client", return_value=FakeStdioClient())
    def test_close_is_safe(self, mock_stdio, mock_session):
        adapter = McpToolAdapter(server_name="demo", command="python")
        adapter.connect()
        adapter.close()  # 不应抛异常


if __name__ == "__main__":
    unittest.main()
