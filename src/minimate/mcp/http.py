"""Streamable HTTP 传输 —— 与远程 MCP Server 的网络连接

通过 HTTP URL（TCP 网络）与远程 MCP Server 通信，支持认证 headers。
"""

from mcp.client.streamable_http import streamable_http_client


def create_client_ctx(config: dict):
    """创建 HTTP 客户端上下文（async context manager）

    config 需含 url，可选 headers（如 Authorization Bearer token）。
    返回 (ctx, http_client)：http_client 由调用方负责关闭。
    """
    url = config.get("url")
    if not url:
        raise ValueError("http 传输需要 url 配置")
    headers = config.get("headers")
    http_client = None
    kwargs = {}
    if headers:
        import httpx

        http_client = httpx.AsyncClient(headers=headers)
        kwargs["http_client"] = http_client
    return streamable_http_client(url, **kwargs), http_client
