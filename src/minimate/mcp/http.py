"""Streamable HTTP 传输 —— 与远程 MCP Server 的网络连接

通过 HTTP URL（TCP 网络）与远程 MCP Server 通信，支持认证 headers。
"""

from mcp.client.streamable_http import streamable_http_client

from .oauth import OAuthClient


def create_client_ctx(config: dict):
    """创建 HTTP 客户端上下文（async context manager）

    config 需含 url，可选 headers（如 Authorization Bearer token）。
    若配置了 oauth（client_id 等），先执行 OAuth 设备授权流获取 token。
    返回 (ctx, http_client)：http_client 由调用方负责关闭。
    """
    url = config.get("url")
    if not url:
        raise ValueError("http 传输需要 url 配置")

    headers = dict(config.get("headers") or {})
    oauth_cfg = config.get("oauth")
    if oauth_cfg:
        client_id = oauth_cfg.get("client_id")
        if not client_id:
            raise ValueError("oauth 配置缺少 client_id（请在 Notion 创建 Integration 获取）")
        oauth_client = OAuthClient(
            client_id=client_id,
            server_url=url,
            scope=oauth_cfg.get("scope", ""),
        )
        headers["Authorization"] = f"Bearer {oauth_client.get_access_token()}"

    http_client = None
    kwargs = {}
    if headers:
        import httpx

        http_client = httpx.AsyncClient(headers=headers)
        kwargs["http_client"] = http_client
    return streamable_http_client(url, **kwargs), http_client
