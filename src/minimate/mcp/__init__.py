"""MCP 包 —— 本地 stdio 与远程 HTTP 双传输适配

结构：
  adapter.py - McpToolAdapter：统一适配器（按 transport 分派连接）
  stdio.py   - stdio 传输：本地子进程连接（进程间通信）
  http.py    - Streamable HTTP 传输：远程 URL 连接
  oauth.py   - OAuth 2.0 客户端（授权码 PKCE / 设备流 / 动态注册）
"""

from .adapter import McpToolAdapter
from .oauth import OAuthClient, OAuthError

__all__ = ["McpToolAdapter", "OAuthClient", "OAuthError"]
