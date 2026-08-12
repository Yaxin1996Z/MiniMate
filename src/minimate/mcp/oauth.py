"""OAuth 2.0 客户端 —— 支持授权码+PKCE（loopback 回调）与设备授权流

自动选择流程（根据授权服务器 metadata）：
  - 有 device_authorization_endpoint → 设备流（RFC 8628）
  - 否则 → 授权码 + PKCE + 本地 loopback 回调（如 Notion MCP）

端点发现（两级）：
  1. 请求 MCP Server → 401 + WWW-Authenticate 的 resource_metadata（RFC 9728）
  2. resource metadata → authorization_servers → 授权服务器 metadata（RFC 8414）
"""

import base64
import hashlib
import json
import os
import re
import secrets
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import quote, urlparse, parse_qs

import httpx

from ..logging import logger


class OAuthError(Exception):
    """OAuth 流程错误"""


def _cache_path() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "tokens.json")


class _CallbackHandler(BaseHTTPRequestHandler):
    """本地 loopback 回调：捕获 /callback?code=xxx&state=xxx"""

    code: str | None = None
    state: str | None = None
    error: str | None = None

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        if "code" in query:
            type(self).code = query["code"][0]
            type(self).state = query.get("state", [None])[0]
            body = "<h1>授权成功，可以关闭此页面</h1>".encode("utf-8")
        else:
            type(self).error = query.get("error", ["missing_code"])[0]
            body = f"<h1>授权失败：{type(self).error}</h1>".encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class OAuthClient:
    """OAuth 客户端（同步实现，供 MCP HTTP 连接前获取 token）"""

    def __init__(
        self,
        client_id: str = "",
        server_url: str = "",
        scope: str = "",
        token_cache_path: str = "",
    ):
        self.client_id = client_id
        self.server_url = server_url
        self.scope = scope
        self.token_cache_path = token_cache_path or _cache_path()
        self._metadata: dict | None = None

    # ----------------------------------------------------------
    # 公开入口
    # ----------------------------------------------------------

    def get_access_token(self) -> str:
        """返回有效 access_token：缓存命中 → refresh 刷新 → 授权流程"""
        self._restore_client_id()
        token = self._load_cache()
        if token:
            if token.get("expires_at", 0) > time.time() + 60:
                logger.info("OAuth 使用缓存 token（%d 秒后过期）", int(token["expires_at"] - time.time()))
                return token["access_token"]
            if token.get("refresh_token"):
                try:
                    return self._refresh(token["refresh_token"])
                except OAuthError as e:
                    logger.warning("OAuth refresh_token 刷新失败：%s，重新授权", e)
        return self._authorize()

    def _restore_client_id(self) -> None:
        """client_id 为占位符时，从缓存恢复已注册的 client_id（避免每次重复动态注册）"""
        if self.client_id and not self.client_id.startswith("YOUR_"):
            return
        token = self._load_cache_raw()
        if token and token.get("client_id") and not token["client_id"].startswith("YOUR_"):
            self.client_id = token["client_id"]
            logger.info("OAuth 从缓存恢复 client_id=%s", self.client_id)

    # ----------------------------------------------------------
    # 端点发现（两级，RFC 9728 + RFC 8414）
    # ----------------------------------------------------------

    def _discover_metadata(self) -> dict:
        if self._metadata:
            return self._metadata

        try:
            resp = httpx.get(self.server_url, timeout=15)
        except httpx.HTTPError as e:
            raise OAuthError(f"无法访问 MCP Server：{e}") from e

        if resp.status_code != 401:
            raise OAuthError(f"服务器未要求 OAuth 认证（状态码 {resp.status_code}）")

        auth_header = resp.headers.get("www-authenticate", "")
        m = re.search(r'resource_metadata="([^"]+)"', auth_header)
        if not m:
            raise OAuthError("WWW-Authenticate 头缺少 resource_metadata，无法发现授权端点")

        try:
            resource_meta = httpx.get(m.group(1), timeout=15).json()
            auth_servers = resource_meta.get("authorization_servers", [])
            if not auth_servers:
                raise OAuthError("resource metadata 缺少 authorization_servers")
            auth_server = auth_servers[0].rstrip("/")
            meta_url = auth_server + "/.well-known/oauth-authorization-server"
            meta = httpx.get(meta_url, timeout=15).json()
        except OAuthError:
            raise
        except (httpx.HTTPError, ValueError) as e:
            raise OAuthError(f"获取授权服务器 metadata 失败：{e}") from e

        if not meta.get("token_endpoint"):
            raise OAuthError("授权服务器 metadata 缺少 token_endpoint")
        self._metadata = meta
        return meta

    # ----------------------------------------------------------
    # 授权流程自动选择
    # ----------------------------------------------------------

    def _authorize(self) -> str:
        meta = self._discover_metadata()
        if meta.get("device_authorization_endpoint"):
            return self._authorize_device(meta)
        return self._authorize_code(meta)

    # ----------------------------------------------------------
    # 设备授权流（RFC 8628，兼容支持该流程的 Server）
    # ----------------------------------------------------------

    def _authorize_device(self, meta: dict) -> str:
        da_endpoint = meta["device_authorization_endpoint"]
        token_endpoint = meta["token_endpoint"]
        client_id = self._ensure_client_id(meta)

        try:
            resp = httpx.post(da_endpoint, data={
                "client_id": client_id,
                "scope": self.scope,
            }, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise OAuthError(f"请求设备码失败：{e}") from e

        device_code = data.get("device_code")
        user_code = data.get("user_code")
        verification_uri = data.get("verification_uri") or data.get("verification_url")
        if not device_code or not user_code or not verification_uri:
            raise OAuthError("设备授权响应缺少必要字段")

        interval = int(data.get("interval", 5))
        expires_in = int(data.get("expires_in", 600))

        print(f"  [OAuth] 请在浏览器打开：{verification_uri}")
        print(f"  [OAuth] 输入授权码：{user_code}")
        try:
            webbrowser.open(verification_uri)
        except Exception:
            pass

        deadline = time.time() + expires_in
        while time.time() < deadline:
            time.sleep(interval)
            try:
                r = httpx.post(token_endpoint, data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": device_code,
                    "client_id": client_id,
                }, timeout=30)
                data = r.json()
            except (httpx.HTTPError, ValueError) as e:
                raise OAuthError(f"轮询 token 失败：{e}") from e

            if r.status_code == 200 and data.get("access_token"):
                return self._save_token(data)

            error = data.get("error", "")
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5
                continue
            if error == "access_denied":
                raise OAuthError("用户拒绝了授权")
            if error == "expired_token":
                raise OAuthError("设备码已过期，请重新运行")
            raise OAuthError(f"轮询失败：{data}")

        raise OAuthError("授权超时，未在有效期内完成")

    # ----------------------------------------------------------
    # 授权码 + PKCE + loopback 回调（Notion 等）
    # ----------------------------------------------------------

    def _authorize_code(self, meta: dict) -> str:
        auth_endpoint = meta.get("authorization_endpoint")
        token_endpoint = meta["token_endpoint"]
        if not auth_endpoint:
            raise OAuthError("授权服务器 metadata 缺少 authorization_endpoint")
        client_id = self._ensure_client_id(meta)

        # PKCE
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("utf-8")).digest()
        ).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)

        # 本地 loopback 回调服务器
        _CallbackHandler.code = None
        _CallbackHandler.state = None
        _CallbackHandler.error = None
        server = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
        port = server.server_address[1]
        redirect_uri = f"http://127.0.0.1:{port}/callback"

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": self.scope,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        sep = "&" if "?" in auth_endpoint else "?"
        auth_url = auth_endpoint + sep + "&".join(
            f"{k}={quote(str(v))}" for k, v in params.items()
        )

        print(f"  [OAuth] 请在浏览器中完成授权（若未自动打开请访问）：")
        print(f"  [OAuth] {auth_url}")
        print(f"  [OAuth] 等待回调：{redirect_uri}（180 秒超时）")
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

        # 等待回调（单次 handle_request，用户授权后浏览器重定向回来）
        server_thread = threading.Thread(target=server.handle_request, daemon=True)
        server_thread.start()
        server_thread.join(timeout=180)
        server.server_close()

        if _CallbackHandler.error:
            raise OAuthError(f"授权失败：{_CallbackHandler.error}")
        if not _CallbackHandler.code:
            raise OAuthError("授权超时：未在 180 秒内完成授权")
        if _CallbackHandler.state != state:
            raise OAuthError("state 校验失败：回调状态不匹配")

        # 用授权码 + PKCE verifier 换 token
        try:
            resp = httpx.post(token_endpoint, data={
                "grant_type": "authorization_code",
                "code": _CallbackHandler.code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "code_verifier": verifier,
            }, timeout=30)
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise OAuthError(f"换取 token 失败：{e}") from e

        if resp.status_code != 200 or not data.get("access_token"):
            raise OAuthError(f"换取 token 被拒绝：{data.get('error', 'unknown')}")
        return self._save_token(data)

    # ----------------------------------------------------------
    # 客户端注册 / 刷新 / 缓存
    # ----------------------------------------------------------

    def _ensure_client_id(self, meta: dict) -> str:
        """配置了有效 client_id 直接用；否则尝试动态注册（RFC 7591）"""
        if self.client_id and not self.client_id.startswith("YOUR_"):
            return self.client_id

        reg_endpoint = meta.get("registration_endpoint")
        if not reg_endpoint:
            raise OAuthError(
                "未配置有效的 client_id，且服务器不支持动态注册。"
                "请创建 OAuth Integration 并将 client_id 填入 config.json"
            )

        try:
            resp = httpx.post(reg_endpoint, json={
                "client_name": "MiniMate",
                "redirect_uris": ["http://127.0.0.1/callback"],
                "grant_types": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_method": "none",
            }, timeout=30)
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise OAuthError(f"动态注册失败：{e}") from e

        if resp.status_code >= 400 or not data.get("client_id"):
            raise OAuthError(
                f"动态注册失败：{data}。请手动创建 OAuth Integration 并配置 client_id"
            )
        self.client_id = data["client_id"]
        logger.info("OAuth 动态注册成功 client_id=%s", self.client_id)
        return self.client_id

    def _refresh(self, refresh_token: str) -> str:
        meta = self._discover_metadata()
        token_endpoint = meta["token_endpoint"]
        client_id = self.client_id or self._ensure_client_id(meta)
        try:
            resp = httpx.post(token_endpoint, data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            }, timeout=30)
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            raise OAuthError(f"刷新 token 失败：{e}") from e

        if resp.status_code != 200 or not data.get("access_token"):
            raise OAuthError(f"刷新 token 被拒绝：{data.get('error', 'unknown')}")
        return self._save_token(data)

    def _save_token(self, data: dict) -> str:
        token = {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token", ""),
            "expires_at": time.time() + int(data.get("expires_in", 3600)),
            "client_id": self.client_id,
            "scope": self.scope,
            "saved_at": time.time(),
        }
        os.makedirs(os.path.dirname(self.token_cache_path), exist_ok=True)
        with open(self.token_cache_path, "w", encoding="utf-8") as f:
            json.dump(token, f, ensure_ascii=False, indent=2)
        logger.info("OAuth token 已保存：%s", self.token_cache_path)
        return token["access_token"]

    def _load_cache(self) -> dict | None:
        token = self._load_cache_raw()
        if token and token.get("client_id") == self.client_id:
            return token
        return None

    def _load_cache_raw(self) -> dict | None:
        if not os.path.exists(self.token_cache_path):
            return None
        try:
            with open(self.token_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None
