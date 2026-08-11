"""OAuth 单元测试 —— mock httpx/回调，不发起真实网络请求"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import patch

from minimate.mcp.oauth import OAuthClient, OAuthError, _CallbackHandler


def _resp(status_code, json_data=None, headers=None):
    class R:
        def __init__(self):
            self.status_code = status_code
            self.headers = headers or {}

        def json(self):
            return json_data or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise OAuthError(f"HTTP {self.status_code}")

    return R()


AUTH_META = {
    "authorization_endpoint": "https://auth.example.com/authorize",
    "token_endpoint": "https://auth.example.com/token",
}

DEVICE_META = {
    "device_authorization_endpoint": "https://auth.example.com/device",
    "token_endpoint": "https://auth.example.com/token",
}


class OAuthClientTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = os.path.join(self.tmp.name, "tokens.json")
        self.client = OAuthClient(
            client_id="test-client",
            server_url="https://mcp.example.com/mcp",
            scope="openid offline_access",
            token_cache_path=self.cache,
        )

    def tearDown(self):
        self.tmp.cleanup()

    # ---------- 端点发现 ----------

    def test_discover_metadata_two_level(self):
        """两级发现：401 → resource_metadata → authorization_servers → 授权服务器 metadata"""
        with patch("minimate.mcp.oauth.httpx.get") as mock_get:
            mock_get.side_effect = [
                _resp(401, headers={
                    "www-authenticate": 'Bearer realm="OAuth", resource_metadata="https://mcp.example.com/.well-known/oauth-protected-resource/mcp"',
                }),
                _resp(200, {
                    "authorization_servers": ["https://auth.example.com"],
                }),
                _resp(200, AUTH_META),
            ]
            meta = self.client._discover_metadata()
        self.assertEqual(meta["token_endpoint"], "https://auth.example.com/token")
        self.assertEqual(mock_get.call_count, 3)

    def test_discover_without_401(self):
        with patch("minimate.mcp.oauth.httpx.get", return_value=_resp(200, {})):
            with self.assertRaises(OAuthError):
                self.client._discover_metadata()

    def test_discover_missing_auth_servers(self):
        with patch("minimate.mcp.oauth.httpx.get") as mock_get:
            mock_get.side_effect = [
                _resp(401, headers={
                    "www-authenticate": 'Bearer resource_metadata="https://mcp.example.com/meta"',
                }),
                _resp(200, {"resource": "x"}),
            ]
            with self.assertRaises(OAuthError):
                self.client._discover_metadata()

    # ---------- 设备流 ----------

    def test_device_flow_full(self):
        with patch.object(self.client, "_discover_metadata", return_value=DEVICE_META), \
             patch("minimate.mcp.oauth.httpx.post") as mock_post, \
             patch("minimate.mcp.oauth.time.sleep"), \
             patch("minimate.mcp.oauth.webbrowser.open"):
            mock_post.side_effect = [
                _resp(200, {
                    "device_code": "dc-1",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://auth.example.com/verify",
                    "interval": 1,
                    "expires_in": 300,
                }),
                _resp(400, {"error": "authorization_pending"}),
                _resp(200, {"access_token": "at-1", "refresh_token": "rt-1", "expires_in": 3600}),
            ]
            token = self.client.get_access_token()

        self.assertEqual(token, "at-1")
        with open(self.cache, encoding="utf-8") as f:
            cached = json.load(f)
        self.assertEqual(cached["access_token"], "at-1")

    def test_device_access_denied(self):
        with patch.object(self.client, "_discover_metadata", return_value=DEVICE_META), \
             patch("minimate.mcp.oauth.httpx.post") as mock_post, \
             patch("minimate.mcp.oauth.time.sleep"), \
             patch("minimate.mcp.oauth.webbrowser.open"):
            mock_post.side_effect = [
                _resp(200, {
                    "device_code": "dc", "user_code": "CODE",
                    "verification_uri": "https://auth.example.com/verify",
                }),
                _resp(400, {"error": "access_denied"}),
            ]
            with self.assertRaises(OAuthError):
                self.client.get_access_token()

    # ---------- 授权码 + PKCE ----------

    def test_auth_code_flow(self):
        """授权码流：PKCE + loopback 回调 + code 换 token"""
        class FakeServer:
            def __init__(self, addr, handler):
                self.server_address = ("127.0.0.1", 43210)

            def handle_request(self):
                _CallbackHandler.code = "auth-code-1"
                _CallbackHandler.state = "FIXED_STATE"
                _CallbackHandler.error = None

            def server_close(self):
                pass

        with patch.object(self.client, "_discover_metadata", return_value=AUTH_META), \
             patch("minimate.mcp.oauth.HTTPServer", return_value=FakeServer(None, None)) as mock_server, \
             patch("minimate.mcp.oauth.secrets.token_urlsafe", side_effect=["VERIFIER_STR", "FIXED_STATE"]), \
             patch("minimate.mcp.oauth.httpx.post") as mock_post, \
             patch("minimate.mcp.oauth.webbrowser.open"):
            mock_post.return_value = _resp(200, {
                "access_token": "at-code", "refresh_token": "rt", "expires_in": 3600,
            })
            token = self.client.get_access_token()

        self.assertEqual(token, "at-code")
        # token 交换参数：authorization_code + PKCE verifier
        post_data = mock_post.call_args.kwargs["data"]
        self.assertEqual(post_data["grant_type"], "authorization_code")
        self.assertEqual(post_data["code"], "auth-code-1")
        self.assertEqual(post_data["code_verifier"], "VERIFIER_STR")

    # ---------- 动态注册 ----------

    def test_dynamic_registration(self):
        client = OAuthClient(
            client_id="YOUR_PLACEHOLDER",
            server_url="https://mcp.example.com/mcp",
            token_cache_path=self.cache,
        )
        meta = dict(AUTH_META, registration_endpoint="https://auth.example.com/register")
        with patch("minimate.mcp.oauth.httpx.post", return_value=_resp(201, {"client_id": "reg-client-1"})):
            cid = client._ensure_client_id(meta)
        self.assertEqual(cid, "reg-client-1")
        self.assertEqual(client.client_id, "reg-client-1")

    def test_dynamic_registration_fail(self):
        client = OAuthClient(
            client_id="YOUR_PLACEHOLDER",
            server_url="https://mcp.example.com/mcp",
            token_cache_path=self.cache,
        )
        meta = dict(AUTH_META, registration_endpoint="https://auth.example.com/register")
        with patch("minimate.mcp.oauth.httpx.post", return_value=_resp(400, {"error": "invalid_client_metadata"})):
            with self.assertRaises(OAuthError):
                client._ensure_client_id(meta)

    # ---------- 缓存与刷新 ----------

    def test_cache_hit(self):
        with open(self.cache, "w", encoding="utf-8") as f:
            json.dump({
                "access_token": "cached-at",
                "refresh_token": "rt",
                "expires_at": time.time() + 3600,
                "client_id": "test-client",
            }, f)
        with patch("minimate.mcp.oauth.httpx.get") as mock_get:
            token = self.client.get_access_token()
        mock_get.assert_not_called()
        self.assertEqual(token, "cached-at")

    def test_placeholder_client_id_restored_from_cache(self):
        """占位 client_id + 缓存含已注册 client_id → 恢复后命中缓存，不重复注册"""
        with open(self.cache, "w", encoding="utf-8") as f:
            json.dump({
                "access_token": "cached-at",
                "refresh_token": "rt",
                "expires_at": time.time() + 3600,
                "client_id": "reg-client-1",
            }, f)
        client = OAuthClient(
            client_id="YOUR_NOTION_OAUTH_CLIENT_ID",
            server_url="https://mcp.example.com/mcp",
            token_cache_path=self.cache,
        )
        with patch("minimate.mcp.oauth.httpx.get") as mock_get:
            token = client.get_access_token()
        mock_get.assert_not_called()
        self.assertEqual(token, "cached-at")
        self.assertEqual(client.client_id, "reg-client-1")

    def test_refresh_flow(self):
        with open(self.cache, "w", encoding="utf-8") as f:
            json.dump({
                "access_token": "old-at",
                "refresh_token": "rt-1",
                "expires_at": time.time() - 10,
                "client_id": "test-client",
            }, f)
        with patch.object(self.client, "_discover_metadata", return_value=AUTH_META), \
             patch("minimate.mcp.oauth.httpx.post") as mock_post:
            mock_post.return_value = _resp(200, {
                "access_token": "new-at", "refresh_token": "rt-2", "expires_in": 3600,
            })
            token = self.client.get_access_token()
        self.assertEqual(token, "new-at")
        self.assertEqual(mock_post.call_args.kwargs["data"]["grant_type"], "refresh_token")


if __name__ == "__main__":
    unittest.main()
