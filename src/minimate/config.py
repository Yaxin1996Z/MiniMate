"""配置模块 —— 从项目根 config.json 加载配置

支持配置项：
  mcp.servers  - MCP 服务器列表，每项含 name/command/args/env
                 配置了 MCP 时，启动会加载对应服务器暴露的工具

config.json 缺失或解析失败时自动回退到默认配置（无 MCP）。
"""

import json
import os
from typing import Any


def _strip_json_comments(text: str) -> str:
    """移除 JSON 中的 // 行注释与 /* */ 块注释（字符串内的 // 不受影响）。"""
    out: list[str] = []
    i, n = 0, len(text)
    in_string = False
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_string:
            out.append(ch)
            if ch == "\\" and nxt:
                out.append(nxt)
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(i + 2, n)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _config_path() -> str:
    """项目根目录下的 config.json"""
    return os.path.join(os.path.dirname(__file__), "..", "..", "config.json")


def _default_config() -> dict:
    return {
        "mcp": {
            "servers": [],
        }
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并：override 覆盖 base（dict 逐层合并，其他类型直接覆盖）"""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str = "") -> dict:
    """加载配置：config.json 与默认配置合并；文件缺失/损坏时回退默认"""
    cfg = _default_config()
    file = path or _config_path()
    if not os.path.exists(file):
        return cfg
    try:
        with open(file, "r", encoding="utf-8") as f:
            user = json.loads(_strip_json_comments(f.read()))
        if isinstance(user, dict):
            cfg = _deep_merge(cfg, user)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  config.json 解析失败（{e}），使用默认配置")
    return cfg


def get_mcp_servers(config: dict | None = None) -> list[dict]:
    """返回配置中的 MCP 服务器列表"""
    cfg = config if config is not None else load_config()
    return cfg.get("mcp", {}).get("servers", []) or []


def get_tool_config(config: dict | None = None) -> dict:
    """预留：其他工具相关配置（当前仅 mcp）"""
    return {}
