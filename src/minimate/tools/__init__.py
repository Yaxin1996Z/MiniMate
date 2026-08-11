"""工具系统包 —— Tool 定义 + 执行器 + 按类分组的工具注册

结构：
  core.py        - Tool / ToolExecutor / ReAct 解析（核心定义）
  file_tools.py  - 文件操作工具（register_file_tools）
  shell_tools.py - Shell 命令工具（register_shell_tools）
  web_tools.py   - 互联网搜索工具（register_web_tools）
  rag_tools.py   - RAG 知识库工具（register_rag_tools）
  registry.py    - 注册中心（register_all_tools）
"""

from .core import Tool, ToolExecutor, classify_error, parse_react, tool, truncate
from .file_tools import (
    find_files,
    grep_files,
    list_files,
    read_file,
    save_file,
    write_file,
)
from .shell_tools import run_shell
from .web_tools import web_search
from .rag_tools import query_knowledge
from .registry import register_all_tools

__all__ = [
    "Tool",
    "ToolExecutor",
    "tool",
    "classify_error",
    "parse_react",
    "truncate",
    "read_file",
    "write_file",
    "list_files",
    "save_file",
    "find_files",
    "grep_files",
    "run_shell",
    "web_search",
    "query_knowledge",
    "register_all_tools",
]
