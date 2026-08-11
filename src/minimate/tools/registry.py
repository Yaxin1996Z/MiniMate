"""工具注册中心 —— 对应 ToolRegistry 构造函数的按类注册调用序列"""

from .core import ToolExecutor
from .file_tools import register_file_tools
from .shell_tools import register_shell_tools
from .web_tools import register_web_tools
from .rag_tools import register_rag_tools


def register_all_tools(executor: ToolExecutor) -> None:
    """注册全部内置工具（本地工具源），按类分组注册

    对应 Java ToolRegistry 构造函数中的调用序列：
      registerFileTools(); registerShellTools(); registerCodeTools(); ...
    """
    register_file_tools(executor)
    register_shell_tools(executor)
    register_web_tools(executor)
    register_rag_tools(executor)
