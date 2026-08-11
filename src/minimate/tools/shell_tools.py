"""Shell 命令工具 —— 对应 ToolRegistry.registerShellTools"""

import os
import subprocess

from .core import ToolExecutor, _first_line, tool


@tool(name="run_shell", description="在终端执行 shell 命令（真实执行，超时 30 秒），参数为命令字符串，返回退出码与输出。用于运行脚本、查看系统信息等")
def run_shell(command: str) -> str:
    """执行 shell 命令，带超时保护与输出截断"""
    command = _first_line(command)
    if not command:
        return "错误：请提供要执行的命令"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "[命令超时] 执行超过 30 秒已终止"
    except Exception as e:
        return f"[命令执行错误] {e}"

    output = (result.stdout or "").strip()
    if result.stderr and result.stderr.strip():
        output += ("\n" if output else "") + result.stderr.strip()
    if not output:
        output = "(无输出)"
    if len(output) > 3000:
        output = output[:3000] + f"\n...[输出过长，已截断 {len(output) - 3000} 字符]"
    text = f"退出码：{result.returncode}\n{output}"
    if result.returncode != 0:
        text += f"\n（提示：当前系统为 {'Windows cmd' if os.name == 'nt' else 'Linux/macOS bash'}，命令可能不适配，可换用系统原生命令）"
    return text


def register_shell_tools(executor: ToolExecutor) -> None:
    """注册 Shell 工具（对应 ToolRegistry.registerShellTools）"""
    executor.register(run_shell)
