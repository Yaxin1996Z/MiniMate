"""确定性断言检查器 —— 判定 Agent 任务是否达成

不使用 LLM 打分，全部为可复现的确定性校验（文件存在 / 内容包含 / 数值比对 /
命令退出码 / 输出关键词），保证评测可信。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from typing import Optional


def run_checker(checker: str, expected: str, output: str = "", cwd: str = "") -> tuple[bool, str]:
    """执行一个断言检查器，返回 (是否通过, 判定理由)"""
    handlers = {
        "file_exists": _check_file_exists,
        "file_contains": _check_file_contains,
        "dir_contains": _check_dir_contains,
        "output_contains": _check_output_contains,
        "math_answer": _check_math_answer,
        "command_exit0": _check_command_exit0,
        "grep_finds": _check_grep_finds,
    }
    handler = handlers.get(checker)
    if handler is None:
        return False, f"未知检查器：{checker}"
    try:
        return handler(expected, output, cwd or os.getcwd())
    except Exception as e:
        return False, f"检查器执行异常：{e}"


def _check_file_exists(expected: str, output: str, cwd: str) -> tuple[bool, str]:
    path = os.path.join(cwd, expected)
    ok = os.path.isfile(path)
    return ok, (f"文件存在：{expected}" if ok else f"文件不存在：{expected}")


def _check_file_contains(expected: str, output: str, cwd: str) -> tuple[bool, str]:
    if "|" not in expected:
        return False, f"file_contains 期望格式应为 路径|关键词，实际：{expected}"
    rel_path, keyword = expected.split("|", 1)
    path = os.path.join(cwd, rel_path.strip())
    if not os.path.isfile(path):
        return False, f"文件不存在：{rel_path.strip()}"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    ok = keyword.strip() in content
    return ok, (
        f"文件 {rel_path.strip()} 包含关键词「{keyword.strip()}」"
        if ok
        else f"文件 {rel_path.strip()} 未包含关键词「{keyword.strip()}」"
    )


def _check_dir_contains(expected: str, output: str, cwd: str) -> tuple[bool, str]:
    if "|" not in expected:
        return False, f"dir_contains 期望格式应为 目录|文件名，实际：{expected}"
    rel_dir, filename = expected.split("|", 1)
    path = os.path.join(cwd, rel_dir.strip())
    if not os.path.isdir(path):
        return False, f"目录不存在：{rel_dir.strip()}"
    ok = os.path.isfile(os.path.join(path, filename.strip()))
    return ok, (
        f"目录 {rel_dir.strip()} 下存在文件 {filename.strip()}"
        if ok
        else f"目录 {rel_dir.strip()} 下不存在文件 {filename.strip()}"
    )


def _check_output_contains(expected: str, output: str, cwd: str) -> tuple[bool, str]:
    ok = expected.strip() in (output or "")
    return ok, (
        f"输出包含关键词「{expected.strip()}」"
        if ok
        else f"输出未包含关键词「{expected.strip()}」"
    )


def _check_math_answer(expected: str, output: str, cwd: str) -> tuple[bool, str]:
    """从输出中提取数字与期望值比较（容差 0.01）"""
    try:
        target = float(expected)
    except ValueError:
        return False, f"期望值不是数字：{expected}"
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", output or "")]
    if not nums:
        return False, "输出中没有可比较的数字"
    actual = nums[-1]
    ok = abs(actual - target) <= 0.01
    return ok, (
        f"输出数字 {actual} 与期望 {target} 一致"
        if ok
        else f"输出数字 {actual} 与期望 {target} 不一致"
    )


def _check_command_exit0(expected: str, output: str, cwd: str) -> tuple[bool, str]:
    command = _resolve_python(expected)
    proc = subprocess.run(
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode == 0:
        return True, f"命令执行成功（退出码 0）：{command}"
    tail = (proc.stderr or proc.stdout or "")[-300:]
    return False, f"命令执行失败（退出码 {proc.returncode}）：{tail}"


def _resolve_python(command: str) -> str:
    """环境适配：python 不可用（缺失 / WindowsApps 占位 stub）时替换为当前解释器"""
    which = shutil.which("python")
    if which is not None and "WindowsApps" not in str(which):
        return command
    exe = sys.executable or "python"
    return re.sub(
        r"(^|\s)python(?=\s|$)",
        lambda m: f'{m.group(1)}"{exe}"',
        command,
    )


def _check_grep_finds(expected: str, output: str, cwd: str) -> tuple[bool, str]:
    """目录内文件内容包含关键词（模拟 grep -r）"""
    if "|" not in expected:
        return False, f"grep_finds 期望格式应为 目录|关键词，实际：{expected}"
    rel_dir, keyword = expected.split("|", 1)
    root = os.path.join(cwd, rel_dir.strip())
    if not os.path.isdir(root):
        return False, f"目录不存在：{rel_dir.strip()}"
    keyword = keyword.strip()
    hits: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.startswith((".", "__pycache__")):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    if keyword in f.read():
                        hits.append(os.path.relpath(path, cwd))
            except OSError:
                continue
    if hits:
        return True, f"关键词「{keyword}」命中 {len(hits)} 个文件：{', '.join(hits[:5])}"
    return False, f"目录 {rel_dir.strip()} 内未找到包含关键词「{keyword}」的文件"
