"""ANSI 彩色输出 —— 自动检测终端，非终端环境自动降级为纯文本"""

import os
import sys


def _enable_windows_vt() -> bool:
    """Windows 下启用 ANSI VT 处理（Windows 10+），失败返回 False"""
    if os.name != "nt" or not sys.stdout.isatty():
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False


class Color:
    """终端颜色工具：paint(text, fg, bold)"""

    FG = {
        "cyan": "36",
        "green": "32",
        "yellow": "33",
        "red": "31",
        "magenta": "35",
        "blue": "34",
    }

    def __init__(self, enabled: bool | None = None):
        if enabled is None:
            enabled = sys.stdout.isatty()
            if enabled:
                _enable_windows_vt()  # Windows 下确保 VT 转义可用
        self.enabled = enabled

    def paint(self, text: str, fg: str = "", bold: bool = False) -> str:
        if not self.enabled or not text:
            return text
        codes = []
        if bold:
            codes.append("1")
        if fg in self.FG:
            codes.append(self.FG[fg])
        if not codes:
            return text
        return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"

    # 便捷方法
    def cyan(self, text: str, bold: bool = False) -> str:
        return self.paint(text, "cyan", bold)

    def green(self, text: str, bold: bool = False) -> str:
        return self.paint(text, "green", bold)

    def yellow(self, text: str, bold: bool = False) -> str:
        return self.paint(text, "yellow", bold)

    def red(self, text: str, bold: bool = False) -> str:
        return self.paint(text, "red", bold)


# 全局实例（按当前 stdout 状态决定是否启用）
color = Color()
