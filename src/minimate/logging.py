"""日志与审计 —— 统一 logger + 工具调用审计

输出：
  - 文件：logs/minimate.log（UTF-8，按天轮转）
  - 控制台：默认关闭（避免刷屏），cli --verbose 或环境变量
    MINIMATE_CONSOLE_LOG=1 开启

审计约定：
  - 工具调用：logger.info("TOOL_CALL tool=... args=... elapsed=...s result=...")
  - 危险工具（write_file / run_shell / save_file）：logger.warning("DANGEROUS_TOOL ...")
  - 错误：logger.error(...)
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler


def _log_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "..", "..", "logs")


def _log_file() -> str:
    return os.path.join(_log_dir(), "minimate.log")


_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logger(
    name: str = "minimate",
    level: int | None = None,
    console: bool | None = None,
) -> logging.Logger:
    """初始化（或复用）全局 logger

    console=True 时追加控制台输出（默认关闭，避免启动日志刷屏）；
    文件输出始终开启（按天轮转）。
    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter(_FORMAT)

    if not logger.handlers:
        logger.setLevel(level if level is not None else logging.INFO)
        os.makedirs(_log_dir(), exist_ok=True)
        file_handler = TimedRotatingFileHandler(
            _log_file(),
            when="midnight",
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False

    # 控制台：默认关闭；显式开启或环境变量开启时追加
    if console is None:
        console = os.getenv("MINIMATE_CONSOLE_LOG") == "1"
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if console and not has_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


logger = setup_logger()


# 需要审计的危险工具（会修改系统状态）
DANGEROUS_TOOLS = {"write_file", "run_shell", "save_file"}


def audit_tool_call(tool_name: str, args: str, result: str, elapsed: float) -> None:
    """记录一次工具调用审计（含耗时与结果摘要）"""
    result_preview = (result or "").replace("\n", " ")[:120]
    if tool_name in DANGEROUS_TOOLS:
        logger.warning(
            "DANGEROUS_TOOL tool=%s args=%s elapsed=%.2fs result=%s",
            tool_name, args, elapsed, result_preview,
        )
    else:
        logger.info(
            "TOOL_CALL tool=%s args=%s elapsed=%.2fs result=%s",
            tool_name, args, elapsed, result_preview,
        )


def log_path() -> str:
    """日志文件路径（供 CLI 提示）"""
    return _log_file()
