"""日志与审计 —— 统一 logger + 工具调用审计

输出：
  - 控制台：同格式日志
  - 文件：logs/minimate.log（UTF-8，按天轮转）

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


def setup_logger(name: str = "minimate", level: int | None = None) -> logging.Logger:
    """初始化（或复用）全局 logger：控制台 + 按天轮转文件"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level if level is not None else logging.INFO)
    formatter = logging.Formatter(_FORMAT)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

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
