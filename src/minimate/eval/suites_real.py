"""真实用户用例 —— 来自 MiniMate 实际使用场景（区别于 AI 生成的 suites_ai）

真实用例以用户真实任务为蓝本，判定仍然使用确定性 checker，
保证可复现；文件路径类用例直接验证产物落盘与内容。
"""

from __future__ import annotations

from .case import EvalCase


REAL_SUITE: list[EvalCase] = [
    EvalCase(
        id="real_multi_001",
        name="真实任务：分析项目结构并生成文档",
        mode="multi",
        prompt=(
            "当前项目的结构是什么样的，分析一下，"
            "写到 D:/Documents/Note/Obsidian/JobAssistant/"
            "投递管理/面试准备/minimate/minimate.md 中"
        ),
        checker="command_exit0",
        expected=(
            "python -c \"import os; "
            "p = r'D:/Documents/Note/Obsidian/JobAssistant/投递管理/面试准备/minimate/minimate.md'; "
            "assert os.path.exists(p), '文件不存在'; "
            "t = open(p, encoding='utf-8').read(); "
            "for k in ['项目结构', '执行模式', '记忆', 'RAG 检索']: "
            "assert k in t, '缺少关键词: ' + k; "
            "print('ok')\""
        ),
        tags=["多Agent", "真实任务"],
    ),
]


def list_suites() -> list[str]:
    """真实用例集名称"""
    return ["real"]


def get_suite(name: str = "real") -> list[EvalCase]:
    """获取真实用例集"""
    if name != "real":
        raise ValueError(f"真实用例集仅支持 real，收到：{name}")
    return REAL_SUITE
