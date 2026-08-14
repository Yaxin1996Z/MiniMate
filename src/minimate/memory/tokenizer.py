"""检索分词器 —— 轻量中文分词（零新依赖）

- 英文保留完整单词（>=3 字符）
- 中文连续串切成重叠双字（如"我家猫咪叫什么" → 我家/家猫/猫咪/咪叫/叫什/什么）
"""

import re

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_CN_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """把查询文本切分为检索 token（去重、保序）"""
    if not text:
        return []
    words = _WORD_RE.findall(text.lower())
    for run in _CN_RUN_RE.findall(text):
        words.extend(run[i : i + 2] for i in range(len(run) - 1))
    seen: set[str] = set()
    result: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            result.append(w)
    return result


def matches(text: str, tokens: list[str]) -> bool:
    """文本是否包含任意一个 token（子串匹配）"""
    if not text or not tokens:
        return False
    lowered = text.lower()
    return any(t in lowered for t in tokens)
