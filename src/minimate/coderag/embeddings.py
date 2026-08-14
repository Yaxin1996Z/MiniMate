"""轻量级代码向量化 —— hashing TF 向量 + 余弦相似度（零模型依赖）

面向代码检索场景：不引入 embedding 模型，用稳定哈希（md5）将
token（标识符 / camelCase 拆分 / 中文滑窗）映射到固定维度词频向量，
L2 归一化后以 JSON 数组持久化，检索时在内存计算余弦相似度。
"""

import hashlib
import math
import re


DIM = 2048


def _stable_hash(token: str) -> int:
    """进程间稳定的字符串哈希（Python hash() 因随机种子不可跨进程）"""
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:4], "big")


def tokenize(text: str) -> list[str]:
    """代码/文本分词：英文标识符（camelCase/snake 拆分）+ 中文 2-4 字滑窗"""
    tokens: list[str] = []
    if not text:
        return tokens

    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text):
        word = m.group(0)
        tokens.append(word.lower())
        parts = re.findall(r"[a-z0-9]+|[A-Z][a-z0-9]*", word)
        tokens.extend(p.lower() for p in parts if len(p) >= 2)

    for m in re.finditer(r"[\u4e00-\u9fff]+", text):
        cn = m.group(0)
        if len(cn) < 2:
            continue
        size = min(len(cn), 4)
        for i in range(len(cn) - size + 1):
            tokens.append(cn[i : i + size])

    return tokens


def embed(text: str, dim: int = DIM) -> list[float]:
    """哈希词频向量（L2 归一化）"""
    vec = [0.0] * dim
    for tok in tokenize(text):
        vec[_stable_hash(tok) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（向量已 L2 归一化时等价于点积）"""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
