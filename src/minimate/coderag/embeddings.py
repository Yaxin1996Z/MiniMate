"""代码向量化 —— 可插拔 EmbeddingProvider

默认使用真实语义模型 bge-m3（sentence-transformers，懒加载），
模型不可用时自动回退到零依赖的哈希词频向量（md5 → 2048 维）。
"""

import hashlib
import math
import os
import re


DIM = 2048
DEFAULT_MODEL_PATH = "D:/Documents/Models/Embedding/bge-m3"


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


class EmbeddingProvider:
    """真实语义 embedding（默认 bge-m3，懒加载；模型不可用时回退哈希向量）"""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.getenv(
            "MINIMATE_EMBED_MODEL", DEFAULT_MODEL_PATH
        )
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_path)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """批量编码（L2 归一化，余弦=点积）；模型加载失败回退哈希向量"""
        try:
            vecs = self._load().encode(
                [t or "" for t in texts],
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
            return [v.tolist() for v in vecs]
        except Exception:
            return [embed(t) for t in texts]

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


class HashEmbeddingProvider:
    """轻量哈希向量 Provider（测试 / 无模型兜底）"""

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [embed(t) for t in texts]

    def encode_one(self, text: str) -> list[float]:
        return embed(text)


_provider: EmbeddingProvider | None = None


def get_provider() -> EmbeddingProvider:
    """全局共享的 embedding Provider（进程内只加载一次模型）"""
    global _provider
    if _provider is None:
        _provider = EmbeddingProvider()
    return _provider
