"""BM25 关键词检索 —— 与余弦向量做 RRF 融合，构成混合检索

Okapi BM25：基于词频（TF）与逆文档频率（IDF）的关键词打分，
擅长精确标识符/符号查询；与向量召回按排名倒数（RRF）融合。
"""

import math
from collections import Counter

from .embeddings import tokenize


class BM25:
    """Okapi BM25 打分器（k1=1.5, b=0.75 为常见默认值）"""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._tokenized = [tokenize(d) for d in docs]
        self._df: Counter = Counter()
        for doc in self._tokenized:
            self._df.update(set(doc))

    @property
    def N(self) -> int:
        return len(self._tokenized)

    @property
    def avgdl(self) -> float:
        total = sum(len(d) for d in self._tokenized)
        return total / max(1, self.N)

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        return math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query: str) -> list[float]:
        """返回每个文档的 BM25 得分（与传入 docs 顺序一致）"""
        q_tokens = tokenize(query)
        if not q_tokens:
            return [0.0] * self.N
        scores: list[float] = []
        for doc in self._tokenized:
            tf = Counter(doc)
            dl = len(doc)
            total = 0.0
            for t in q_tokens:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                total += self._idf(t) * f * (self.k1 + 1) / denom
            scores.append(total)
        return scores
