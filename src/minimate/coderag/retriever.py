"""代码检索 —— BM25 + 余弦向量 RRF 混合检索 + 调用链查询"""

from collections import deque

from .bm25 import BM25
from .embeddings import cosine, embed


class CodeRetriever:
    def __init__(self, storage, provider=None):
        self.storage = storage
        self._embed_one = provider.encode_one if provider else embed

    # ----------------------------------------------------------
    # 语义检索（余弦相似度，内存计算）
    # ----------------------------------------------------------

    def search(self, repo: str, query: str, top_k: int = 5) -> list[dict]:
        """混合检索：BM25 关键词召回 + 余弦向量召回，RRF 按排名倒数融合"""
        chunks = self.storage.load_chunks(repo)
        if not chunks:
            return []
        texts = [
            f"{c.get('name', '')}\n{c.get('doc', '')}\n{c.get('code', '')}"
            for c in chunks
        ]
        q_vec = self._embed_one(query)
        bm25 = BM25(texts)
        bm_scores = bm25.score(query)
        cos_scores = [
            cosine(q_vec, c.get("vector") or []) if c.get("vector") else 0.0
            for c in chunks
        ]

        # RRF：多路召回按排名倒数融合（k=60）
        rrf: dict[int, float] = {}
        for scores in (bm_scores, cos_scores):
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            for rank, i in enumerate(order):
                if scores[i] > 0:
                    rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)

        top = sorted(rrf.items(), key=lambda x: -x[1])[:top_k]
        results = []
        for i, rrf_score in top:
            c = dict(chunks[i])
            c["score"] = round(rrf_score, 4)
            c["bm25"] = round(bm_scores[i], 4)
            c["cosine"] = round(cos_scores[i], 4)
            results.append(c)
        return results

    # ----------------------------------------------------------
    # 调用链查询
    # ----------------------------------------------------------

    def call_chain(self, repo: str, target: str, depth: int = 5) -> list[dict]:
        """查询某类/方法的调用链：谁调用了它（callers）与它调用了谁（callees）"""
        relations = self.storage.load_relations(repo)
        graph: dict[str, list] = {}
        for fid, fname, rel, tid, tname in relations:
            if rel == "calls":
                graph.setdefault(fname, {"callers": [], "callees": []})
                graph[fname]["callees"].append(tname)
                graph.setdefault(tname, {"callers": [], "callees": []})
                graph[tname]["callers"].append(fname)

        if target not in graph:
            return []
        result = {
            "target": target,
            "callers": self._walk(graph, target, "callers", depth),
            "callees": self._walk(graph, target, "callees", depth),
        }
        return result

    def _walk(self, graph: dict, start: str, direction: str, depth: int) -> list[str]:
        """BFS 沿调用方向展开，返回调用链路径"""
        paths: list[str] = []
        visited = set()
        queue = deque([(start, [start])])
        while queue:
            node, path = queue.popleft()
            if len(path) > depth:
                continue
            if node in visited and node != start:
                continue
            visited.add(node)
            if len(path) > 1:
                paths.append(" -> ".join(path))
            for neighbor in graph.get(node, {}).get(direction, []):
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))
        return paths[:20]

    # ----------------------------------------------------------
    # 结构关系查询（extends/imports/contains）
    # ----------------------------------------------------------

    def relations_of(self, repo: str, name: str) -> list[dict]:
        """查询某符号的所有结构关系"""
        relations = self.storage.load_relations(repo)
        result = []
        for fid, fname, rel, tid, tname in relations:
            if fname == name or tname == name:
                result.append({
                    "from": fname,
                    "relation": rel,
                    "to": tname,
                })
        return result
