"""代码检索 —— 余弦相似度 + 调用链查询"""

from collections import deque

from .embeddings import cosine, embed


class CodeRetriever:
    def __init__(self, storage):
        self.storage = storage

    # ----------------------------------------------------------
    # 语义检索（余弦相似度，内存计算）
    # ----------------------------------------------------------

    def search(self, repo: str, query: str, top_k: int = 5) -> list[dict]:
        """按自然语言查询相关代码块（文件/类/方法）"""
        chunks = self.storage.load_chunks(repo)
        if not chunks:
            return []
        q_vec = embed(query)
        scored = []
        for c in chunks:
            if not c.get("vector"):
                continue
            sim = cosine(q_vec, c["vector"])
            if sim > 0.01:
                scored.append((sim, c))
        scored.sort(key=lambda x: -x[0])
        for _, c in scored[:top_k]:
            c["score"] = round(_, 4)
        return [c for _, c in scored[:top_k]]

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
