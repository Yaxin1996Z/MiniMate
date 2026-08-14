"""AST 代码索引 —— 多粒度分块（文件/类/方法）+ 关系图谱提取

关系类型：extends / implements / imports / calls / contains
"""

import ast
import os


GRANULARITY_FILE = "file"
GRANULARITY_CLASS = "class"
GRANULARITY_METHOD = "method"

# 接口/抽象基类命名特征（implements 判定）
_INTERFACE_HINTS = ("ABC", "Interface", "Protocol", "abstract")


class CodeChunk:
    """代码块：粒度 + 位置 + 代码 + 检索文本"""

    def __init__(
        self,
        chunk_id: str,
        file_path: str,
        granularity: str,
        name: str,
        code: str,
        start_line: int,
        end_line: int,
        doc: str = "",
    ):
        self.chunk_id = chunk_id
        self.file_path = file_path
        self.granularity = granularity
        self.name = name
        self.code = code
        self.start_line = start_line
        self.end_line = end_line
        self.doc = doc

    @property
    def search_text(self) -> str:
        """检索用文本：代码 + 名称 + docstring（供向量化）"""
        return f"{self.name}\n{self.doc}\n{self.code}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "granularity": self.granularity,
            "name": self.name,
            "code": self.code,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "doc": self.doc,
        }


class Relation:
    """代码关系：from → relation → to"""

    def __init__(self, from_id, from_name, relation, to_id, to_name):
        self.from_id = from_id
        self.from_name = from_name
        self.relation = relation
        self.to_id = to_id
        self.to_name = to_name

    def to_tuple(self):
        return (self.from_id, self.from_name, self.relation, self.to_id, self.to_name)


def _node_name(node: ast.AST) -> str:
    """从 AST 节点提取名称（Call 的 callee / Name / Attribute）"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _node_name(node.value) + "." + node.attr
    if isinstance(node, ast.Call):
        return _node_name(node.func)
    if isinstance(node, ast.Constant):
        return str(node.value)
    return ""


def _is_interface(base_name: str) -> bool:
    return any(h in base_name for h in _INTERFACE_HINTS)


def _extract_imports(tree: ast.Module, file_id: str, file_name: str, relations: list[Relation]):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                relations.append(Relation(file_id, file_name, "imports", None, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = f"{module}.{alias.name}" if module else alias.name
                relations.append(Relation(file_id, file_name, "imports", None, name))


def _extract_calls(node: ast.AST, from_id: str, from_name: str, relations: list[Relation]):
    for call in ast.walk(node):
        if isinstance(call, ast.Call):
            callee = _node_name(call.func)
            if callee:
                relations.append(Relation(from_id, from_name, "calls", None, callee))


def _chunk_method(
    node,
    file_path: str,
    parent_id: str,
    parent_name: str,
    relations: list[Relation],
    chunks: list[CodeChunk],
):
    name = node.name
    method_id = f"{file_path}#method:{parent_name}.{name}" if parent_id else f"{file_path}#method:{name}"
    doc = ast.get_docstring(node) or ""
    code = ast.get_source_segment(_source, node) or ""
    chunk = CodeChunk(
        chunk_id=method_id,
        file_path=file_path,
        granularity=GRANULARITY_METHOD,
        name=f"{parent_name}.{name}" if parent_name else name,
        code=code,
        start_line=node.lineno,
        end_line=getattr(node, "end_lineno", node.lineno),
        doc=doc,
    )
    chunks.append(chunk)
    if parent_id:
        relations.append(Relation(parent_id, parent_name, "contains", method_id, chunk.name))
    _extract_calls(node, method_id, chunk.name, relations)


def _chunk_class(
    node: ast.ClassDef,
    file_path: str,
    file_id: str,
    file_name: str,
    relations: list[Relation],
    chunks: list[CodeChunk],
):
    class_id = f"{file_path}#class:{node.name}"
    doc = ast.get_docstring(node) or ""
    code = ast.get_source_segment(_source, node) or ""
    chunk = CodeChunk(
        chunk_id=class_id,
        file_path=file_path,
        granularity=GRANULARITY_CLASS,
        name=node.name,
        code=code,
        start_line=node.lineno,
        end_line=getattr(node, "end_lineno", node.lineno),
        doc=doc,
    )
    chunks.append(chunk)
    relations.append(Relation(file_id, file_name, "contains", class_id, node.name))

    # 继承关系
    for base in node.bases:
        base_name = _node_name(base)
        if base_name:
            rel = "implements" if _is_interface(base_name) else "extends"
            relations.append(Relation(class_id, node.name, rel, None, base_name))

    # 类内方法 / 嵌套类
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _chunk_method(item, file_path, class_id, node.name, relations, chunks)
        elif isinstance(item, ast.ClassDef):
            _chunk_class(item, file_path, file_id, file_name, relations, chunks)


def index_file(file_path: str, repo_name: str = "") -> tuple[list[CodeChunk], list[Relation]]:
    """解析单个 .py 文件：多粒度分块 + 关系图谱"""
    global _source
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        _source = f.read()
    try:
        tree = ast.parse(_source)
    except SyntaxError:
        return [], []

    chunks: list[CodeChunk] = []
    relations: list[Relation] = []
    file_name = os.path.basename(file_path)
    file_id = f"{file_path}#file"
    doc = ast.get_docstring(tree) or ""
    chunks.append(CodeChunk(
        chunk_id=file_id,
        file_path=file_path,
        granularity=GRANULARITY_FILE,
        name=file_name,
        code=_source,
        start_line=1,
        end_line=max(1, _source.count("\n") + 1),
        doc=doc,
    ))

    _extract_imports(tree, file_id, file_name, relations)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            _chunk_class(node, file_path, file_id, file_name, relations, chunks)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _chunk_method(node, file_path, file_id, file_name, relations, chunks)

    return chunks, relations


def index_directory(root: str, repo_name: str = "", exts=(".py",)) -> tuple[list[CodeChunk], list[Relation]]:
    """递归索引目录下所有代码文件"""
    all_chunks: list[CodeChunk] = []
    all_relations: list[Relation] = []
    skip_dirs = {".git", ".venv", "node_modules", "__pycache__", ".cache", "rag_db", "output", "dist", "build"}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
        for fn in filenames:
            if not fn.endswith(exts):
                continue
            path = os.path.join(dirpath, fn)
            chunks, relations = index_file(path, repo_name)
            all_chunks.extend(chunks)
            all_relations.extend(relations)
    return all_chunks, all_relations
