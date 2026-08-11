"""文件操作工具 —— 对应 ToolRegistry.registerFileTools"""

import os

from .core import ToolExecutor, _first_line, tool


@tool(name="save_file", description="保存文本到文件，参数格式：文件名 | 内容")
def save_file(filename: str, content: str) -> str:
    """保存内容到 output 目录"""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"文件已保存：{filepath}"


@tool(name="read_file", description="读取指定文件的内容，参数为文件路径（绝对路径或相对路径），返回文件正文")
def read_file(path: str) -> str:
    """读取指定文件的内容，长文件自动截断"""
    path = _first_line(path)
    if not path:
        return "错误：请提供文件路径"
    if not os.path.exists(path):
        return f"错误：文件不存在 - {path}"
    if os.path.isdir(path):
        return f"错误：{path} 是目录，请用 list_files 查看目录内容"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"[读文件错误] {e}"
    if len(content) > 5000:
        return content[:5000] + f"\n...[内容过长，已截断 {len(content) - 5000} 字符]"
    return content


@tool(name="list_files", description="列出指定目录下的文件和子目录，参数为目录路径，返回文件名与大小")
def list_files(path: str = ".") -> str:
    """列出指定目录下的文件和子目录"""
    path = _first_line(path) or "."
    if not os.path.exists(path):
        return f"错误：目录不存在 - {path}"
    if not os.path.isdir(path):
        return f"错误：{path} 不是目录"
    try:
        entries = sorted(os.listdir(path))
    except Exception as e:
        return f"[列目录错误] {e}"

    lines = []
    for name in entries:
        full = os.path.join(path, name)
        if os.path.isdir(full):
            lines.append(f"[目录] {name}/")
        else:
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            lines.append(f"[文件] {name} ({size} 字节)")
    if not lines:
        return f"目录为空：{path}"
    return f"{path} 下共 {len(lines)} 项：\n" + "\n".join(lines)


@tool(name="write_file", description="写入内容到指定文件（覆盖模式），参数格式：文件路径 | 内容，自动创建父目录")
def write_file(path: str, content: str) -> str:
    """写入内容到指定文件，自动创建父目录"""
    if not path:
        return "错误：请提供文件路径"
    try:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        return f"[写文件错误] {e}"
    return f"文件已写入：{path}（{len(content)} 字符）"


@tool(name="find_files", description="按文件名模式在目录树中查找文件，参数格式：目录 | 模式（如 src | *.py），模式支持 * 通配符，返回匹配文件列表")
def find_files(directory: str, pattern: str) -> str:
    """按文件名模式（glob）递归查找文件"""
    import glob

    root = directory.strip().strip('"').strip("'") or "."
    if not pattern:
        return "错误：请提供文件模式"
    if not os.path.exists(root):
        return f"错误：目录不存在 - {root}"
    if not os.path.isdir(root):
        return f"错误：{root} 不是目录"

    try:
        matches = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    except Exception as e:
        return f"[搜索错误] {e}"

    files = [m for m in matches if os.path.isfile(m)]
    if not files:
        return f"未找到匹配「{pattern}」的文件"

    shown = files[:50]
    lines = [f"[文件] {os.path.relpath(f, root)}" for f in shown]
    if len(files) > 50:
        lines.append(f"...共 {len(files)} 个文件，仅显示前 50 个")
    return f"找到 {len(files)} 个匹配文件：\n" + "\n".join(lines)


@tool(name="grep_files", description="在目录树中的文本文件中搜索关键词，参数格式：目录 | 关键词（如 src | ReAct），返回匹配文件与行号，最多显示 30 条")
def grep_files(directory: str, keyword: str) -> str:
    """在文本文件内容中搜索关键词，带行号"""
    root = directory.strip().strip('"').strip("'") or "."
    if not keyword:
        return "错误：请提供搜索关键词"
    if not os.path.exists(root):
        return f"错误：目录不存在 - {root}"
    if not os.path.isdir(root):
        return f"错误：{root} 不是目录"

    TEXT_EXTS = (
        ".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json",
        ".js", ".ts", ".html", ".css", ".jsx", ".tsx",
    )
    SKIP_DIRS = {
        ".git", ".venv", "node_modules", "__pycache__",
        "rag_db", "output", ".idea", ".vscode",
    }
    MAX_MATCHES = 30
    MAX_FILE_SIZE = 200_000

    matches: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if not name.lower().endswith(TEXT_EXTS):
                continue
            path = os.path.join(dirpath, name)
            try:
                if os.path.getsize(path) > MAX_FILE_SIZE:
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, 1):
                        if keyword in line:
                            rel = os.path.relpath(path, root)
                            matches.append(f"{rel}:{lineno}: {line.strip()[:120]}")
                            if len(matches) >= MAX_MATCHES:
                                break
            except Exception:
                continue
            if len(matches) >= MAX_MATCHES:
                break
        if len(matches) >= MAX_MATCHES:
            break

    if not matches:
        return f"未找到包含「{keyword}」的内容"
    note = "\n...（仅显示前 30 条）" if len(matches) >= MAX_MATCHES else ""
    return f"找到 {len(matches)} 处匹配：\n" + "\n".join(matches) + note


def register_file_tools(executor: ToolExecutor) -> None:
    """注册文件操作工具（对应 ToolRegistry.registerFileTools）"""
    executor.register(read_file)
    executor.register(write_file)
    executor.register(list_files)
    executor.register(save_file)
    executor.register(find_files)
    executor.register(grep_files)
