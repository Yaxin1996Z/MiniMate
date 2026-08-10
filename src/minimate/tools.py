"""
工具系统 —— Agent 可调用的工具

支持两种工具：
  1. 函数工具：通过 @tool 装饰器注册
  2. 记忆工具：读写 ResearchMemory
"""

import re
import json
import os
import inspect
from typing import Any, Callable


# ============================================================
# 工具定义
# ============================================================

class Tool:
    """工具定义：名称 + 描述 + JSON Schema 参数 + 执行函数

    parameters 未显式提供时，通过 inspect 从函数签名自动推断，
    供 Function Calling 通道使用。
    """

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        parameters: dict | None = None,
    ):
        self.name = name
        self.description = description
        self.func = func
        self.parameters = parameters or _infer_parameters(func)

    def run_kwargs(self, kwargs: dict) -> str:
        """按命名参数执行（Function Calling 通道）"""
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"[工具错误] {e}"

    def run_text(self, text: str) -> str:
        """按文本协议执行（fallback 通道）：单参数取整段，多参数按 | 拆分"""
        names = list(self.parameters["properties"].keys())
        if len(names) == 1:
            return self.run_kwargs({names[0]: _first_line(text)})
        parts = (text or "").split("|", len(names) - 1)
        parts = [p.strip().strip('"').strip("'") for p in parts]
        if len(parts) < len(names):
            return f"错误：需要 {len(names)} 个参数（{' | '.join(names)}）"
        return self.run_kwargs(dict(zip(names, parts)))

    @property
    def schema(self) -> dict:
        """Function Calling 用的工具 Schema（OpenAI 兼容格式）"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _infer_parameters(func: Callable) -> dict:
    """从函数签名自动推断 JSON Schema 参数定义"""
    sig = inspect.signature(func)
    props: dict[str, dict] = {}
    required: list[str] = []
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}

    for pname, param in sig.parameters.items():
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        ptype = "string"
        if param.annotation is not inspect.Parameter.empty:
            ptype = type_map.get(param.annotation, "string")
        prop: dict = {"type": ptype}
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(pname)
        props[pname] = prop

    return {"type": "object", "properties": props, "required": required}

    def run(self, *args, **kwargs) -> str:
        try:
            result = self.func(*args, **kwargs)
            return str(result)
        except Exception as e:
            return f"[工具错误] {e}"


def tool(name: str = "", description: str = ""):
    """工具装饰器"""
    def decorator(func):
        return Tool(
            name=name or func.__name__,
            description=description or func.__doc__ or "",
            func=func,
        )
    return decorator


# ============================================================
# 参数清洗
# ============================================================

def _first_line(text: str) -> str:
    """取第一行作为参数：模型可能在 Action Input 后追加散文（如"请稍候..."），
    单行参数（路径/命令/关键词）只取第一行，避免整段被当成参数"""
    line = (text or "").strip().strip('"').strip("'")
    if "\n" in line:
        line = line.split("\n", 1)[0].strip().strip('"').strip("'")
    return line


def _system_hint() -> str:
    """返回当前系统的命令适配提示，帮助模型避免写错命令"""
    if os.name == "nt":
        return "当前系统：Windows（cmd）。请使用 Windows 命令（dir/type/echo/where），不要使用 Linux 命令（ls/cat/pwd/grep）。"
    return "当前系统：Linux/macOS（bash）。请使用 bash 命令。"


# ============================================================
# 内置工具
# ============================================================

@tool(name="save_file", description="保存文本到文件，参数格式：文件名 | 内容")
def save_file(filename: str, content: str) -> str:
    """保存内容到 output 目录"""
    import os

    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return f"文件已保存：{filepath}"


@tool(name="web_search", description="搜索互联网，输入搜索关键词，返回搜索结果标题和摘要")
def web_search(query: str) -> str:
    """搜索互联网，返回最新信息"""
    query = _first_line(query)
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                results.append(f"- {title}\n  {body}\n  {href}")
        if not results:
            return "没有找到相关结果。"
        return "搜索结果：\n" + "\n\n".join(results)
    except Exception as e:
        return f"[搜索出错] {e}"


@tool(name="query_knowledge", description="从本地知识库中搜索相关内容。知识库已预加载了本地文档，输入问题返回最相关的文档片段，可作为调研参考")
def query_knowledge(question: str) -> str:
    """从已加载的本地知识库中检索相关内容"""
    question = _first_line(question)
    from .rag import get_knowledge_base
    kb = get_knowledge_base()
    return kb.query(question) or "知识库中没有相关内容。"


@tool(name="read_file", description="读取指定文件的内容，参数为文件路径（绝对路径或相对路径），返回文件正文")
def read_file(path: str) -> str:
    """读取指定文件的内容，长文件自动截断"""
    import os

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
    import os

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
    import os

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


@tool(name="run_shell", description="在终端执行 shell 命令（真实执行，超时 30 秒），参数为命令字符串，返回退出码与输出。用于运行脚本、查看系统信息等")
def run_shell(command: str) -> str:
    """执行 shell 命令，带超时保护与输出截断"""
    import subprocess

    command = _first_line(command)
    if not command:
        return "错误：请提供要执行的命令"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "[命令超时] 执行超过 30 秒已终止"
    except Exception as e:
        return f"[命令执行错误] {e}"

    output = (result.stdout or "").strip()
    if result.stderr and result.stderr.strip():
        output += ("\n" if output else "") + result.stderr.strip()
    if not output:
        output = "(无输出)"
    if len(output) > 3000:
        output = output[:3000] + f"\n...[输出过长，已截断 {len(output) - 3000} 字符]"
    text = f"退出码：{result.returncode}\n{output}"
    if result.returncode != 0:
        text += f"\n（提示：当前系统为 {'Windows cmd' if os.name == 'nt' else 'Linux/macOS bash'}，命令可能不适配，可换用系统原生命令）"
    return text


@tool(name="find_files", description="按文件名模式在目录树中查找文件，参数格式：目录 | 模式（如 src | *.py），模式支持 * 通配符，返回匹配文件列表")
def find_files(directory: str, pattern: str) -> str:
    """按文件名模式（glob）递归查找文件"""
    import glob
    import os

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
    import os

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


# ============================================================
# 工具执行器
# ============================================================

class ToolExecutor:
    """管理工具注册和调用"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def __bool__(self) -> bool:
        """空执行器视为 falsy，方便 Agent 判断是否有工具可用"""
        return bool(self._tools)

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]):
        for t in tools:
            self.register(t)

    def get_tools_prompt(self, protocol: bool = True) -> str:
        """生成供 LLM 使用的工具描述

        protocol=True：附带文本协议格式说明（fallback 通道用）
        protocol=False：仅工具列表（Function Calling 通道用，避免干扰模型输出格式）
        """
        if not self._tools:
            return ""
        lines = ["可用工具："]
        for name, t in self._tools.items():
            desc = t.description
            if name == "run_shell":
                desc += f"（{_system_hint()}）"
            lines.append(f"  - {t.name}: {desc}")
        lines.append("")
        lines.append("使用原则：当任务需要计算、检索外部信息或保存文件时，请调用对应工具，不要自行猜测。")
        if protocol:
            lines.append("")
            lines.append("调用方式（严格按行输出）：")
            lines.append("Thought: 当前推理（分析已有信息，决定下一步）")
            lines.append("Action: 工具名")
            lines.append("Action Input: 传给工具的参数")
            lines.append("")
            lines.append("格式纪律：Action Input 只填写参数本身，输出参数后立即结束，禁止追加任何说明文字（如'请稍候'等）。")
            lines.append("")
            lines.append("当信息已足够、可以给出最终回答时，输出：")
            lines.append("Final Answer: 最终回答内容")
        return "\n".join(lines)

    def execute_from_text(self, text: str) -> tuple[str, bool]:
        """解析文本中的 TOOL_CALL 并执行，返回 (执行结果, 是否调用了工具)"""
        match = re.search(r"TOOL_CALL:\s*(\w+)\s*\|\s*(.+)", text, re.MULTILINE)
        if not match:
            return "", False

        name = match.group(1)
        args = match.group(2).strip()

        tool = self._tools.get(name)
        if not tool:
            return f"错误：未知工具 '{name}'", True

        result = tool.run_text(args)
        return result, True

    def execute(self, name: str, **kwargs) -> str:
        """按命名参数执行（Function Calling 通道）"""
        tool = self._tools.get(name)
        if not tool:
            available = ", ".join(self._tools) if self._tools else "无"
            return f"[工具错误] 未知工具 '{name}'，可用工具：{available}"
        return tool.run_kwargs(kwargs)

    def execute_text(self, name: str, text: str = "") -> str:
        """按文本协议执行（fallback 通道）"""
        tool = self._tools.get(name)
        if not tool:
            available = ", ".join(self._tools) if self._tools else "无"
            return f"[工具错误] 未知工具 '{name}'，可用工具：{available}"
        return tool.run_text(text)

    def execute_action(self, name: str, args: str = "") -> str:
        """按工具名执行文本参数（旧接口，兼容）"""
        return self.execute_text(name, args)

    @property
    def tool_list(self) -> list[Tool]:
        return list(self._tools.values())

    @property
    def schemas(self) -> list[dict]:
        """导出 Function Calling 用的工具 Schema 列表"""
        return [t.schema for t in self._tools.values()]


# ============================================================
# ReAct 输出解析
# ============================================================

_THOUGHT_RE = re.compile(
    r"Thought:\s*(.*?)(?=\n\s*(?:Action|Final Answer):|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_ACTION_RE = re.compile(r"Action:\s*([\w\-_]+)", re.IGNORECASE)
_INPUT_RE = re.compile(
    r"Action Input:\s*(.*?)(?=\n\s*(?:Final Answer|Thought):|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_FINAL_RE = re.compile(
    r"Final Answer:\s*(.*?)(?=\n\s*(?:Action|Thought):|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def parse_react(text: str) -> dict:
    """解析模型输出的 ReAct 文本，返回 {thought, action, action_input, final_answer}

    优先识别标准标记格式；若模型输出 JSON（如 function calling 风格），
    也能兼容解析其中的 action / action_input / final_answer 字段。
    """
    text = (text or "").strip()
    result = {
        "thought": "",
        "action": "",
        "action_input": "",
        "final_answer": "",
    }
    if not text:
        return result

    # 兼容 JSON 输出
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            result["thought"] = str(data.get("thought", ""))
            result["action"] = str(data.get("action", "")).strip()
            result["action_input"] = str(data.get("action_input", data.get("input", "")))
            result["final_answer"] = str(
                data.get("final_answer", data.get("answer", data.get("output", "")))
            )
            if result["action"] or result["final_answer"]:
                return result
    except (json.JSONDecodeError, ValueError):
        pass

    m = _THOUGHT_RE.search(text)
    if m:
        result["thought"] = m.group(1).strip()
    m = _ACTION_RE.search(text)
    if m:
        result["action"] = m.group(1).strip()
    m = _INPUT_RE.search(text)
    if m:
        result["action_input"] = m.group(1).strip()
    m = _FINAL_RE.search(text)
    if m:
        result["final_answer"] = m.group(1).strip()
    return result


def truncate(text: str, max_chars: int = 3000) -> str:
    """截断工具返回结果，防止 Observation 撑爆上下文"""
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[结果过长，已截断 {len(text) - max_chars} 字符]"
