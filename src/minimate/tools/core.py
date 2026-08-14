"""
工具系统 —— Agent 可调用的工具

支持两种工具：
  1. 函数工具：通过 @tool 装饰器注册
  2. 记忆工具：读写 知识库
"""

import re
import json
import os
import inspect
import time
from typing import Any, Callable, Literal

from ..logging import audit_tool_call


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
            return classify_error(str(result))
        except Exception as e:
            # 参数名不匹配（如 file_path vs path）时，尝试按 Schema 属性名模糊映射
            remapped = self._remap_kwargs(kwargs)
            if remapped is not None:
                try:
                    result = self.func(**remapped)
                    return classify_error(str(result))
                except Exception as e2:
                    return classify_error(f"[工具错误] {e2}")
            return classify_error(f"[工具错误] {e}")

    def _remap_kwargs(self, kwargs: dict) -> dict | None:
        """参数名容错：把传入 kwargs 的 key 映射到 Schema 声明的参数名

        匹配规则：精确 → 忽略下划线/大小写 → 包含关系（file_path → path）。
        必填参数缺失时返回 None（不强行调用）。
        """
        props = self.parameters.get("properties", {})
        if not props:
            return None

        normalized = {
            p.replace("_", "").lower(): p for p in props
        }
        remapped: dict = {}
        used: set[str] = set()

        for key, value in kwargs.items():
            if key in props:
                remapped[key] = value
                used.add(key)
                continue
            norm = key.replace("_", "").lower()
            match = normalized.get(norm)
            if match is None:
                # 包含关系匹配：file_path → path
                for pnorm, pname in normalized.items():
                    if norm in pnorm or pnorm in norm:
                        match = pname
                        break
            if match is None or match in used:
                return None
            remapped[match] = value
            used.add(match)

        required = self.parameters.get("required", [])
        if required and not all(r in remapped for r in required):
            return None
        return remapped

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
    """从函数签名自动推断 JSON Schema 参数定义

    支持：
      - 类型注解（str/int/float/bool）→ JSON 类型
      - typing.Literal[...] → enum 限制取值范围
      - 默认值 → required 排除 + default 记录
      - additionalProperties: false → 禁止模型传 Schema 外的脏字段
    """
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
        annotation = param.annotation
        enum_values = None

        # typing.Literal[...] → enum 限制取值
        if getattr(annotation, "__origin__", None) is Literal:
            enum_values = list(annotation.__args__)
            if enum_values:
                ptype = type_map.get(type(enum_values[0]), "string")
        elif annotation is not inspect.Parameter.empty:
            ptype = type_map.get(param.annotation, "string")

        prop: dict = {"type": ptype}
        if enum_values is not None:
            prop["enum"] = enum_values
        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(pname)
        props[pname] = prop

    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


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

    def execute(self, tool_name: str, **kwargs) -> str:
        """按命名参数执行（Function Calling 通道）"""
        start = time.time()
        tool = self._tools.get(tool_name)
        if not tool:
            available = ", ".join(self._tools) if self._tools else "无"
            result = classify_error(f"[工具错误] 未知工具 '{tool_name}'，可用工具：{available}")
        else:
            result = tool.run_kwargs(kwargs)
        audit_tool_call(
            tool_name,
            json.dumps(kwargs, ensure_ascii=False)[:200],
            result,
            time.time() - start,
        )
        return result

    def execute_text(self, tool_name: str, text: str = "") -> str:
        """按文本协议执行（fallback 通道）"""
        start = time.time()
        tool = self._tools.get(tool_name)
        if not tool:
            available = ", ".join(self._tools) if self._tools else "无"
            result = classify_error(f"[工具错误] 未知工具 '{tool_name}'，可用工具：{available}")
        else:
            result = tool.run_text(text)
        audit_tool_call(
            tool_name,
            text[:200],
            result,
            time.time() - start,
        )
        return result

    def execute_action(self, tool_name: str, args: str = "") -> str:
        """按工具名执行文本参数（旧接口，兼容）"""
        return self.execute_text(tool_name, args)

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


# ============================================================
# 错误分级标记
# ============================================================

_RETRYABLE_KEYWORDS = (
    "超时", "timeout", "网络", "连接", "暂时", "稍后",
    "busy", "rate", "限流", "搜索出错",
)
_NON_RETRYABLE_KEYWORDS = (
    "不存在", "不是目录", "未知工具", "参数", "格式",
    "需要", "无效", "拒绝", "请提供", "未找到",
    "不能为空", "错误：",
)
_ERROR_PREFIXES = (
    "错误", "[工具错误]", "[MCP 工具错误]", "[命令",
    "[搜索出错]", "[读文件错误]", "[写文件错误]",
    "[列目录错误]", "[搜索错误]",
)


def classify_error(text: str) -> str:
    """给工具错误附加可恢复性标记：[可重试] / [不可重试]

    - 已带标记（幂等）或不是错误文本 → 原样返回
    - 命中可重试关键词（超时/网络/临时） → [可重试]
    - 其他错误默认 [不可重试]（参数/确定性问题不重试）
    """
    text = text or ""
    if "[可重试]" in text or "[不可重试]" in text:
        return text
    if not text.startswith(_ERROR_PREFIXES):
        return text
    if any(k in text for k in _RETRYABLE_KEYWORDS):
        return f"[可重试] {text}"
    if any(k in text for k in _NON_RETRYABLE_KEYWORDS):
        return f"[不可重试] {text}"
    return f"[不可重试] {text}"
