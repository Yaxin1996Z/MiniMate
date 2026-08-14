"""LLM 对话调用 —— react / plan / memory 压缩统一走这里"""

from .client import get_client, get_model
from .stats import _record_usage

DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.3


def call(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    """调用 LLM 并返回文本（单轮，兼容旧接口）"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, temperature=temperature)


def chat(messages: list[dict], temperature: float = 0.3) -> str:
    """调用 LLM 并返回文本（支持完整消息列表，供 ReAct 循环累积推理链）"""
    client = get_client()

    try:
        resp = client.chat.completions.create(
            model=get_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        _record_usage(resp)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[API 错误] {e}"


def chat_tools(
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.3,
) -> dict:
    """调用 LLM 并返回结构化结果（Function Calling 通道）

    返回：{"content": str, "tool_calls": [{"id", "name", "arguments"(JSON字符串)}]}
    无工具调用时 tool_calls 为空列表。
    """
    client = get_client()

    kwargs = {}
    if tools:
        kwargs["tools"] = tools

    try:
        resp = client.chat.completions.create(
            model=get_model(),
            messages=messages,
            temperature=temperature,
            max_tokens=DEFAULT_MAX_TOKENS,
            **kwargs,
        )
        _record_usage(resp)
    except Exception as e:
        return {"content": f"[API 错误] {e}", "tool_calls": []}

    msg = resp.choices[0].message
    tool_calls = []
    for tc in msg.tool_calls or []:
        tool_calls.append({
            "id": tc.id,
            "name": tc.function.name,
            "arguments": tc.function.arguments or "{}",
        })
    return {"content": msg.content or "", "tool_calls": tool_calls}
