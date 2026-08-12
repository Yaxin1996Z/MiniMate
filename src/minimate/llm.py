"""
LLM 调用封装 —— 统一管理 API 调用
"""

import os
from openai import OpenAI

_client: OpenAI | None = None

# Token 用量统计（进程级累计）
_stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def get_stats() -> dict:
    """返回累计 Token 统计（调用次数 / prompt / completion）"""
    return dict(_stats)


def reset_stats() -> None:
    """重置 Token 统计"""
    _stats["calls"] = 0
    _stats["prompt_tokens"] = 0
    _stats["completion_tokens"] = 0


def _record_usage(resp) -> None:
    """记录一次响应的 token 用量"""
    global _stats
    _stats["calls"] += 1
    usage = getattr(resp, "usage", None)
    if usage is not None:
        _stats["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _stats["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        model = os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")

        if not api_key:
            raise RuntimeError(
                "请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量"
            )

        _client = OpenAI(api_key=api_key, base_url=base_url)
        _client._model = model  # type: ignore

    return _client


def get_model() -> str:
    return os.getenv("OPENAI_MODEL_NAME", "deepseek-chat")


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
            max_tokens=4096,
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
            max_tokens=4096,
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
