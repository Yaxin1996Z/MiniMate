"""LLM 客户端管理 —— 密钥 / BaseURL / 模型名统一从这里读取"""

import os

from openai import OpenAI

_client: OpenAI | None = None


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
