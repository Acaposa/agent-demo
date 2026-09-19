# ============================================================
# agent/llm_client.py —— LLM 客户端（延迟初始化）
# ------------------------------------------------------------
# 不在 import 时读取 API Key，避免测试/CI 环境直接崩溃。
# 第一次调用 get_client() 时才真正创建客户端。
# ============================================================

import os
from openai import OpenAI

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """获取（或首次创建）OpenAI 兼容客户端。"""
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "未找到 DEEPSEEK_API_KEY，请检查项目根目录的 .env 文件。"
        )

    _client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return _client