#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立 LLM 客户端（pythonTest 自包含，不依赖 TestBrain）
=====================================================
用 OpenAI 兼容的 chat/completions 协议直接调用大模型（DeepSeek / Qwen 等）。

设计参考：TestBrain 的 apps/llm 工厂（架构思想：provider 统一管理、环境变量取 key），
但实现完全独立 —— 只需 requests，无 Django / LangChain 依赖。

【Provider 配置】内置默认 base_url / 模型名 / 环境变量 key：

| provider | base_url                     | 默认模型        | 环境变量          |
| ---      | ---                          | ---             | ---               |
| deepseek | https://api.deepseek.com     | deepseek-chat   | DEEPSEEK_API_KEY  |
| qwen     | https://dashscope.aliyun.com | qwen-max        | QWEN_API_KEY      |
| openai   | https://api.openai.com       | gpt-4o-mini     | OPENAI_API_KEY    |

可通过参数覆盖（api_key / base_url / model），也可在实例化时传入。

【用法】
    from pyst.llm_client import chat_completion
    text = chat_completion("你好", provider="deepseek", temperature=0.7)
"""

import os
import re
import time as _time
from pathlib import Path
from typing import Any

import requests

# Provider 默认配置：base_url / 默认模型 / 环境变量名
_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyun.com",
        "model": "qwen-max",
        "env_key": "QWEN_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
}


def _load_key_from_config() -> dict[str, str]:
    """从 pyst/.config 文件读取 API key（键值对，如 `DeepSeek_KEY = "sk-..."`）。

    作为环境变量之后的兜底来源。.config 格式（大小写不敏感匹配 provider）：
        DEEPSEEK_API_KEY = "sk-..."
        QWEN_API_KEY = "sk-..."
    返回 {provider_key: value}，匹配不到返回空 dict。
    """
    # .config 在 pyst 根目录（llm/client.py 在 pyst/llm 下，需要上跳一级）
    cfg_path = Path(__file__).parent.parent / ".config"
    result: dict[str, str] = {}
    if not cfg_path.exists():
        return result
    try:
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip().strip("\"'")
            if not value:
                continue
            result[key.strip().upper()] = value
    except OSError:
        pass
    return result


# 模块级缓存 .config，避免每次构造都读文件
_CONFIG_KEYS = _load_key_from_config()


class LLMClient:
    """OpenAI 兼容的 chat/completions 客户端。

    Args:
        provider: deepseek | qwen | openai（不认识的走 openai 兼容）
        api_key:  API key，缺省从对应环境变量读取
        base_url: 覆盖默认 base_url
        model:    覆盖默认模型名
        timeout:  请求超时（秒）
    """

    def __init__(self, provider: str = "deepseek", api_key: str | None = None,
                 base_url: str | None = None, model: str | None = None,
                 timeout: int = 120, connect_timeout: float = 5.0,
                 max_retries: int = 5):
        self.provider = provider
        defaults = _PROVIDER_DEFAULTS.get(provider, _PROVIDER_DEFAULTS["openai"])
        self.base_url = (base_url or defaults["base_url"]).rstrip("/")
        self.model = model or defaults["model"]
        env_key = defaults.get("env_key", "OPENAI_API_KEY")
        # key 来源优先级：显式传入 > 环境变量 > pyst/.config 文件
        self.api_key = api_key or os.getenv(env_key) or self._config_key(env_key)
        if not self.api_key:
            raise ValueError(
                f"未找到 API Key：请设置环境变量 {env_key}、pyst/.config 文件，或传入 api_key"
            )
        self.timeout = timeout
        # 连接超时与读取超时分开：外部 LLM 走 CDN 多 IP 轮询，部分边缘 IP 不可达
        # （实测 api.deepseek.com 15 个 IP 中 6 个黑洞），若用单值超时，连到坏 IP 会
        # 一直干等整个 timeout。连接超时设短，坏 IP 快速放弃并转下一个 IP/重试。
        self.connect_timeout = connect_timeout
        self.max_retries = max(1, max_retries)
        # Session 复用连接：一旦某次调用落到"好 IP"，后续调用复用 TCP 连接，
        # 不再每次都重新赌 DNS 轮询（requests.post 模块级函数每次都新建连接）
        self._session = requests.Session()

    @staticmethod
    def _config_key(env_key: str) -> str | None:
        """从 .config 的键中按 provider 前缀匹配 key。

        .config 里可能是 DEEPSEEK_API_KEY 或 DeepSeek_KEY 等变体，统一按
        provider 前缀（如 DEEPSEEK）匹配，容忍键名差异。
        """
        if not _CONFIG_KEYS:
            return None
        # 1. 精确匹配
        if env_key in _CONFIG_KEYS:
            return _CONFIG_KEYS[env_key]
        # 2. 前缀匹配：DEEPSEEK_API_KEY -> 匹配 DEEPSEEK* 开头的键
        prefix = env_key.split("_")[0]           # "DEEPSEEK"
        for k, v in _CONFIG_KEYS.items():
            if k == prefix or k.startswith(prefix + "_"):
                return v
        return None

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.7,
             max_tokens: int | None = None, **kwargs: Any) -> str:
        """发起一次 chat/completions 请求，返回助手回复文本。

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            **kwargs: 透传给请求体的其他参数（如 top_p）
        """
        url = f"{self.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        # 输出 token 上限给足（DeepSeek 默认 4k，长用例+超长边界值会被截断在字符串中间）
        payload["max_tokens"] = max_tokens or 8192
        payload.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # 分连接/读取超时 + 失败重试：CDN 多 IP 中部分边缘 IP 不可达时，
        # 短连接超时能快速放弃坏 IP（socket.create_connection 会自动尝试下一个 IP），
        # 重试则进一步规避偶发的连接抖动。
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(
                    url, json=payload, headers=headers,
                    timeout=(self.connect_timeout, self.timeout))
                resp.raise_for_status()
                data = resp.json()
                try:
                    return data["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    raise RuntimeError(f"LLM 响应格式异常: {data}")
            except (requests.exceptions.ConnectTimeout,
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError) as e:
                last_err = e
                if attempt < self.max_retries:
                    _time.sleep(min(2 ** (attempt - 1), 5))   # 1s, 2s, 4s... 最多 5s
                    continue
                raise
        raise RuntimeError(f"LLM 调用失败（已重试 {self.max_retries} 次）: {last_err}")


def chat_completion(prompt: str, provider: str = "deepseek",
                    system: str | None = None,
                    api_key: str | None = None,
                    base_url: str | None = None,
                    model: str | None = None,
                    temperature: float = 0.7,
                    max_tokens: int | None = None,
                    **kwargs: Any) -> str:
    """便捷函数：把单个 prompt 变成一条 user 消息发给 LLM。

    Args:
        prompt: 用户输入（完整 prompt 文本）
        provider: deepseek | qwen | openai
        system: 可选的 system 消息（默认 None，即无系统消息）
        api_key / base_url / model / temperature / max_tokens: 透传给 LLMClient
        **kwargs: 透传给 chat() 的额外请求参数
    """
    client = LLMClient(provider=provider, api_key=api_key,
                       base_url=base_url, model=model)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return client.chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "你好，请自我介绍"
    print(chat_completion(q, provider="deepseek"))
