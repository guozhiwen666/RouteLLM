"""模型调用客户端：按「档位」组织节点，负责多 Key 轮转、Fallback 链、
冷却摘除与 SSE 流式。传输/解析在 utils/http_transport.py，异常在
utils/llm_errors.py；本模块只关心「选哪个节点、要不要重试」。
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator
from typing import Any

from utils.endpoint import Endpoint  # noqa: F401 - 兼容性重导出
from utils.http_transport import (
    Headers,
    StreamTransport,
    Transport,
    ChatResponse,
    iter_sse_content,
    parse_chat_response,
    urllib_stream_transport,
    urllib_transport,
)
# 兼容性重导出：调用方仍可从 utils.llm_client 导入这些异常
from utils.llm_errors import (  # noqa: F401
    AllBackendsFailedError,
    AuthLLMError,
    LLMError,
    NoEndpointError,
    PermanentLLMError,
    TransientLLMError,
)

__all__ = [
    "AllBackendsFailedError",
    "AuthLLMError",
    "ChatClient",
    "Endpoint",
    "LLMError",
    "NoEndpointError",
    "PermanentLLMError",
    "TransientLLMError",
]


# Endpoint 定义在 utils/endpoint.py（重导出见上方 import）
class ChatClient:
    """模型调用客户端。"""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        backoff_base_s: float = 0.5,
        transport: Transport | None = None,
        stream_transport: StreamTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
        jitter: bool = True,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries 不能为负")
        if timeout <= 0:
            raise ValueError("timeout 必须为正数")

        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._transport = transport or urllib_transport
        self._stream_transport = stream_transport or urllib_stream_transport
        self._sleep = sleep
        self._now = now
        self._jitter = jitter
        self._endpoints: dict[str, list[Endpoint]] = {}
        self._round_robin: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # 节点注册
    # ------------------------------------------------------------------ #

    def add_endpoint(self, tier: str, endpoint: Endpoint) -> None:
        """为一个档位注册节点；同一档位注册多个即构成 Fallback 链。"""
        if not tier:
            raise ValueError("tier 不能为空")
        self._endpoints.setdefault(tier, []).append(endpoint)

    def endpoints_for(self, tier: str) -> list[Endpoint]:
        return list(self._endpoints.get(tier) or [])

    def health_snapshot(self) -> dict[str, list[dict[str, Any]]]:
        """各档位节点健康快照，供监控排障。"""
        now = self._now()
        return {
            tier: [endpoint.snapshot(now) for endpoint in endpoints]
            for tier, endpoints in self._endpoints.items()
        }

    # ------------------------------------------------------------------ #
    # 调用
    # ------------------------------------------------------------------ #

    def chat(
        self,
        tier: str,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """非流式调用：Key 轮转 → 节点轮转 → 指数退避。"""
        endpoints = self.endpoints_for(tier)
        if not endpoints:
            raise NoEndpointError(f"档位 {tier} 没有注册任何节点")

        deadline = timeout if timeout is not None else self.timeout
        last_error: LLMError | None = None
        start = self._round_robin.get(tier, 0)

        for attempt in range(self.max_retries + 1):
            # 每轮从不同的节点开始，避免所有请求都挤在同一个刚恢复的节点上
            for endpoint in _rotate(endpoints, start + attempt):
                if not endpoint.available(self._now()):
                    continue
                for _ in range(endpoint.key_count):
                    try:
                        status, body = self._transport(
                            endpoint.chat_url,
                            self._build_payload(
                                model, messages, temperature, max_tokens, response_format, stream=False
                            ),
                            self._build_headers(endpoint.next_key()),
                            deadline,
                        )
                    except TransientLLMError as exc:
                        # 传输层网络错误：摘除节点 + 退避，换下一个节点
                        last_error = exc
                        endpoint.mark_unhealthy(str(exc), self._now(), scale=attempt + 1)
                        self._backoff(attempt)
                        break
                    try:
                        response = parse_chat_response(status, body, endpoint=endpoint.name)
                    except AuthLLMError as exc:
                        last_error = exc
                        continue  # 换下一个 Key 再试
                    except TransientLLMError as exc:
                        last_error = exc
                        endpoint.mark_unhealthy(str(exc), self._now(), scale=attempt + 1)
                        self._backoff(attempt)
                        break  # 该节点本轮放弃，换下一个节点
                    # PermanentLLMError 不拦截：请求本身有问题，重试没意义
                    endpoint.mark_healthy()
                    self._round_robin[tier] = (self._round_robin.get(tier, 0) + 1) % len(endpoints)
                    return response

        raise AllBackendsFailedError(
            f"档位 {tier} 的所有节点均调用失败（共 {self.max_retries + 1} 轮）：{last_error}"
        )

    def chat_stream(
        self,
        tier: str,
        model: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: float | None = None,
    ) -> Iterator[str]:
        """流式调用：吐出去的内容收不回，故不重试，异常交给上层决策。"""
        endpoints = self.endpoints_for(tier)
        if not endpoints:
            raise NoEndpointError(f"档位 {tier} 没有注册任何节点")

        deadline = timeout if timeout is not None else self.timeout
        start = self._round_robin.get(tier, 0)
        last_error: LLMError | None = None
        for offset in range(len(endpoints)):
            endpoint = endpoints[(start + offset) % len(endpoints)]
            if not endpoint.available(self._now()):
                continue
            try:
                lines = self._stream_transport(
                    endpoint.chat_url,
                    self._build_payload(model, messages, temperature, max_tokens, None, stream=True),
                    self._build_headers(endpoint.next_key()),
                    deadline,
                )
            except TransientLLMError as exc:
                last_error = exc
                endpoint.mark_unhealthy(str(exc), self._now())
                continue
            # 包一层：正常消费完标记健康，中途网络错误则摘除节点
            return self._guarded_stream(endpoint, lines)
        raise AllBackendsFailedError(f"档位 {tier} 的流式调用全部失败：{last_error}")

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #

    def _guarded_stream(self, endpoint: Endpoint, lines: Iterator[str]) -> Iterator[str]:
        """包一层流式迭代：正常结束标记健康，中途异常摘除节点。"""
        try:
            for text in iter_sse_content(lines):
                yield text
        except TransientLLMError as exc:
            endpoint.mark_unhealthy(str(exc), self._now())
            raise
        else:
            endpoint.mark_healthy()

    @staticmethod
    def _build_payload(
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if response_format:
            payload["response_format"] = response_format
        if stream:
            # 让服务端把 usage 一并带回，否则流式请求的计量会缺失
            payload["stream_options"] = {"include_usage": True}
        return payload

    @staticmethod
    def _build_headers(api_key: str) -> Headers:
        headers: Headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _backoff(self, attempt: int) -> None:
        """指数退避 + 抖动：避免客户端同时重试打挂刚恢复的服务。"""
        delay = self.backoff_base_s * (2**attempt)
        if self._jitter:
            delay += random.uniform(0, self.backoff_base_s)
        self._sleep(delay)

def _rotate(items: list[Endpoint], offset: int) -> list[Endpoint]:
    """把列表循环左移，用于节点轮转。"""
    if not items:
        return []
    return items[offset % len(items) :] + items[: offset % len(items)]
