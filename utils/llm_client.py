from __future__ import annotations

class LLMError(Exception):
    """所有模型调用异常的基类。"""


class TransientLLMError(LLMError):
    """可恢复错误：限流 / 5xx / 网络超时 —— 换 Key、换节点、退避重试。"""


class AuthLLMError(LLMError):
    """鉴权失败（401/403）—— 换下一个 Key 试，重试同一个 Key 没意义。"""


class PermanentLLMError(LLMError):
    """不可恢复错误（400/404/422，请求本身有问题）—— 重试多少次都是同一个结果。"""


class NoEndpointError(LLMError):
    """该档位没有可用节点（配置漏了或全部在冷却中）。"""


class AllBackendsFailedError(LLMError):
    """Fallback 链上所有节点都试过了，全部失败。"""


import json
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any


Headers = dict[str, str]
HttpResponse = tuple[int, str]
Transport = Callable[[str, dict[str, Any], Headers, float], HttpResponse]
StreamTransport = Callable[[str, dict[str, Any], Headers, float], Iterator[str]]


@dataclass
class ChatResponse:
    """一次模型调用的结果。"""

    content: str
    model: str
    endpoint: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def usage(self) -> dict[str, int]:
        """注意 cached_input_tokens 单独统计 —— README 明确要求，
        它的单价和普通 input 差一个数量级，混在一起成本报表就是错的。"""
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
        }


def urllib_transport(url: str, payload: dict[str, Any], headers: Headers, timeout: float) -> HttpResponse:
    """默认非流式传输：POST JSON，返回 (状态码, 响应体)。

    注意 4xx/5xx 不在这里抛异常，而是把状态码交给 `parse_chat_response`
    去归类 —— 传输层只管把字节拿回来，语义判断放在解析层。
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        # HTTPError 也是类文件对象，body 里通常有供应商给出的错误原因
        return exc.code, _safe_read(exc)
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        raise TransientLLMError(f"网络请求失败: {exc}") from exc


def urllib_stream_transport(
    url: str, payload: dict[str, Any], headers: Headers, timeout: float
) -> Iterator[str]:
    """默认流式传输：逐行产出 SSE 原始文本行（含 'data: ' 前缀）。

    调用方必须消费完（或显式 close），否则连接会一直挂着。
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        kind, message = classify_http_error(exc.code, _safe_read(exc))
        raise (TransientLLMError(message) if kind == "transient" else PermanentLLMError(message)) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as exc:
        raise TransientLLMError(f"网络请求失败: {exc}") from exc

    try:
        for raw_line in response:
            yield raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
    finally:
        response.close()


def classify_http_error(status: int, body: str) -> tuple[str, str]:
    """把 HTTP 状态码归类为 auth / transient / permanent，并给出可读信息。"""
    snippet = (body or "").strip()[:300]
    if status in (401, 403):
        return "auth", f"鉴权失败 ({status}): {snippet}"
    if status == 429:
        return "transient", f"触发限流 (429): {snippet}"
    if 500 <= status < 600:
        return "transient", f"服务端错误 ({status}): {snippet}"
    return "permanent", f"请求被拒绝 ({status}): {snippet}"


def parse_chat_response(status: int, body: str, *, endpoint: str = "") -> ChatResponse:
    """解析 OpenAI 兼容响应；异常状态翻译成对应的异常类型。"""
    if status < 200 or status >= 300:
        kind, message = classify_http_error(status, body)
        if kind == "auth":
            raise AuthLLMError(message)
        if kind == "transient":
            raise TransientLLMError(message)
        raise PermanentLLMError(message)

    payload = _load_json(body)
    choices = payload.get("choices") or []
    if not choices:
        raise TransientLLMError("响应缺少 choices 字段")

    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    content = message.get("content")
    if content is None:
        # 少数非 OpenAI 兼容实现会把内容放在 text 字段
        content = first.get("text") or ""
    if not isinstance(content, str):
        content = "" if content is None else str(content)

    usage = payload.get("usage") or {}
    # 各家「缓存命中 token」的字段名不统一，这里做兼容
    details = usage.get("prompt_tokens_details") or {}
    return ChatResponse(
        content=content,
        model=str(payload.get("model") or ""),
        endpoint=endpoint,
        input_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        cached_input_tokens=int(
            usage.get("cached_tokens")
            or details.get("cached_tokens")
            or usage.get("cached_input_tokens")
            or 0
        ),
        finish_reason=str(first.get("finish_reason") or ""),
        raw=payload,
    )


def iter_sse_content(lines: Iterator[str]) -> Iterator[str]:
    """从 SSE 原始行里提取增量文本；`[DONE]` 表示流结束。"""
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except (TypeError, ValueError):
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        text = (choices[0].get("delta") or {}).get("content")
        if text:
            yield text


def _safe_read(exc: urllib.error.HTTPError) -> str:
    """读取错误响应体；读失败也不能把原始异常吞掉。"""
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001 - 读错误体失败不该掩盖原始异常
        return ""


def _load_json(body: str) -> dict[str, Any]:
    """解析响应 JSON，失败按可恢复错误处理（可能是被网关截断了）。"""
    try:
        payload = json.loads(body)
    except (TypeError, ValueError) as exc:
        raise TransientLLMError(f"响应不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TransientLLMError("响应 JSON 结构不是对象")
    return payload


from dataclasses import dataclass, field
from typing import Any


@dataclass
class Endpoint:
    """一个可调用的模型节点（vLLM 实例或云端供应商地址）。"""

    name: str
    base_url: str
    api_keys: list[str] = field(default_factory=list)
    api_key: str = ""
    unhealthy_cooldown_s: float = 30.0

    # ---- 运行期状态 ----
    cooldown_until: float = 0.0
    last_error: str = ""
    failure_count: int = 0
    success_count: int = 0
    _key_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Endpoint 缺少 name")
        if not self.base_url:
            raise ValueError(f"Endpoint {self.name} 缺少 base_url")
        self.base_url = self.base_url.rstrip("/")
        keys = list(self.api_keys)
        if self.api_key and self.api_key not in keys:
            keys.append(self.api_key)
        self.api_keys = keys

    @property
    def chat_url(self) -> str:
        """OpenAI 兼容的对话补全地址。"""
        return f"{self.base_url}/v1/chat/completions"

    @property
    def key_count(self) -> int:
        return max(1, len(self.api_keys))

    def next_key(self) -> str:
        """轮询下一个 API Key（无 Key 返回空串，本地 vLLM 通常不需要鉴权）。"""
        if not self.api_keys:
            return ""
        key = self.api_keys[self._key_index % len(self.api_keys)]
        self._key_index += 1
        return key

    def available(self, now: float) -> bool:
        return now >= self.cooldown_until

    def mark_healthy(self) -> None:
        """调用成功：清除冷却与错误记录。"""
        self.last_error = ""
        self.cooldown_until = 0.0
        self.success_count += 1

    def mark_unhealthy(self, reason: str, now: float, scale: int = 1) -> None:
        """标记失败并进入冷却；连续失败会拉长冷却（指数退避的节点版）。"""
        self.failure_count += 1
        self.last_error = reason
        self.cooldown_until = now + min(self.unhealthy_cooldown_s * scale, 300.0)

    def snapshot(self, now: float) -> dict[str, Any]:
        """健康状态快照，供监控与排障。"""
        return {
            "name": self.name,
            "base_url": self.base_url,
            "available": self.available(now),
            "cooldown_remaining_s": round(max(0.0, self.cooldown_until - now), 2),
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_error": self.last_error,
        }


import random
import time
from collections.abc import Callable, Iterator
from typing import Any

# 兼容性重导出：调用方仍可从 utils.llm_client 导入这些异常

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
