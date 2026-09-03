"""HTTP 传输与响应解析：把「发请求」和「解析 OpenAI 兼容响应」封在一起。

传输层抽象成可注入的函数（`Transport` / `StreamTransport`）：
  - 生产用 `urllib_transport` / `urllib_stream_transport`（标准库，零依赖）；
  - 测试注入假实现，不碰网络；
  - 接入内部网关 SDK 时只需替换这两个函数，上层客户端不用改。

默认实现基于标准库 urllib，没有引入 requests/httpx。
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from utils.llm_errors import (
    AuthLLMError,
    PermanentLLMError,
    TransientLLMError,
)

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
