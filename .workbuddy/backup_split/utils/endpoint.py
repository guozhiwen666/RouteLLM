"""模型调用节点（Endpoint）：一个可调用的 vLLM 实例或云端供应商地址。

把节点从客户端里拆出来单独放，是因为「节点健康状态」是独立的运维概念：
任何上层（客户端、监控、运维脚本）都要能查看/标记节点健康，
不需要被迫认识整个 ChatClient。

冷却机制实现 README 的「摘除节点，恢复后重新加入」：
失败后进入冷却期不再被选中，冷却结束后的第一次调用相当于一次探测。
"""

from __future__ import annotations

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
