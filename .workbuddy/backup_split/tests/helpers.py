"""端到端冒烟测试：用假的传输层把整条链路跑起来（不需要真实 GPU / API Key）。

运行方式（标准库 unittest，零额外依赖）：

    python tests/test_smoke.py
    或 python -m unittest discover -s tests

覆盖点基本对应 README 的边界处理表与失败案例：
  强制升级规则、缓存实体一致性、PII 不出网、输出自检升级、
  本地实例不可用自动切云端、守门自动升高阈值、图引擎的结构校验。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

# 让测试可以直接以脚本方式运行（不依赖安装）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import ConfigError, RoutingConfig, TierConfig, parse_yaml_subset  # noqa: E402
from flow.graph import END, GraphError, StateGraph, Workflow  # noqa: E402
from utils.guardian import (  # noqa: E402
    ACTION_FORCE_FRONTIER,
    ACTION_RAISE_THRESHOLD,
    GuardianMonitor,
)
from utils.heuristics import detect_code, detect_math, detect_realtime_data  # noqa: E402
from utils.llm_client import (  # noqa: E402
    AllBackendsFailedError,
    ChatClient,
    Endpoint,
    NoEndpointError,
    PermanentLLMError,
    TransientLLMError,
)
from utils.embedding import hash_embedding  # noqa: E402
from utils.metering import Meter, RequestRecord, compute_cost  # noqa: E402
from utils.selfcheck import check_output  # noqa: E402
from utils.semantic_cache import SemanticCache  # noqa: E402
from utils.text_utils import (  # noqa: E402
    detect_pii,
    estimate_tokens,
    extract_entities,
    extract_json_object,
)

SELF_EVAL_MARK = "路由难度评估器"
JUDGE_MARK = "结果评审员"


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


class MockLLM:
    """假的模型后端：按 payload 内容返回不同响应。"""

    def __init__(
        self,
        *,
        slm_reply: str = "这是本地小模型的回答。",
        cloud_reply: str = "这是云端强模型的回答。",
        self_eval: dict[str, Any] | None = None,
        judge: dict[str, Any] | None = None,
        failure_mode: str | None = None,
    ) -> None:
        self.slm_reply = slm_reply
        self.cloud_reply = cloud_reply
        self.self_eval = self_eval or {"can_answer": True, "confidence": 0.92, "difficulty": "simple"}
        self.judge = judge or {"a_score": 8, "b_score": 8}
        # None | "slm_down" | "cloud_500"；另外可单独让自评失败
        self.failure_mode = failure_mode
        self.self_eval_error: Exception | None = None
        self.calls: list[dict[str, Any]] = []

    @property
    def transport(self):
        return lambda url, payload, headers, timeout: self(url, payload, headers, timeout)

    def __call__(self, url: str, payload: dict[str, Any], headers: dict, timeout: float):
        self.calls.append({"url": url, "payload": payload})
        model = payload.get("model", "")
        first = payload["messages"][0] if payload.get("messages") else {}

        # 自评请求
        if first.get("role") == "system" and SELF_EVAL_MARK in (first.get("content") or ""):
            if self.self_eval_error is not None:
                raise self.self_eval_error
            return 200, self._body(model, json.dumps(self.self_eval, ensure_ascii=False))
        # judge 请求
        if first.get("role") == "system" and JUDGE_MARK in (first.get("content") or ""):
            return 200, self._body(model, json.dumps(self.judge, ensure_ascii=False))

        # 正常推理：本地 vs 云端
        if model.endswith("frontier") or "frontier" in model:
            if self.failure_mode == "cloud_500":
                return 500, '{"error":"upstream boom"}'
            return 200, self._body(model, self.cloud_reply)
        if self.failure_mode == "slm_down":
            raise TransientLLMError("本地 vLLM 实例连接失败")
        return 200, self._body(model, self.slm_reply)

    @staticmethod
    def _body(model: str, content: str, in_tokens: int = 16, out_tokens: int = 32) -> str:
        return json.dumps(
            {
                "model": model,
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": in_tokens,
                    "completion_tokens": out_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
            ensure_ascii=False,
        )


def make_config(**overrides: Any) -> RoutingConfig:
    """测试用配置：三档结构与 README 一致，价格与超时按需覆盖。"""
    tiers = [
        TierConfig("slm_tiny", "mock-tiny", "http://mock-tiny", 8192, local=True, gpu_hourly_cost_cny=3.0),
        TierConfig("slm_mid", "mock-mid", "http://mock-mid", 32768, local=True, gpu_hourly_cost_cny=6.0),
        TierConfig(
            "frontier",
            "mock-frontier",
            "http://mock-frontier",
            200000,
            local=False,
            price_input_per_mtok=70.0,
            price_output_per_mtok=350.0,
            price_cached_input_per_mtok=7.0,
        ),
    ]
    defaults: dict[str, Any] = {
        "tiers": tiers,
        "classifier_mode": "tiny_model_self_eval",
        "confidence_threshold": 0.75,
        "semantic_threshold": 0.95,
        "shadow_ratio": 0.0,  # 测试默认关闭影子流量，需要时单独打开
        "window_min_samples": 2,
        "consecutive_windows": 3,
        "threshold_cooldown_seconds": 0,
        "max_upgrade_attempts": 1,
        "self_eval_timeout_s": 1.0,
        "slm_timeout_s": 5.0,
        "cloud_timeout_s": 5.0,
        "judge_timeout_s": 5.0,
        "max_retries": 1,
        "backoff_base_s": 0.001,
        "cache_ttl_hours": 72,
    }
    defaults.update(overrides)
    return RoutingConfig(**defaults).validate()


def make_client(llm: MockLLM, cfg: RoutingConfig) -> ChatClient:
    """把假后端注册成三个档位的节点。"""
    client = ChatClient(
        transport=llm.transport,
        max_retries=cfg.max_retries,
        backoff_base_s=0.001,
        sleep=lambda _seconds: None,  # 测试里不要真的睡
        jitter=False,
    )
    for tier in cfg.tiers:
        client.add_endpoint(tier.name, Endpoint(name=tier.name, base_url=tier.endpoint))
    return client


def make_workflow(llm: MockLLM, cfg: RoutingConfig | None = None, **workflow_kwargs: Any) -> Workflow:
    cfg = cfg or make_config()
    return Workflow(client=make_client(llm, cfg), config=cfg, **workflow_kwargs)


def user_request(content: str, **extra: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": content}], **extra}
