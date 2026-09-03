"""计量：token、成本、延迟分位、路由分布。

README 强调的两条口径纪律在这里落地：
  1. **按 token 类型分别计价**：input / output / cached_input 单价差几倍，
     混在一起算，优化方向就会跑偏；
  2. **成本口径要能说清**：本模块同时给出「总量」与「单请求均值」，
     并且把缓存命中请求（成本近似为 0）单独标记，避免平均值失真。

延迟分位用有界水位采样实现（保留最近 N 条），够用且不需要引入额外的直方图库。
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

from config.config import TierConfig


def empty_usage() -> dict[str, int]:
    """空的 token 计量字典（三种类型缺一不可）。"""
    return {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}


def merge_usage(base: dict[str, int] | None, extra: dict[str, int] | None) -> dict[str, int]:
    """合并两段计量（例如本地推理失败后又升级到云端，两段都要算）。"""
    result = empty_usage()
    for source in (base, extra):
        if not source:
            continue
        for key in result:
            result[key] += int(source.get(key) or 0)
    return result


def compute_cost(tier: TierConfig, usage: dict[str, int], latency_s: float) -> float:
    """计算一次请求的成本（CNY）。

    - 云端档位：按 token 类型分别计价，cached_input 走折扣价；
    - 本地档位：**没有按量费用，但必须把 GPU 折旧算进去** ——
      GPU 是固定开支不是按量付费，不算折旧的话「本地≈免费」这个账是错的；
      这里按请求占用 GPU 的时长线性分摊（小时成本 × 占用小时数）。
    """
    if latency_s < 0:
        latency_s = 0.0
    if tier.is_local:
        return tier.gpu_hourly_cost_cny * (latency_s / 3600.0)

    usage = usage or empty_usage()
    per_million = 1_000_000.0
    return (
        usage.get("input_tokens", 0) / per_million * tier.price_input_per_mtok
        + usage.get("output_tokens", 0) / per_million * tier.price_output_per_mtok
        + usage.get("cached_input_tokens", 0) / per_million * tier.price_cached_input_per_mtok
    )


@dataclass
class RequestRecord:
    """一条请求的计量快照。"""

    request_id: str
    tier: str
    cache_hit: bool
    upgraded: bool
    usage: dict[str, int]
    cost_cny: float
    latency_ms: float
    error: str = ""


class Meter:
    """进程内计量器：累计计数 + 有界延迟采样 + Prometheus 文本导出。"""

    def __init__(self, max_samples: int = 5000) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples 必须为正数")
        self._lock = threading.RLock()
        self._max_samples = max_samples
        self.records: deque[RequestRecord] = deque(maxlen=max_samples)
        # (tier, cache_hit) → 次数
        self.requests: dict[tuple[str, bool], int] = {}
        # tier → 成本 / token / 延迟样本
        self.cost_by_tier: dict[str, float] = {}
        self.tokens_by_tier: dict[str, dict[str, int]] = {}
        self.latency_samples: dict[str, deque[float]] = {}
        self.quality_gap: float | None = None
        self.route_threshold: float | None = None

    # ------------------------------------------------------------------ #

    def record(self, record: RequestRecord) -> None:
        """记录一条请求。"""
        with self._lock:
            self.records.append(record)
            key = (record.tier, record.cache_hit)
            self.requests[key] = self.requests.get(key, 0) + 1
            self.cost_by_tier[record.tier] = self.cost_by_tier.get(record.tier, 0.0) + record.cost_cny
            tokens = self.tokens_by_tier.setdefault(record.tier, empty_usage())
            for name in tokens:
                tokens[name] += int(record.usage.get(name) or 0)
            samples = self.latency_samples.setdefault(record.tier, deque(maxlen=self._max_samples))
            samples.append(record.latency_ms)

    def observe_quality_gap(self, gap: float) -> None:
        """记录最近一次守门测得的质量差（供监控面板展示趋势）。"""
        with self._lock:
            self.quality_gap = gap

    def observe_route_threshold(self, threshold: float) -> None:
        with self._lock:
            self.route_threshold = threshold

    # ------------------------------------------------------------------ #

    def quantile(self, tier: str, q: float = 0.99) -> float:
        """指定档位的延迟分位数（q ∈ (0, 1]）。样本不足时返回 0。"""
        if not 0.0 < q <= 1.0:
            raise ValueError("分位数 q 必须落在 (0, 1]")
        with self._lock:
            samples = sorted(self.latency_samples.get(tier) or [])
        if not samples:
            return 0.0
        index = min(len(samples) - 1, max(0, int(round(q * (len(samples) - 1)))))
        return samples[index]

    def summary(self) -> dict[str, Any]:
        """汇总视图：总量、均值、路由分布、分位延迟。"""
        with self._lock:
            total = sum(self.requests.values())
            total_cost = sum(self.cost_by_tier.values())
            by_tier = {
                tier: {
                    "requests": sum(count for (name, _), count in self.requests.items() if name == tier),
                    "cost_cny": round(cost, 6),
                    "tokens": dict(tokens),
                    "p50_ms": round(self.quantile(tier, 0.50), 2),
                    "p95_ms": round(self.quantile(tier, 0.95), 2),
                    "p99_ms": round(self.quantile(tier, 0.99), 2),
                }
                for tier, cost in self.cost_by_tier.items()
                for tokens in [self.tokens_by_tier.get(tier, empty_usage())]
            }
            cache_hits = sum(count for (_, hit), count in self.requests.items() if hit)
            return {
                "requests_total": total,
                "cost_cny_total": round(total_cost, 6),
                "cost_cny_per_request": round(total_cost / total, 6) if total else 0.0,
                "cache_hit_rate": round(cache_hits / total, 4) if total else 0.0,
                "by_tier": by_tier,
                "quality_gap": self.quality_gap,
                "route_threshold": self.route_threshold,
            }

    # ------------------------------------------------------------------ #

    def render_prometheus(self) -> str:
        """导出 Prometheus 文本格式（可直接被 /metrics 接口返回）。

        指标命名遵循 Prometheus 约定：counter 用 _total 后缀，label 用下划线。
        """
        with self._lock:
            lines: list[str] = []
            lines.append("# HELP routellm_requests_total 按档位与缓存命中统计的请求数")
            lines.append("# TYPE routellm_requests_total counter")
            for (tier, cache_hit), count in sorted(self.requests.items()):
                lines.append(
                    f'routellm_requests_total{{tier="{_escape(tier)}",cache_hit="{str(cache_hit).lower()}"}} {count}'
                )

            lines.append("# HELP routellm_cost_cny_total 按档位累计的推理成本（CNY）")
            lines.append("# TYPE routellm_cost_cny_total counter")
            for tier, cost in sorted(self.cost_by_tier.items()):
                lines.append(f'routellm_cost_cny_total{{tier="{_escape(tier)}"}} {cost:.6f}')

            lines.append("# HELP routellm_tokens_total 按档位与类型累计的 token 数")
            lines.append("# TYPE routellm_tokens_total counter")
            for tier, tokens in sorted(self.tokens_by_tier.items()):
                for name, value in sorted(tokens.items()):
                    lines.append(
                        f'routellm_tokens_total{{tier="{_escape(tier)}",type="{name}"}} {value}'
                    )

            lines.append("# HELP routellm_latency_ms 端到端延迟分位数（毫秒）")
            lines.append("# TYPE routellm_latency_ms gauge")
            for tier in sorted(self.latency_samples):
                for q in (0.50, 0.95, 0.99):
                    lines.append(
                        f'routellm_latency_ms{{tier="{_escape(tier)}",quantile="{q}"}} {self.quantile(tier, q):.2f}'
                    )

            lines.append("# HELP routellm_quality_gap 影子流量测得的质量差（SLM 相对强模型）")
            lines.append("# TYPE routellm_quality_gap gauge")
            lines.append(f"routellm_quality_gap {self.quality_gap if self.quality_gap is not None else 0.0}")

            lines.append("# HELP routellm_route_threshold 当前生效的路由置信度阈值")
            lines.append("# TYPE routellm_route_threshold gauge")
            lines.append(
                f"routellm_route_threshold {self.route_threshold if self.route_threshold is not None else 0.0}"
            )
            return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    """Prometheus label 值转义。"""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def quantile(values: Iterable[float], q: float = 0.99) -> float:
    """独立的分位数工具（便于对任意样本列表求分位）。"""
    samples = sorted(values)
    if not samples:
        return 0.0
    if not 0.0 < q <= 1.0:
        raise ValueError("分位数 q 必须落在 (0, 1]")
    index = min(len(samples) - 1, max(0, int(round(q * (len(samples) - 1)))))
    return samples[index]
