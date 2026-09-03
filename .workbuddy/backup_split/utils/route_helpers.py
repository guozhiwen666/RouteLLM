"""路由落点工具：把路由决策变成 state 上的确定性写入。

为什么放到 utils：`route_to_local` / `route_to_frontier` 只依赖
「配置 + state」，不依赖节点实例 —— 它们是纯函数，可以被路由节点、
测试、甚至未来的离线回放直接调用，放节点里只会让节点文件越来越胖。

语义约定：
  - route_to_local：挑能装下上下文的最小本地档位；装不下则内部升级到云端；
  - route_to_frontier：写云端档位；含 PII 的请求会附带合规告警；
  - record_route_latency：记录路由总耗时，超出 15ms 预算要告警。
"""

from __future__ import annotations

import logging
import time

from config.config import RoutingConfig
from flow.state import GraphState

logger = logging.getLogger(__name__)


def route_to_local(
    cfg: RoutingConfig,
    state: GraphState,
    *,
    stage: str,
    confidence: float,
    reason: str,
    started: float,
) -> GraphState:
    """路由到本地档位：挑能装下上下文的**最小**档位（最便宜）。

    本地档位都装不下时（硬塞会截断丢失信息，README 边界处理表），
    内部升级到云端并保留升级理由。
    """
    tier = cfg.pick_local_tier(int(state.get("context_tokens") or 0))
    if tier is None:
        return route_to_frontier(
            cfg,
            state,
            reason=f"上下文 {state.get('context_tokens')} token 超出所有本地档位，强制升级",
            hits=[f"context_tokens > {cfg.force_context_tokens}"],
            started=started,
            confidence=confidence,
        )

    state.update(
        {
            "route_stage": stage,
            "route_tier": tier.name,
            "route_reason": reason,
            "confidence": confidence,
            "difficulty": state.get("difficulty") or "simple",
            "upgrade_count": int(state.get("upgrade_count") or 0),
        }
    )
    record_route_latency(state, started, cfg.route_budget_ms)
    return state


def route_to_frontier(
    cfg: RoutingConfig,
    state: GraphState,
    *,
    reason: str,
    hits: list[str],
    started: float,
    confidence: float = 0.0,
) -> GraphState:
    """路由到云端强模型。PII 请求需附加合规告警（数据要出网了）。"""
    frontier = cfg.frontier_tier()
    state.update(
        {
            "route_stage": "force_upgrade" if hits else "self_eval",
            "route_tier": frontier.name,
            "route_reason": reason,
            "force_upgrade_hits": hits,
            "confidence": confidence,
            "difficulty": state.get("difficulty") or "hard",
        }
    )
    if state.get("has_pii"):
        state["warnings"] = list(state.get("warnings") or []) + [
            "请求含 PII 但因强制升级规则需出网，请确认合规策略允许"
        ]
    record_route_latency(state, started, cfg.route_budget_ms)
    return state


def record_route_latency(state: GraphState, started: float, budget_ms: float) -> None:
    """记录路由总耗时；超出预算要告警（路由的延迟必须低于收益）。"""
    latency = state.setdefault("latency_ms", {})
    total_ms = round((time.perf_counter() - started) * 1000, 3)
    latency["route_ms"] = total_ms
    if total_ms > budget_ms:
        message = f"路由耗时 {total_ms:.2f}ms 超出预算 {budget_ms}ms"
        state["warnings"] = list(state.get("warnings") or []) + [message]
        logger.warning(message)
