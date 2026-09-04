"""结果处理：定稿 → 写缓存 → 计量 → 影子流量抽样。

链路位置：{cache_query | slm_inference | cloud_inference} → result_process → {END | comparative_scoring}

README 主流程图的 L 节点：返回给用户之后要做的三件事
（异步写语义缓存、写计量数据、5% 影子流量双跑）全部在这里完成。

注意写缓存的几道闸：
  失败的请求不写、涉实时数据不写、含 PII 不写、自检没通过不写。
  一条错误答案被缓存后会被反复返回给后续查询（README 风险清单第 2 条）。
"""

from __future__ import annotations

import logging

from config.config import RoutingConfig, routing_config
from flow.base import BaseNode
from utils.guardian import GuardianMonitor
from utils.metering import Meter, RequestRecord, compute_cost, merge_usage
from utils.semantic_cache import SemanticCache
from flow.state import GraphState

logger = logging.getLogger(__name__)


class ResultProcessNode(BaseNode):
    """结果定稿、缓存回写、计量与影子抽样。"""

    name = "result_process"

    def __init__(
        self,
        *,
        cache: SemanticCache | None = None,
        meter: Meter | None = None,
        guardian: GuardianMonitor | None = None,
        config: RoutingConfig | None = None,
    ) -> None:
        super().__init__()
        self.cache = cache
        self.meter = meter
        self.guardian = guardian
        self.cfg = config or routing_config

    def process(self, state: GraphState) -> GraphState:
        new_state = dict(state)
        new_state.setdefault("latency_ms", {})
        new_state.setdefault("warnings", [])

        # ---------------- 1. 定稿：缓存答案 > 云端答案 > 本地答案 ----------------
        final_output = (
            new_state.get("final_output")
            or new_state.get("cloud_response")
            or new_state.get("slm_response")
            or ""
        )
        new_state["final_output"] = final_output
        new_state["final_tier"] = new_state.get("final_tier") or new_state.get("route_tier") or "unknown"

        upgraded = bool(new_state.get("upgrade_count")) or new_state.get("route_stage") in {
            "self_check_upgrade",
            "cloud",
        }

        # ---------------- 2. 回写语义缓存 ----------------
        self._write_cache(new_state)

        # ---------------- 3. 计量 ----------------
        self._record(new_state, upgraded)

        # ---------------- 4. 影子流量抽样 ----------------
        # 只对本机 SLM 实际作答的请求抽样：本来就是强模型回答的，
        # 再双跑一次强模型没有对比意义，纯属浪费钱。
        local_names = {item.name for item in self.cfg.local_tiers}
        if (
            self.guardian is not None
            and not new_state.get("error")
            and final_output
            and new_state.get("final_tier") in local_names
            and self.guardian.should_shadow(new_state.get("request_id") or "")
        ):
            new_state["shadow"] = True

        return new_state

    # ------------------------------------------------------------------ #

    def _write_cache(self, state: GraphState) -> None:
        """把本次结果写入语义缓存（不满足条件则静默跳过）。"""
        if self.cache is None:
            return
        if state.get("cache_hit"):
            return  # 命中就不必回写
        if state.get("error"):
            return  # 失败结果不入库
        if state.get("no_cache"):
            return
        if state.get("has_pii") and self.cfg.exclude_pii:
            return

        answer = state.get("final_output") or ""
        if not answer.strip():
            return

        check = state.get("self_check") or {}
        if check and not check.get("passed", True):
            # 写缓存前做质量校验：不通过的结果绝不入库
            logger.info("输出自检未通过，跳过缓存写入: %s", check.get("reasons"))
            return

        query = state.get("normalized_query") or ""
        vector = state.get("embedding") or self.cache.embed(query)
        if not vector:
            return

        self.cache.store(
            query=query,
            answer=answer,
            vector=vector,
            entities=state.get("entities") or {},
            model_version=state.get("model_version") or "",
            prompt_version=state.get("prompt_version") or "",
            has_pii=bool(state.get("has_pii")),
            no_cache=bool(state.get("no_cache")),
            tier=state.get("final_tier") or "",
            usage=state.get("usage") or {},
        )

    def _record(self, state: GraphState, upgraded: bool) -> None:
        """计算成本并写入计量器。"""
        latency = dict(state.get("latency_ms") or {})
        # 端到端耗时 = 各阶段之和（缓存查询 / 路由 / 推理）
        latency["total_ms"] = round(
            sum(value for key, value in latency.items() if key != "total_ms"), 3
        )
        state["latency_ms"] = latency

        usage = merge_usage(state.get("usage"), None)
        state["usage"] = usage

        tier_name = state.get("final_tier") or "unknown"
        cache_hit = bool(state.get("cache_hit"))

        cost_cny = 0.0
        if not cache_hit:
            try:
                tier = self.cfg.tier(tier_name)
            except Exception:  # noqa: BLE001 - 档位缺失（如本地不可用）时按 0 成本记录
                tier = None
            if tier is not None:
                # 本地按 GPU 占用时长分摊折旧，云端按 token 类型计价
                stage_ms = (
                    latency.get("slm_ms", 0.0) if tier.is_local else latency.get("cloud_ms", 0.0)
                )
                cost_cny = compute_cost(tier, usage, stage_ms / 1000.0)
        # 缓存命中的请求成本近似为 0（embedding 开销忽略不计）
        state["cost_cny"] = round(cost_cny, 8)

        if self.meter is not None:
            self.meter.record(
                RequestRecord(
                    request_id=state.get("request_id") or "",
                    tier=tier_name,
                    cache_hit=cache_hit,
                    upgraded=upgraded,
                    usage=usage,
                    cost_cny=cost_cny,
                    latency_ms=latency["total_ms"],
                    error=state.get("error") or "",
                )
            )
            if self.guardian is not None:
                self.meter.observe_route_threshold(self.guardian.threshold)
