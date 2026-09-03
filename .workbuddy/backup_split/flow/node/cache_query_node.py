"""语义缓存查询 + 路由决策（整套系统的大脑）。

链路位置：query_process → cache_query → {result_process | slm_inference | cloud_inference}

流程：缓存查询 → 强制升级规则 → 启发式快筛 → 小模型自评 → 定档。
强制升级规则命中时自评是**无效的**，故先跑规则（命中即短路，省下 8ms）。
核心原则：**任何不确定都升级到强模型**（fail-safe）。
本节点独有状态见 `CacheQueryState`。
"""

from __future__ import annotations

import time

from config.config import RoutingConfig, routing_config
from flow.base import BaseNode
from flow.state import GraphState
from utils.guardian import GuardianMonitor
from utils.llm_client import ChatClient
from utils.semantic_cache import CacheLookup, SemanticCache
from utils.self_eval import SelfEvalResult, evaluate_difficulty
from utils.route_helpers import route_to_frontier, route_to_local
from utils.text_utils import estimate_tokens

class CacheQueryState(GraphState):
    """缓存查询 + 路由决策阶段的独有状态字段。"""

    # ---- 语义缓存 ----
    embedding: list[float]
    cache_hit: bool
    cache_similarity: float
    cache_entity_consistent: bool
    cache_reason: str  # 未命中原因
    cached_answer: str

    # ---- 路由决策 ----
    route_stage: str  # cache | force_upgrade | heuristic | self_eval | self_check_upgrade
    route_tier: str  # 档位名
    route_reason: str
    difficulty: str  # simple | moderate | hard
    confidence: float  # 自评置信度
    self_eval_raw: str
    force_upgrade_hits: list[str]
    upgrade_count: int  # 升级次数，防乒乓

class CacheQueryNode(BaseNode):
    """语义缓存查询与路由决策节点。"""

    name = "cache_query"
    state_schema = CacheQueryState

    def __init__(
        self,
        *,
        client: ChatClient | None = None,
        cache: SemanticCache | None = None,
        guardian: GuardianMonitor | None = None,
        config: RoutingConfig | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.cache = cache
        self.guardian = guardian
        self.cfg = config or routing_config

    # ------------------------------------------------------------------ #

    def process(self, state: GraphState) -> GraphState:
        new_state = dict(state)
        new_state.setdefault("latency_ms", {})
        new_state.setdefault("cache_hit", False)
        started = time.perf_counter()

        if new_state.get("error") or new_state.get("cache_hit"):
            return new_state

        query = new_state.get("normalized_query") or ""
        has_pii = bool(new_state.get("has_pii"))
        no_cache = bool(new_state.get("no_cache"))
        context_tokens = int(new_state.get("context_tokens") or 0)

        # 1) 语义缓存查询
        lookup = self._lookup_cache(new_state, query, has_pii, no_cache)
        new_state["latency_ms"]["cache_lookup_ms"] = _elapsed_ms(started)
        if lookup is not None:
            if lookup.hit:
                self._on_cache_hit(new_state, lookup, context_tokens)
                return new_state
            # 未命中也留痕，否则答不了「为什么没省到钱」
            new_state["cache_similarity"] = lookup.similarity
            new_state["cache_entity_consistent"] = lookup.entity_consistent
            new_state["cache_reason"] = lookup.reason

        # 2) 守门极端降级：一键全量切强模型
        if self.guardian is not None and self.guardian.force_all_frontier:
            return route_to_frontier(self.cfg, new_state, reason="守门触发全量强模型降级", hits=[], started=started)

        # 3) 强制升级规则（命中即短路）
        hits = self._force_upgrade_hits(new_state)
        if hits:
            return route_to_frontier(
                self.cfg,
                new_state,
                reason=f"命中强制升级规则: {', '.join(hits)}",
                hits=hits,
                started=started,
            )

        # 4) 启发式快筛：极高置信的简单请求
        is_simple, reason = self._heuristic_screen(new_state)
        if is_simple:
            return route_to_local(
                self.cfg, new_state, stage="heuristic", confidence=0.99,
                reason=f"启发式快筛判定为简单: {reason}", started=started,
            )

        # 5) 小模型自评
        if self.cfg.classifier_mode == "heuristic":
            # 纯规则模式：快筛判不了就保守升级（不花钱做自评）
            return route_to_frontier(
                self.cfg, new_state, reason="纯启发式模式下判定为不确定 → 保守升级", hits=[], started=started
            )

        result = self._self_eval(new_state, query)
        new_state["confidence"] = result.confidence
        new_state["difficulty"] = result.difficulty
        new_state["self_eval_raw"] = result.raw
        if result.warnings:
            new_state["warnings"] = list(new_state.get("warnings") or []) + result.warnings

        threshold = self._threshold()
        if result.confidence < threshold:
            return route_to_frontier(
                self.cfg,
                new_state,
                reason=f"自评置信度 {result.confidence:.2f} 低于阈值 {threshold:.2f} → 保守升级",
                hits=[],
                started=started,
                confidence=result.confidence,
            )
        return route_to_local(
            self.cfg,
            new_state,
            stage="self_eval",
            confidence=result.confidence,
            reason=f"自评通过（{result.difficulty}），置信度 {result.confidence:.2f}",
            started=started,
        )

    # ---- 缓存 ----

    def _lookup_cache(
        self, state: GraphState, query: str, has_pii: bool, no_cache: bool
    ) -> CacheLookup | None:
        """查询语义缓存；未启用缓存时返回 None。"""
        if self.cache is None or not query:
            return None

        # PII / no_cache 请求连向量都不生成（避免留下痕迹）
        skip = no_cache or (has_pii and self.cfg.exclude_pii)
        vector = [] if skip else self.cache.embed(query)
        state["embedding"] = vector
        return self.cache.lookup(
            query=query,
            vector=vector,
            entities=state.get("entities") or {},
            model_version=state.get("model_version") or "",
            prompt_version=state.get("prompt_version") or "",
            has_pii=has_pii,
            no_cache=no_cache,
        )

    def _on_cache_hit(self, state: GraphState, lookup: CacheLookup, context_tokens: int) -> None:
        """缓存命中：直接定稿，不再调用任何模型。"""
        answer = lookup.answer or ""
        state.update(
            {
                "cache_hit": True,
                "cache_similarity": lookup.similarity,
                "cache_entity_consistent": lookup.entity_consistent,
                "cache_reason": lookup.reason,
                "cached_answer": answer,
                "final_output": answer,
                "route_stage": "cache",
                "route_tier": "cache",
                "final_tier": "cache",
                "route_reason": "语义缓存命中并通过二次校验",
                # 成本近似 0，token 仍计入 cached_input_tokens（保留省钱参照系）
                "usage": {
                    "input_tokens": 0,
                    "cached_input_tokens": context_tokens,
                    "output_tokens": estimate_tokens(answer),
                },
            }
        )

    # ---- 规则与判定 ----

    def _force_upgrade_hits(self, state: GraphState) -> list[str]:
        """README 的强制升级规则：4bit 量化对数学/代码/长上下文伤害是实测结论，
        规则让它们不落到小模型手里（配置里没开的规则 = 关闭）。"""
        features = state.get("features") or {}
        hits: list[str] = []

        if self.cfg.force_math_or_code and (
            features.get("contains_math") or features.get("contains_code")
        ):
            hits.append("contains_math_or_code")
        if int(state.get("context_tokens") or 0) > self.cfg.force_context_tokens:
            hits.append(f"context_tokens > {self.cfg.force_context_tokens}")
        if self.cfg.force_realtime_data and features.get("requires_realtime_data"):
            hits.append("requires_realtime_data")
        if self.cfg.force_strict_json and state.get("output_format") == "strict_json":
            hits.append("output_format == strict_json")
        return hits

    def _heuristic_screen(self, state: GraphState) -> tuple[bool, str]:
        """启发式快筛：只认「极短 + 命中简单模板 + 指令不多」。判不了就交给自评。"""
        features = state.get("features") or {}
        template = features.get("simple_template")
        context_tokens = int(state.get("context_tokens") or 0)
        instructions = int(features.get("instruction_count") or 0)

        if not template:
            return False, "未命中已知简单模板"
        if context_tokens >= self.cfg.simple_template_max_tokens:
            return False, f"上下文 {context_tokens} token 超过快筛上限 {self.cfg.simple_template_max_tokens}"
        if instructions > self.cfg.simple_template_max_instructions:
            return False, f"指令条数 {instructions} 超过快筛上限 {self.cfg.simple_template_max_instructions}"
        if features.get("language") not in {"zh", "en", "mixed"}:
            return False, f"语种 {features.get('language')} 不在量化友好范围内"
        return True, f"模板={template}, {context_tokens} token, {instructions} 条指令"

    def _self_eval(self, state: GraphState, query: str) -> SelfEvalResult:
        """小模型自评（实现见 utils/self_eval.py）。"""
        if self.cfg.classifier_mode == "trained":
            # 训练分类器是文档里的"终态"，当前没有可用的模型文件，
            # 不做任何假实现，退化为自评并明确告警
            state["warnings"] = list(state.get("warnings") or []) + [
                "classifier.mode=trained 尚未提供分类器实现，已退化为 tiny_model_self_eval"
            ]
        if self.client is None or not self.cfg.local_tiers:
            return SelfEvalResult(
                confidence=0.0,
                difficulty="unknown",
                raw="",
                warnings=["自评不可用（无客户端或本地档位），按未知处理"],
            )

        tier = self.cfg.local_tiers[0]  # 只用最小的本地模型（约 8ms）
        return evaluate_difficulty(
            self.client,
            tiny_tier=tier.name,
            tiny_model=tier.model,
            query=query,
            timeout_s=self.cfg.self_eval_timeout_s,
        )

    def _threshold(self) -> float:
        """当前生效阈值：优先取守门监控器的动态阈值。"""
        if self.guardian is not None:
            return self.guardian.threshold
        return self.cfg.confidence_threshold

def _elapsed_ms(started: float) -> float:
    """秒 → 毫秒（本模块私有工具）。"""
    return round((time.perf_counter() - started) * 1000, 3)
