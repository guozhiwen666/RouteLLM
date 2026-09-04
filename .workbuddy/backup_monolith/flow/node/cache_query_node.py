"""语义缓存查询 + 路由决策（整套系统的大脑）。

链路位置：query_process → cache_query → {result_process | slm_inference | cloud_inference}

流程：缓存查询 → 强制升级规则 → 启发式快筛 → 小模型自评 → 定档。
强制升级规则命中时自评是**无效的**，故先跑规则（命中即短路，省下 8ms）。
核心原则：**任何不确定都升级到强模型**（fail-safe）。
"""

from __future__ import annotations

import time

from config.config import RoutingConfig, routing_config
from flow.base import BaseNode
from flow.state import GraphState
from utils.guardian import GuardianMonitor
from utils.llm_client import ChatClient
from utils.semantic_cache import CacheLookup, SemanticCache
from utils.heuristics import estimate_tokens


class CacheQueryNode(BaseNode):
    """语义缓存查询与路由决策节点。"""

    name = "cache_query"

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


# ================ 小模型自评能力（原 utils/self_eval.py，回滚后内联） ================

import logging
from dataclasses import dataclass, field
from typing import Any

from utils.llm_client import ChatClient
from utils.heuristics import extract_json_object

logger = logging.getLogger(__name__)

# 自评提示词：要求严格 JSON，且明确「宁可说不行」
SELF_EVAL_SYSTEM = (
    "你是路由难度评估器。判断给定请求能否由一个 0.6B 的本地量化小模型正确完成。\n"
    "判定标准：格式转换、简单改写、分类、抽取、短摘要 → 可以；\n"
    "数学计算、代码生成、长文档推理、多步规划、小语种 → 不行。\n"
    "拿不准就判定为不行。只输出严格 JSON，不要任何多余文字。"
)
SELF_EVAL_TEMPLATE = (
    "待评估请求：\n{query}\n\n"
    '输出 JSON：{{"can_answer": true/false, "confidence": 0~1, '
    '"difficulty": "simple|moderate|hard", "reason": "..."}}'
)


@dataclass
class SelfEvalResult:
    """一次自评的输出。"""

    confidence: float  # 0~1；置信度越低越要升级
    difficulty: str  # simple / moderate / hard / unknown
    raw: str  # 模型原始输出，排查自评失效时有用
    warnings: list[str] = field(default_factory=list)


def evaluate_difficulty(
    client: ChatClient,
    *,
    tiny_tier: str,
    tiny_model: str,
    query: str,
    timeout_s: float = 0.3,
) -> SelfEvalResult:
    """让最小的本地模型评估一条请求的难度。

    :param client: 已注册节点的模型客户端
    :param tiny_tier: 最小的本地档位名（约 8ms，不去占用 8B 实例）
    :param query: 归一化后的请求文本（截断到 2000 字符防超长）
    :return: 置信度 + 难度 + 原始输出；任何失败路径置信度恒为 0.0
    """
    prompt = [
        {"role": "system", "content": SELF_EVAL_SYSTEM},
        {"role": "user", "content": SELF_EVAL_TEMPLATE.format(query=query[:2000])},
    ]

    try:
        response = client.chat(
            tiny_tier,
            tiny_model,
            prompt,
            temperature=0.0,
            max_tokens=128,
            timeout=timeout_s,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001 - 自评失败必须降级为「不确定」而不是中断链路
        return SelfEvalResult(
            confidence=0.0,
            difficulty="unknown",
            raw="",
            warnings=[f"自评调用失败，按未知处理并保守升级: {type(exc).__name__}: {exc}"],
        )

    raw = response.content or ""
    parsed = extract_json_object(raw)
    if parsed is None:
        return SelfEvalResult(
            confidence=0.0,
            difficulty="unknown",
            raw=raw,
            warnings=[f"自评输出无法解析为 JSON，按未知处理: {raw[:200]!r}"],
        )

    confidence = _to_float(parsed.get("confidence"))
    difficulty = str(parsed.get("difficulty") or "").strip().lower()
    if difficulty not in {"simple", "moderate", "hard"}:
        difficulty = "unknown"

    if parsed.get("can_answer") is False:
        # 模型自己说不行 —— 那就别硬撑
        return SelfEvalResult(confidence=0.0, difficulty=difficulty or "hard", raw=raw)
    if confidence is None:
        return SelfEvalResult(
            confidence=0.0,
            difficulty=difficulty,
            raw=raw,
            warnings=["自评缺少 confidence 字段，按未知处理"],
        )
    return SelfEvalResult(confidence=confidence, difficulty=difficulty, raw=raw)


def _to_float(value: Any) -> float | None:
    """宽松地把自评输出里的 confidence 转成 [0,1] 的浮点数。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return max(0.0, min(1.0, number))


# ================ 路由落点（原 utils/route_helpers.py，回滚后内联） ================

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
