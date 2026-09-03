"""RouteLLM 工作流编排：把节点与路由规则组装成一条可执行链路。

本模块只做「装配」：
  - 具体图引擎（StateGraph / CompiledGraph）在 `flow/engine.py`；
  - 六个业务节点在 `flow/node/`，各自声明了本节点的状态 schema；
  - 支撑能力（缓存 / 客户端 / 计量 / 守门）在 `utils/`。

链路（与 README 主流程图一致）：

    query_process → cache_query ─┬─ 缓存命中 ─────────────→ result_process
                                 ├─ 本地档位 → slm_inference ─┤
                                 └─ 强制升级 → cloud_inference ─┤
                                                                ↓
                                   result_process → {影子流量}? → END

所有外部依赖（客户端 / 缓存 / 计量 / 守门）都可注入，便于测试替换成假实现、
生产替换成 Redis / Prometheus 等真实实现。
"""

from __future__ import annotations

import logging
from typing import Any

from config.config import RoutingConfig, routing_config

from flow.base import BaseNode
from flow.engine import END, START, CompiledGraph, GraphError, StateGraph  # noqa: F401 - 兼容旧导入路径
from flow.node.cache_query_node import CacheQueryNode, CacheQueryState
from flow.node.cloud_inference_node import CloudInferenceNode
from flow.node.comparative_scoring_node import ComparativeScoringNode
from flow.node.query_process_node import QueryProcessNode
from flow.node.result_process_node import ResultProcessNode
from flow.node.slm_inference_node import SLMInferenceNode
from flow.state import GraphState
from utils.guardian import GuardianMonitor
from utils.llm_client import ChatClient, Endpoint
from utils.metering import Meter
from utils.semantic_cache import SemanticCache

logger = logging.getLogger(__name__)

# 路由分支名（条件边的返回值）
BRANCH_CACHE_HIT = "cache_hit"
BRANCH_SLM = "slm"
BRANCH_FRONTIER = "frontier"
BRANCH_UPGRADE = "upgrade"
BRANCH_OK = "ok"
BRANCH_FAILED = "failed"
BRANCH_SHADOW = "shadow"
BRANCH_END = "end"


def build_chat_client(
    cfg: RoutingConfig,
    *,
    api_keys: list[str] | None = None,
    base_url_override: str | None = None,
    timeout: float | None = None,
    **client_kwargs: Any,
) -> ChatClient:
    """按配置构建客户端：把每个档位注册成一个（或多个）节点。

    - 本地档位：endpoint 就是 vLLM 实例地址；
    - 云端档位：endpoint 为 "upstream" 时用 LLM 配置的 base_url；
      API Key 可传多个实现轮转。
    """
    client = ChatClient(
        timeout=timeout or cfg.cloud_timeout_s,
        max_retries=cfg.max_retries,
        backoff_base_s=cfg.backoff_base_s,
        **client_kwargs,
    )
    keys = [key.strip() for key in (api_keys or []) if key and key.strip()]

    for tier in cfg.tiers:
        if tier.is_local:
            client.add_endpoint(tier.name, Endpoint(name=tier.name, base_url=tier.endpoint, api_keys=list(keys)))
            continue
        base_url = base_url_override or ("" if tier.endpoint == "upstream" else tier.endpoint)
        if not base_url:
            logger.warning(
                "档位 %s 未配置可用地址（endpoint=%s），调用时会报 NoEndpointError",
                tier.name,
                tier.endpoint,
            )
            continue
        client.add_endpoint(tier.name, Endpoint(name=tier.name, base_url=base_url, api_keys=list(keys)))
    return client


class Workflow:
    """RouteLLM 主链路编排（见模块 docstring 的链路图）。"""

    def __init__(
        self,
        *,
        client: ChatClient | None = None,
        cache: SemanticCache | None = None,
        meter: Meter | None = None,
        guardian: GuardianMonitor | None = None,
        config: RoutingConfig | None = None,
        max_steps: int = 32,
    ) -> None:
        self.cfg = config or routing_config
        self.client = client
        self.meter = meter or Meter()
        self.guardian = guardian or GuardianMonitor(
            shadow_ratio=self.cfg.shadow_ratio,
            max_quality_drop=self.cfg.max_quality_drop,
            consecutive_windows=self.cfg.consecutive_windows,
            auto_rollback=self.cfg.auto_rollback,
            threshold=self.cfg.confidence_threshold,
            threshold_max_step=self.cfg.threshold_max_step,
            threshold_min=self.cfg.threshold_min,
            threshold_max=self.cfg.threshold_max,
            threshold_cooldown_seconds=self.cfg.threshold_cooldown_seconds,
            window_min_samples=self.cfg.window_min_samples,
            extreme_ratio=self.cfg.shadow_extreme_ratio,
        )
        self.cache = cache or SemanticCache(
            threshold=self.cfg.semantic_threshold,
            ttl_hours=self.cfg.cache_ttl_hours,
            entity_check=self.cfg.entity_consistency_check,
            exclude_pii=self.cfg.exclude_pii,
            invalidate_on=self.cfg.cache_invalidate_on,
            length_ratio_min=self.cfg.cache_length_ratio_min,
            length_ratio_max=self.cfg.cache_length_ratio_max,
        )

        # 1. 初始化工作流
        self.workflow = StateGraph(GraphState)
        # 2. 初始化节点
        self._init_nodes()
        # 3. 注册节点
        self._register_nodes()
        # 4. 设置入口与路由规则
        self._setup_routes()
        # 5. 延迟编译（首次执行时编译，之后复用）
        self._compiled_app: CompiledGraph | None = None
        self._max_steps = max_steps

    # ------------------------------------------------------------------ #

    def _init_nodes(self) -> None:
        """创建所有业务节点（依赖注入集中在这里，测试替换都改这一处）。"""
        self.query_process_node = QueryProcessNode()
        self.cache_query_node = CacheQueryNode(
            client=self.client, cache=self.cache, guardian=self.guardian, config=self.cfg
        )
        self.slm_inference_node = SLMInferenceNode(client=self.client, config=self.cfg)
        self.cloud_inference_node = CloudInferenceNode(client=self.client, config=self.cfg)
        self.result_process_node = ResultProcessNode(
            cache=self.cache, meter=self.meter, guardian=self.guardian, config=self.cfg
        )
        self.comparative_scoring_node = ComparativeScoringNode(
            client=self.client, guardian=self.guardian, meter=self.meter, config=self.cfg
        )

    def _register_nodes(self) -> None:
        """注册节点到图。节点标识与实例属性名保持一致，便于对照维护。"""
        self.workflow.add_node("query_process", self.query_process_node)
        self.workflow.add_node("cache_query", self.cache_query_node)
        self.workflow.add_node("slm_inference", self.slm_inference_node)
        self.workflow.add_node("cloud_inference", self.cloud_inference_node)
        self.workflow.add_node("result_process", self.result_process_node)
        self.workflow.add_node("comparative_scoring", self.comparative_scoring_node)

    def _setup_routes(self) -> None:
        """定义边：入口、两条条件边、两条静态边。"""
        self.workflow.set_entry_point("query_process")
        self.workflow.add_edge(START, "query_process")
        self.workflow.add_edge("query_process", "cache_query")

        self.workflow.add_conditional_edges(
            "cache_query",
            self._route_after_cache_query,
            {
                BRANCH_CACHE_HIT: "result_process",
                BRANCH_SLM: "slm_inference",
                BRANCH_FRONTIER: "cloud_inference",
                BRANCH_FAILED: "result_process",
            },
        )
        self.workflow.add_conditional_edges(
            "slm_inference",
            self._route_after_slm,
            {
                BRANCH_OK: "result_process",
                BRANCH_UPGRADE: "cloud_inference",
                BRANCH_FAILED: "result_process",
            },
        )
        self.workflow.add_edge("cloud_inference", "result_process")
        self.workflow.add_conditional_edges(
            "result_process",
            self._route_after_result,
            {BRANCH_SHADOW: "comparative_scoring", BRANCH_END: END},
        )
        self.workflow.add_edge("comparative_scoring", END)

    # ------------------------------------------------------------------ #
    # 条件路由
    # ------------------------------------------------------------------ #

    def _local_tier_names(self) -> set[str]:
        return {tier.name for tier in self.cfg.local_tiers}

    def _route_after_cache_query(self, state: CacheQueryState) -> str:
        """缓存查询之后：命中直接返回，否则按档位分流。"""
        # 上游失败且没有可用输出 → 直接收尾，避免带着错误继续烧钱
        if state.get("error") and not state.get("final_output"):
            return BRANCH_FAILED
        if state.get("cache_hit"):
            return BRANCH_CACHE_HIT

        tier = state.get("route_tier")
        if tier in self._local_tier_names():
            return BRANCH_SLM
        if tier == self.cfg.frontier_tier().name:
            return BRANCH_FRONTIER
        # 走到了未知档位，说明路由逻辑有 bug —— 明确落到收尾并留痕
        logger.error("未知的路由档位 %r，直接结束链路", tier)
        return BRANCH_FAILED

    def _route_after_slm(self, state: GraphState) -> str:
        """SLM 推理之后：自检不通过或调用失败就升级到云端。"""
        check = state.get("self_check") or {}
        check_failed = bool(check) and not check.get("passed", True)

        if state.get("error"):
            # 本地实例不可用（README 边界处理：健康检查 + 自动切云端）
            return BRANCH_UPGRADE if self._can_upgrade(state) else BRANCH_FAILED
        if check_failed:
            # 自检未通过：还有重试机会就升级，否则原样返回（降级但不中断）
            return BRANCH_UPGRADE if self._can_upgrade(state) else BRANCH_OK
        return BRANCH_OK

    def _route_after_result(self, state: GraphState) -> str:
        """结果处理之后：被抽中的请求进入影子对比，否则结束。"""
        return BRANCH_SHADOW if state.get("shadow") else BRANCH_END

    def _can_upgrade(self, state: GraphState) -> bool:
        """是否还能再升级一次（防止在 SLM 与云端之间来回打乒乓）。"""
        return int(state.get("upgrade_count") or 0) < self.cfg.max_upgrade_attempts

    # ------------------------------------------------------------------ #
    # 执行
    # ------------------------------------------------------------------ #

    def compile(self) -> CompiledGraph:
        """编译工作流（幂等）。"""
        if self._compiled_app is None:
            self._compiled_app = self.workflow.compile(max_steps=self._max_steps)
        return self._compiled_app

    def run(self, init_state: GraphState, stream: bool = False):
        """执行工作流。

        :param init_state: 初始状态，至少包含 messages
        :param stream: True 时返回逐步产出的迭代器
        """
        if not isinstance(init_state, dict):
            raise TypeError(f"init_state 必须是 dict，实际为 {type(init_state).__name__}")
        app = self.compile()
        return app.stream(init_state) if stream else app.invoke(init_state)

    @classmethod
    def create_and_run(cls, init_state: GraphState, stream: bool = False, **kwargs: Any):
        """快捷方法：创建工作流实例并执行。"""
        return cls(**kwargs).run(init_state, stream=stream)

    # ------------------------------------------------------------------ #
    # 运维接口
    # ------------------------------------------------------------------ #

    def metrics(self) -> str:
        """Prometheus 文本格式的指标（可直接挂到 /metrics）。"""
        return self.meter.render_prometheus()

    def summary(self) -> dict[str, Any]:
        """一次看全：计量汇总 + 缓存统计 + 守门状态。"""
        return {
            "metering": self.meter.summary(),
            "cache": self.cache.stats(),
            "guardian": self.guardian.snapshot(),
        }

    def shutdown(self) -> None:
        """优雅退出：等待影子任务结束并释放线程池。"""
        self.comparative_scoring_node.shutdown(wait=True)
