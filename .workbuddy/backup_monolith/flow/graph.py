"""状态图引擎 + Workflow 装配（回滚后的单文件布局）。

前半部分为自写的极简状态图引擎（StateGraph / CompiledGraph，纯标准库，
替代 LangGraph）；后半部分为 Workflow：把六个业务节点装配成 README 主流程图
描述的执行链，并持有依赖注入。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from flow.base import BaseNode
from flow.state import GraphState, state_fields

logger = logging.getLogger(__name__)

# 起止哨兵
START = "__start__"
END = "__end__"


class GraphError(RuntimeError):
    """图定义或执行期的结构性错误。"""


class StateGraph:
    """有向状态图（支持静态边与条件边）。"""

    def __init__(self, schema: type | None = None) -> None:
        # 公共状态 schema（如 GraphState），节点级 schema 由 add_node 收集
        self._schema = schema
        self._nodes: dict[str, Any] = {}
        self._static_edges: dict[str, str] = {}
        self._conditional_edges: dict[str, tuple[Callable[[GraphState], str], dict[str, str]]] = {}
        self._entry: str | None = None

    # ---------------- 定义 ---------------- #

    def add_node(self, name: str, node: Any) -> "StateGraph":
        """注册节点。node 可以是 BaseNode 实例，也可以是 (state) -> state 的可调用对象。"""
        if not isinstance(name, str) or not name:
            raise GraphError("节点名必须是非空字符串")
        if name in {START, END}:
            raise GraphError(f"节点名不能使用保留字 {name}")
        if name in self._nodes:
            raise GraphError(f"节点重复注册: {name}")
        if not isinstance(node, BaseNode) and not callable(node):
            raise GraphError(f"节点 {name} 必须是 BaseNode 实例或可调用对象")

        self._nodes[name] = node
        return self

    def add_edge(self, source: str, target: str) -> "StateGraph":
        """添加静态边。source 可以是 START，target 可以是 END。"""
        if not source or not target:
            raise GraphError("边的两端不能为空")
        if source in self._conditional_edges:
            raise GraphError(f"节点 {source} 已注册条件边，不能再注册静态边")
        self._static_edges[source] = target
        return self

    def add_conditional_edges(
        self, source: str, router: Callable[[GraphState], str], mapping: dict[str, str]
    ) -> "StateGraph":
        """添加条件边：由 router(state) 返回分支名，再按 mapping 找到下一个节点。"""
        if not source:
            raise GraphError("条件边的起点不能为空")
        if not callable(router):
            raise GraphError(f"节点 {source} 的路由函数必须是可调用对象")
        if not isinstance(mapping, dict) or not mapping:
            raise GraphError(f"节点 {source} 的条件边映射不能为空")
        for branch, target in mapping.items():
            if not isinstance(branch, str) or not isinstance(target, str):
                raise GraphError(f"节点 {source} 的条件边映射键值必须都是字符串")
        if source in self._static_edges:
            raise GraphError(f"节点 {source} 已注册静态边，不能再注册条件边")
        self._conditional_edges[source] = (router, mapping)
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        if name not in self._nodes:
            raise GraphError(f"入口节点未注册: {name}")
        self._entry = name
        return self

    # ---------------- 编译 ---------------- #

    def compile(self, max_steps: int = 32) -> "CompiledGraph":
        """编译为可执行图，编译期做完整性校验。"""
        if self._entry is None:
            raise GraphError("入口节点未设置（请先调用 set_entry_point）")
        if self._entry not in self._nodes:
            raise GraphError(f"入口节点未注册: {self._entry}")

        known = set(self._nodes) | {END}
        for source, target in self._static_edges.items():
            if source != START and source not in self._nodes:
                raise GraphError(f"静态边的起点未注册: {source}")
            if target not in known:
                raise GraphError(f"静态边 {source} -> {target} 的终点未注册")
        for source, (_router, mapping) in self._conditional_edges.items():
            if source not in self._nodes:
                raise GraphError(f"条件边的起点未注册: {source}")
            for branch, target in mapping.items():
                if target not in known:
                    raise GraphError(f"条件边 {source} -[{branch}]-> {target} 的终点未注册")

        # 合法字段 = GraphState 声明的全部字段
        allowed = state_fields(self._schema)
        return CompiledGraph(self, max_steps=max_steps, allowed_keys=allowed)

    # ---------------- 内部访问（供 CompiledGraph 使用） ---------------- #

    @property
    def nodes(self) -> dict[str, Any]:
        return self._nodes

    @property
    def static_edges(self) -> dict[str, str]:
        return self._static_edges

    @property
    def conditional_edges(self) -> dict[str, tuple[Callable[[GraphState], str], dict[str, str]]]:
        return self._conditional_edges

    @property
    def entry(self) -> str | None:
        return self._entry

    @property
    def schema(self) -> type | None:
        return self._schema


class CompiledGraph:
    """编译后的可执行图。"""

    def __init__(
        self,
        graph: StateGraph,
        max_steps: int = 32,
        allowed_keys: set[str] | None = None,
    ) -> None:
        if max_steps <= 0:
            raise GraphError("max_steps 必须为正数")
        self._graph = graph
        self._max_steps = max_steps
        self._allowed_keys = allowed_keys

    def invoke(self, state: GraphState) -> GraphState:
        """执行到 END，返回最终状态。"""
        final = dict(state)
        for _name, snapshot in self.stream(final):
            final = snapshot
        return final

    def stream(self, state: GraphState) -> Iterator[tuple[str, GraphState]]:
        """逐步执行，每执行完一个节点产出 (节点名, 状态快照)。"""
        if not isinstance(state, dict):
            raise TypeError(f"初始状态必须是 dict，实际为 {type(state).__name__}")

        current_state: GraphState = dict(state)
        _warn_unknown_keys(current_state, self._allowed_keys)
        trace: list[str] = list(current_state.get("trace") or [])

        entry = self._graph.entry
        if entry is None:  # pragma: no cover - 编译期已校验
            raise GraphError("入口节点未设置")

        current = entry
        steps = 0
        while current != END:
            steps += 1
            if steps > self._max_steps:
                # 图里出现环就会走到这里，宁可报错也不要把进程跑死
                raise GraphError(f"执行步数超过上限 {self._max_steps}，可能存在循环边（trace={trace}）")

            node = self._graph.nodes.get(current)
            if node is None:
                raise GraphError(f"节点未注册: {current}")

            current_state = _execute_node(node, current_state)
            trace.append(current)
            current_state["trace"] = list(trace)
            _warn_unknown_keys(current_state, self._allowed_keys)

            yield current, dict(current_state)
            current = self._resolve_next(current, current_state)

    def _resolve_next(self, node_name: str, state: GraphState) -> str:
        """求解下一个节点：条件边优先，其次静态边，都没有就是 END。"""
        conditional = self._graph.conditional_edges.get(node_name)
        if conditional is not None:
            router, mapping = conditional
            branch = router(state)
            if branch not in mapping:
                raise GraphError(
                    f"节点 {node_name} 的路由函数返回了未注册分支 {branch!r}，可选: {sorted(mapping)}"
                )
            return mapping[branch]
        return self._graph.static_edges.get(node_name, END)


def _execute_node(node: Any, state: GraphState) -> GraphState:
    """执行单个节点并校验返回值。"""
    result = node.run(state) if isinstance(node, BaseNode) else node(state)
    if not isinstance(result, dict):
        raise TypeError(f"节点 {node} 必须返回 dict，实际返回 {type(result).__name__}")
    return result


def _warn_unknown_keys(state: GraphState, allowed_keys: set[str] | None) -> None:
    """状态里出现未声明字段时告警（不中断执行，但拼错字段会立刻暴露）。"""
    if allowed_keys is None or not allowed_keys:
        return
    unknown = set(state) - allowed_keys
    if unknown:
        logger.warning("检测到未声明的状态字段: %s（请在 flow/state.py 的 GraphState 中声明）", sorted(unknown))


import logging
from typing import Any

from config.config import RoutingConfig, routing_config

from flow.base import BaseNode
from flow.node.cache_query_node import CacheQueryNode
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

    def _route_after_cache_query(self, state: GraphState) -> str:
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
