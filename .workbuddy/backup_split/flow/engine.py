"""极简状态图引擎（纯标准库）：节点 + 静态边 + 条件边。

为什么自己写而不引入 LangGraph：
  1. 这里只需要「节点 + 静态边 + 条件边」三件套，图的规模是个位数；
  2. 少一个重依赖，就少一分发版、依赖冲突与行为不可控的风险；
  3. 图的执行语义（异常如何冒泡、状态如何拷贝）必须完全可控 ——
     这是路由系统的核心，交给框架反而不好排查。

引擎能力：
  - 节点可以是 BaseNode 实例，也可以是任何 `state -> state` 的可调用对象；
  - 支持静态边（add_edge）与条件边（add_conditional_edges）；
  - 编译期校验：入口缺失、边指向不存在、路由分支未注册，全部提前报错；
  - 运行期保护：步数上限（防环）、状态类型校验、未声明字段告警。

字段拼写检查：合法字段 = 公共 GraphState ∪ 各节点声明的 state_schema。
节点把本节点独有的字段声明在 `state_schema` 后，拼错字段名会在
stream/invoke 时被日志告警出来。
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
        self._node_schemas: set[type] = set()
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
        # 收集节点级状态 schema，编译时并入合法字段集合
        node_schema = getattr(node, "state_schema", None)
        if isinstance(node_schema, type):
            self._node_schemas.add(node_schema)
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

        # 合法字段 = 公共 schema + 全部节点级 schema
        allowed = state_fields(self._schema, *self._node_schemas)
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
        logger.warning("检测到未声明的状态字段: %s（请在 GraphState 或对应节点的 state_schema 中声明）", sorted(unknown))
