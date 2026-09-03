"""节点基类：所有业务节点的统一契约。

另外一个关键点是 `run()`：节点内部抛出的异常**不应该炸掉整条链路**，
而应该被记录到 state['error']，交给条件路由去做保守决策（fail-safe）。
比如 SLM 实例挂了，正确的行为是「记错 + 升级到云端」，而不是把请求直接 500 掉。

关于状态字段：`GraphState` 只保留公共字段，节点独有的字段由节点自己在
所在模块里声明（`class XxxState(GraphState)`），并通过 `state_schema` 类属性
注册给图引擎 —— 引擎据此校验状态字段拼写，拼错会立刻告警。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from flow.state import GraphState

logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """工作流节点基类。

    子类必须：
      1. 设置唯一的类属性 `name`；
      2. 实现 `process(state) -> state`；
      3. 如有本节点独有的状态字段，声明 `state_schema` 供引擎校验。
    """

    name: str = "base_node"

    # 本节点独有的状态字段声明（GraphState 的子类），没有则留空
    state_schema: type | None = None

    def __init__(self) -> None:
        """强制子类设置 name（沿用原实现的校验逻辑）。"""
        if not self.name or self.name == "base_node":
            raise ValueError(f"{self.__class__.__name__} 必须设置 name 属性")

    @abstractmethod
    def process(self, state: GraphState) -> GraphState:
        """节点核心逻辑。

        约定：入参 state 视为只读，返回修改后的新 state（先 `dict(state)` 再改）。
        """
        raise NotImplementedError

    def run(self, state: GraphState) -> GraphState:
        """带统一异常处理的执行入口，供图引擎调用。

        节点抛异常时只在 state 里留痕，不中断链路 —— 决策权交给路由。
        """
        try:
            result = self.process(state)
        except Exception as exc:  # noqa: BLE001 - 兜底，避免单节点异常打穿整条链路
            logger.exception("节点 %s 执行失败", self.name)
            new_state = dict(state)
            new_state["error"] = f"{self.name}: {type(exc).__name__}: {exc}"
            new_state["warnings"] = list(state.get("warnings") or []) + [new_state["error"]]
            return new_state

        if not isinstance(result, dict):
            raise TypeError(f"节点 {self.name} 必须返回 dict 类型的 state，实际返回 {type(result).__name__}")
        return result

    def __call__(self, state: GraphState) -> GraphState:
        """让节点可以像函数一样被直接调用。"""
        return self.run(state)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"

    @staticmethod
    def updates(state: GraphState, **fields: Any) -> GraphState:
        """小工具：基于旧状态生成新状态并写入若干字段（保证不改原对象）。"""
        new_state = dict(state)
        new_state.update(fields)
        return new_state
