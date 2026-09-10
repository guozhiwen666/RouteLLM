"""云端强模型推理（Fallback 链、多 Key 轮转、指数退避都在客户端里）。

链路位置：{cache_query | slm_inference} → cloud_inference → result_process

这是整条链路的**兜底出口**，所以这里的策略是「尽可能拿到一个答案」：
客户端已经负责了多 Key 轮转、跨供应商回退与指数退避，节点只负责
把结果（或失败原因）规整地写回 state，交给 result_process 去计量。
"""

from __future__ import annotations

import logging
import time

from config.loader import routing_config
from config.models import RoutingConfig
from flow.base import BaseNode
from flow.state import GraphState
from utils.llm_client import ChatClient
from utils.metering import merge_usage
from utils.selfcheck import check_output

logger = logging.getLogger(__name__)




class CloudInferenceState(GraphState):
    """云端推理阶段独有的状态字段。"""

    cloud_response: str  # 云端模型原始输出


class CloudInferenceNode(BaseNode):
    """云端推理节点。"""

    name = "cloud_inference"
    state_schema = CloudInferenceState

    def __init__(self, *, client: ChatClient | None = None, config: RoutingConfig | None = None) -> None:
        super().__init__()
        self.client = client
        self.cfg = config or routing_config

    def process(self, state: GraphState) -> GraphState:
        new_state = dict(state)
        new_state.setdefault("latency_ms", {})
        new_state.setdefault("warnings", [])

        if self.client is None:
            raise RuntimeError("云端推理没有可用的客户端")

        tier = self.cfg.frontier_tier()
        messages = new_state.get("messages") or []
        if not messages:
            raise ValueError("messages 为空，无法调用云端模型")

        require_json = new_state.get("output_format") == "strict_json"
        response_format = {"type": "json_object"} if require_json else None

        # 走到这里就说明发生了一次升级（无论是强制升级还是自检失败重试），
        # 计数放在本节点而不是路由函数里 —— 路由函数只做判断，不偷偷改状态
        was_retry = new_state.get("route_stage") == "self_check_upgrade"
        new_state["upgrade_count"] = int(new_state.get("upgrade_count") or 0) + 1

        started = time.perf_counter()
        try:
            response = self.client.chat(
                tier.name,
                tier.model,
                messages,
                temperature=float(new_state.get("temperature", 0.7)),
                max_tokens=int(new_state.get("max_tokens") or 1024),
                timeout=self.cfg.cloud_timeout_s,
                response_format=response_format,
            )
        except Exception as exc:  # noqa: BLE001 - 云端是兜底出口：失败要"留下来"而不是炸掉链路
            # 云端失败必须让计数保留下来（升级计数是防止打乒乓的关键），
            # 所以在这里收敛成错误状态返回，而不是抛给 BaseNode.run 去重建状态
            new_state["latency_ms"]["cloud_ms"] = round((time.perf_counter() - started) * 1000, 3)
            new_state["error"] = f"cloud_inference: {type(exc).__name__}: {exc}"
            new_state["warnings"] = list(new_state.get("warnings") or []) + [new_state["error"]]
            logger.error("云端推理失败（Fallback 链已耗尽）：%s", exc)
            return new_state
        new_state["latency_ms"]["cloud_ms"] = round((time.perf_counter() - started) * 1000, 3)

        new_state["cloud_response"] = response.content
        new_state["usage"] = merge_usage(new_state.get("usage"), response.usage)
        new_state["final_tier"] = tier.name
        new_state["route_stage"] = "self_check_upgrade" if was_retry else "cloud"

        # 云端已经是最后一级，没有地方可升级了。
        # 但仍然做一次体检并留痕 —— 强模型崩了这件事必须能被看见，而不是静默返回。
        check = check_output(
            response.content,
            require_json=require_json,
            min_output_tokens=self.cfg.min_output_tokens,
            max_repeat_ratio=self.cfg.max_repeat_ngram_ratio,
            mode="full",
        )
        new_state["self_check"] = check
        if not check["passed"]:
            new_state["warnings"] = new_state["warnings"] + [
                f"云端输出自检未通过且已无可升级路径: {'/'.join(check['reasons'])}"
            ]
            logger.error("云端输出自检未通过: %s", check["reasons"])
        return new_state
