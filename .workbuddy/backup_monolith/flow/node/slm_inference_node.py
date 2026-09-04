"""本地 SLM 推理 + 输出自检。

链路位置：cache_query → slm_inference → {result_process | cloud_inference}

README 主流程图里这一步是「本地 vLLM 推理 → 输出自检 → 异常即升级」。
自检是保守路由的**第二道保险**：路由已经放行了，但小模型实际输出崩了，
必须在返回给用户之前拦下来重来一次。

流式场景（SSE）的特殊处理：流没结束之前无法判断长度与 JSON 完整性，
所以只做前缀检查（拒答词 + 复读），流结束后再补一次完整检查并记录。
"""

from __future__ import annotations

import logging
import time

from config.config import RoutingConfig, routing_config
from flow.base import BaseNode
from flow.state import GraphState
from utils.llm_client import ChatClient
from utils.metering import merge_usage
from utils.selfcheck import check_output
from utils.heuristics import estimate_tokens

logger = logging.getLogger(__name__)

# 前缀自检的最小文本长度：太短了一段话都看不出崩没崩
PREFIX_CHECK_MIN_CHARS = 120




class SLMInferenceNode(BaseNode):
    """调用本地 vLLM 推理，并对输出做体检。"""

    name = "slm_inference"

    def __init__(self, *, client: ChatClient | None = None, config: RoutingConfig | None = None) -> None:
        super().__init__()
        self.client = client
        self.cfg = config or routing_config

    def process(self, state: GraphState) -> GraphState:
        new_state = dict(state)
        new_state.setdefault("latency_ms", {})
        new_state.setdefault("warnings", [])

        # 上游失败 → 直接交给上层路由去升级，别在这里浪费时间
        if new_state.get("error"):
            return new_state

        tier_name = new_state.get("route_tier")

        local_names = {item.name for item in self.cfg.local_tiers}
        if tier_name not in local_names:
            # 路由没有选择本地档位（比如缓存命中），本节点不该被执行
            logger.debug("route_tier=%s 不是本地档位，跳过 SLM 推理", tier_name)
            return new_state

        tier = self.cfg.tier(tier_name)
        if self.client is None:
            raise RuntimeError(f"档位 {tier_name} 没有可用的推理客户端")

        messages = new_state.get("messages") or []
        require_json = new_state.get("output_format") == "strict_json"
        response_format = {"type": "json_object"} if require_json else None
        started = time.perf_counter()

        if new_state.get("stream"):
            content, usage, check = self._run_stream(
                tier.name, tier.model, messages, require_json, new_state
            )
        else:
            response = self.client.chat(
                tier.name,
                tier.model,
                messages,
                temperature=float(new_state.get("temperature", 0.7)),
                max_tokens=int(new_state.get("max_tokens") or 1024),
                timeout=self.cfg.slm_timeout_s,
                response_format=response_format,
            )
            content = response.content
            usage = response.usage
            check = self._check(content, require_json, new_state, mode="full")

        new_state["latency_ms"]["slm_ms"] = round((time.perf_counter() - started) * 1000, 3)
        new_state["slm_response"] = content
        new_state["usage"] = merge_usage(new_state.get("usage"), usage)
        new_state["self_check"] = check
        new_state["final_tier"] = tier.name

        if not check["passed"]:
            # 自检不通过：标记待升级，由路由决定是否还有重试机会
            new_state["route_stage"] = "self_check_upgrade"
            new_state["warnings"] = new_state["warnings"] + [
                f"SLM 输出自检未通过（{'/'.join(check['reasons'])}），准备升级到强模型"
            ]
            logger.warning("SLM 输出自检未通过: %s", check["reasons"])
        return new_state

    # ------------------------------------------------------------------ #

    def _run_stream(
        self,
        tier: str,
        model: str,
        messages: list[dict[str, str]],
        require_json: bool,
        state: GraphState,
    ) -> tuple[str, dict[str, int], dict[str, object]]:
        """流式推理：边收边做前缀检查，流结束后补一次完整检查。"""
        chunks: list[str] = []
        prefix_checked = False
        early_failure: dict[str, object] | None = None

        for delta in self.client.chat_stream(  # type: ignore[union-attr]
            tier,
            model,
            messages,
            temperature=float(state.get("temperature", 0.7)),
            max_tokens=int(state.get("max_tokens") or 1024),
            timeout=self.cfg.slm_timeout_s,
        ):
            chunks.append(delta)
            text = "".join(chunks)
            if not prefix_checked and len(text) >= PREFIX_CHECK_MIN_CHARS:
                # 前缀检查只做拒答与复读 —— 长度和 JSON 完整性此时无从判断
                prefix_result = self._check(text, require_json, state, mode="prefix")
                prefix_checked = True
                if not prefix_result["passed"]:
                    early_failure = prefix_result
                    break

        content = "".join(chunks)
        # 流式下服务端 usage 依赖 stream_options.include_usage 回传，
        # 客户端没拿到就以估算值兜底，保证计量不至于缺失
        usage = {
            "input_tokens": int(state.get("context_tokens") or 0),
            "cached_input_tokens": 0,
            "output_tokens": estimate_tokens(content),
        }

        if early_failure is not None:
            return content, usage, early_failure

        # 流已经结束，这里补一次完整检查（README：完整流结束后异步补检并记录）
        full = self._check(content, require_json, state, mode="full")
        if not full["passed"] and prefix_checked:
            # 前缀检查已通过但最终不合格，说明问题出在后半段
            state["warnings"] = list(state.get("warnings") or []) + [
                "流式输出前缀检查通过但最终检查未通过（问题出现在后半段）"
            ]
        return content, usage, full

    def _check(
        self, text: str, require_json: bool, state: GraphState, mode: str
    ) -> dict[str, object]:
        """输出自检：参数全部来自配置，便于按业务调优。"""
        return check_output(
            text,
            require_json=require_json,
            min_output_tokens=self.cfg.min_output_tokens,
            max_repeat_ratio=self.cfg.max_repeat_ngram_ratio,
            min_confidence=self.cfg.confidence_threshold,
            confidence=float(state.get("confidence") or 0.0),
            mode=mode,
        )
