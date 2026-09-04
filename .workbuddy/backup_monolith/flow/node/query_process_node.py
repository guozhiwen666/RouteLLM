"""输入处理节点：归一化、上下文估算、启发式特征提取、实体与 PII 抽取。

链路位置：START → query_process → cache_query

它只做「看清这个请求是什么」，不做任何路由决策。分析产物全部写进
state（features / entities / pii_types / context_tokens 等），
供缓存二次校验、强制升级规则、启发式快筛共同复用。

本节点独有的状态字段见 `QueryProcessState`。
"""

from __future__ import annotations

import logging
import uuid

from config.config import llm_config
from flow.base import BaseNode
from flow.state import GraphState
from utils.heuristics import build_features
from utils.heuristics import detect_pii, estimate_messages_tokens, extract_entities, normalize_text

logger = logging.getLogger(__name__)

_VALID_ROLES = frozenset({"system", "user", "assistant"})
DEFAULT_MAX_TOKENS = 1024




class QueryProcessNode(BaseNode):
    """请求入口节点：把原始请求整理成下游节点可以安全消费的形态。"""

    name = "query_process"

    def process(self, state: GraphState) -> GraphState:
        messages = state.get("messages") or []
        self._validate_messages(messages)

        # ---- 1. 主文本：取最后一条 user 消息作为缓存/自评的对象 ----
        user_contents = [m.get("content", "") for m in messages if m.get("role") == "user"]
        query = (user_contents[-1] if user_contents else "").strip()
        normalized = normalize_text(query)

        # ---- 2. 上下文长度：整段对话都要算，不能只看最后一句 ----
        context_tokens = estimate_messages_tokens(messages)

        # ---- 3. 启发式特征：代码 / 数学 / 实时数据 / 指令条数 / 语种 / 简单模板 ----
        features = build_features(query, messages)

        # ---- 4. 输出格式：显式声明优先，其次从文本里识别 ----
        output_format = (state.get("output_format") or "").strip() or (
            "strict_json" if features["strict_json"] else "text"
        )

        # ---- 5. PII 与实体（对整个对话扫描，不只扫最后一句）----
        pii_types = detect_pii("\n".join(str(m.get("content") or "") for m in messages))
        entities = extract_entities(query)
        no_cache = bool(state.get("no_cache")) or bool(features["requires_realtime_data"])
        has_pii = bool(pii_types)

        updates = dict(state)
        updates.update(
            {
                "request_id": state.get("request_id") or uuid.uuid4().hex,
                "query": query,
                "normalized_query": normalized,
                "context_tokens": context_tokens,
                "features": features,
                "output_format": output_format,
                "pii_types": pii_types,
                "has_pii": has_pii,
                "entities": entities,
                "no_cache": no_cache,
                # 缺省值补齐，下游节点就可以无条件读取
                "prompt_version": state.get("prompt_version") or "v1",
                "model_version": state.get("model_version") or "unknown",
                "temperature": float(state.get("temperature", llm_config.llm_temperature)),
                "max_tokens": int(state.get("max_tokens") or DEFAULT_MAX_TOKENS),
                "stream": bool(state.get("stream", False)),
                "usage": state.get("usage") or {},
                "latency_ms": dict(state.get("latency_ms") or {}),
                "warnings": list(state.get("warnings") or []),
                "trace": list(state.get("trace") or []),
            }
        )

        if has_pii:
            updates["warnings"] = updates["warnings"] + [
                f"请求命中 PII 检测（{','.join(pii_types)}），不写缓存并优先本地推理"
            ]
        return updates

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> None:
        """入参校验：结构不对直接抛错，让调用方立刻知道自己传错了什么。"""
        if not messages:
            raise ValueError("messages 不能为空")
        if not isinstance(messages, list):
            raise TypeError(f"messages 必须是列表，实际为 {type(messages).__name__}")
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(f"messages[{index}] 必须是 dict，实际为 {type(message).__name__}")
            role = message.get("role")
            if role not in _VALID_ROLES:
                raise ValueError(f"messages[{index}].role={role!r} 非法，可选 {sorted(_VALID_ROLES)}")
            content = message.get("content")
            if content is None:
                raise ValueError(f"messages[{index}].content 不能为空")
            if not isinstance(content, str):
                raise TypeError(f"messages[{index}].content 必须是字符串")
            if not content.strip():
                raise ValueError(f"messages[{index}].content 不能是空白字符串")
        if not any(m.get("role") == "user" for m in messages):
            raise ValueError("messages 中至少需要一条 user 消息")
