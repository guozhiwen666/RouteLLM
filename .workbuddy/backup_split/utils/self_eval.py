"""小模型自评能力：让 0.6B 模型判断「这题我答得了吗」。

README 的三层兜底里这是中间层（启发式 < 自评 < 分类器）：
自评成本低（约 8ms）、泛化好，是主力方案。

**fail-safe 纪律**：三种失败路径（超时 / 调用异常 / JSON 解析失败）一律
按置信度 0 处理 —— 自评模型自己出问题，就当它说「我不确定」，
由上层升级到强模型。宁可不省这笔钱，不可错配弱模型。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from utils.llm_client import ChatClient
from utils.text_utils import extract_json_object

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
