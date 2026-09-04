"""输出自检：SLM 结果的轻量体检。

README 把它定位为「保守路由的第二道保险」—— 路由判定已经放行了，
但小模型实际输出得一塌糊涂时，必须在**返回给用户之前**拦下来并升级重试。

检查项全部来自文档：过短、重复 n-gram、格式错误（JSON 是否合法）、拒答词、自评低。
原则是**宁可误杀（多花一次云端推理的钱），不可放过（用户拿到垃圾答案）**。
"""

from __future__ import annotations

import json
import re
from typing import Any

from utils.heuristics import estimate_tokens

# 拒答/崩坏信号：模型一旦开始自我声明身份或道歉，输出基本不可用
_REFUSAL_PATTERNS = (
    "作为一个ai", "作为ai", "我是人工智能", "我无法回答", "我不能回答", "我无法提供",
    "很抱歉，我", "抱歉，我无法", "我不是一个", "无法协助",
    "as an ai", "i cannot", "i can't", "i'm sorry, but i cannot", "i am not able to",
)


def repeat_ngram_ratio(text: str, n: int = 8) -> float:
    """重复 n-gram 占比：小模型最常见的崩坏方式是陷入循环复读。

    用字符 n-gram 而不是词 n-gram，中文没有空格分词，字符级更稳。
    返回 0~1，越高越像复读机。
    """
    cleaned = re.sub(r"\s+", "", text or "")
    if len(cleaned) < n * 2:
        return 0.0
    grams = [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def contains_refusal(text: str) -> bool:
    """是否出现拒答话术。"""
    lowered = (text or "").lower()
    return any(pattern in lowered for pattern in _REFUSAL_PATTERNS)


def is_valid_json(text: str) -> bool:
    """要求严格 JSON 时，输出必须能被解析。"""
    stripped = re.sub(r"^\s*```(?:json)?|```\s*$", "", (text or "").strip(), flags=re.MULTILINE).strip()
    try:
        json.loads(stripped)
    except (TypeError, ValueError):
        return False
    return True


def check_output(
    text: str,
    *,
    require_json: bool = False,
    min_output_tokens: int = 4,
    max_repeat_ratio: float = 0.35,
    min_confidence: float = 0.0,
    confidence: float = 1.0,
    mode: str = "full",
) -> dict[str, Any]:
    """对一次输出做体检。

    :param require_json: 是否要求严格 JSON
    :param min_output_tokens: 过短判定阈值（估算 token）
    :param max_repeat_ratio: 重复 n-gram 占比上限
    :param min_confidence: 路由自评置信度下限，低于它说明当初的判定本身就勉强
    :param confidence: 本次请求的自评置信度
    :param mode: ``full`` 完整检查；``prefix`` 流式场景的前缀检查（只做拒答与复读检查，
                 因为长度和 JSON 完整性在流还没结束时根本无从判断）
    :return: {passed, mode, reasons, repeat_ratio, output_tokens}
    """
    reasons: list[str] = []
    if mode not in {"full", "prefix"}:
        raise ValueError(f"不支持的自检模式: {mode}")

    if not text or not text.strip():
        return {
            "passed": False,
            "mode": mode,
            "reasons": ["输出为空"],
            "repeat_ratio": 0.0,
            "output_tokens": 0,
        }

    # 逐字符扫描的拒答/复读检查在任何模式下都成立
    if contains_refusal(text):
        reasons.append("命中拒答话术")

    ratio = repeat_ngram_ratio(text)
    if ratio > max_repeat_ratio:
        reasons.append(f"重复 n-gram 占比 {ratio:.2f} 超过 {max_repeat_ratio}")

    output_tokens = estimate_tokens(text)

    if mode == "full":
        # 长度与格式完整性只在拿到完整输出后才检查
        if output_tokens < min_output_tokens:
            reasons.append(f"输出过短（约 {output_tokens} token < {min_output_tokens}）")
        if require_json and not is_valid_json(text):
            reasons.append("要求严格 JSON 但输出不可解析")
    elif require_json and not text.lstrip().startswith(("{" "[")):
        # 前缀模式下退而求其次：开头连 JSON 的左括号都没有，基本可以判死刑
        reasons.append("要求严格 JSON 但输出前缀不是 JSON")

    if confidence < min_confidence:
        reasons.append(f"自评置信度 {confidence:.2f} 低于下限 {min_confidence:.2f}")

    return {
        "passed": not reasons,
        "mode": mode,
        "reasons": reasons,
        "repeat_ratio": round(ratio, 4),
        "output_tokens": output_tokens,
    }
