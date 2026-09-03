"""LLM-as-judge：给两份回答打质量分的评审能力（影子流量守门用）。

judge 是守门闭环的度量工具：同一条问题，SLM 的答案 A 和强模型的答案 B
各打 0~10 分，质量差 = (B - A) / 10。正值表示 SLM 更差。

**脏样本纪律**：judge 输出无法解析时返回 ``None`` 结果，由调用方决定
不写入样本 —— 宁可少一个样本，也不要让守门被假信号骗了。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from utils.llm_client import ChatClient
from utils.text_utils import extract_json_object

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "你是严格的结果评审员。同一问题有两份回答，请分别就「事实正确性、完整性、指令遵循」"
    "打分，0~10 分（10 分为完美）。不要考虑文风与长度，只看内容质量。"
    "只输出严格 JSON，不要任何多余文字。"
)
JUDGE_TEMPLATE = (
    "【问题】\n{question}\n\n"
    "【回答 A】\n{answer_a}\n\n"
    "【回答 B】\n{answer_b}\n\n"
    '输出 JSON：{{"a_score": 0~10, "b_score": 0~10, "reason": "..."}}'
)


@dataclass
class JudgeResult:
    """一次 judge 打分的输出。"""

    gap: float | None  # (B - A)/10，正值表示 A（SLM）更差；None 表示打分失败
    a_score: float | None
    b_score: float | None
    shadow_answer: str = ""  # 双跑出的强模型答案（不返回给用户）
    error: str = ""  # 失败原因


def run_shadow_evaluation(
    client: ChatClient,
    *,
    tier: str,
    model: str,
    judge_tier: str,
    judge_model: str,
    question: str,
    answer_a: str,
    shadow_messages: list[dict[str, str]] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    judge_timeout_s: float = 20.0,
    cloud_timeout_s: float = 30.0,
) -> JudgeResult:
    """完整跑一遍影子对比：强模型双跑 → judge 打分。

    :param tier/model: 强模型所在的档位（与 judge 可不同）
    :param question: 用户问题（用于拼 judge 提示词）
    :param answer_a: SLM 实际返回给用户的答案
    :param shadow_messages: 原始对话消息（多轮时强模型要看到完整上下文）；
                            缺省则退化为只用 question 的单轮对话
    """
    try:
        messages = shadow_messages if shadow_messages else _user_messages(question)
        shadow_response = client.chat(
            tier,
            model,
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=cloud_timeout_s,
        )
        shadow_answer = shadow_response.content or ""
    except Exception as exc:  # noqa: BLE001 - 双跑失败不算守门结论
        return JudgeResult(gap=None, a_score=None, b_score=None, error=f"强模型双跑失败: {exc}")

    try:
        judge_response = client.chat(
            judge_tier,
            judge_model,
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": JUDGE_TEMPLATE.format(
                        question=question[:4000],
                        answer_a=answer_a[:4000],
                        answer_b=shadow_answer[:4000],
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=256,
            timeout=judge_timeout_s,
            response_format={"type": "json_object"},
        )
        parsed = extract_json_object(judge_response.content or "")
        if parsed is None:
            raise ValueError(f"judge 输出无法解析为 JSON: {(judge_response.content or '')[:200]!r}")

        a_score, b_score = _to_score(parsed.get("a_score")), _to_score(parsed.get("b_score"))
        if a_score is None or b_score is None:
            raise ValueError(f"judge 输出缺少有效分数: {parsed!r}")

        # 质量差：正值表示 A（SLM）比 B（强模型）差
        return JudgeResult(
            gap=(b_score - a_score) / 10.0,
            a_score=a_score,
            b_score=b_score,
            shadow_answer=shadow_answer,
        )
    except Exception as exc:  # noqa: BLE001 - 打分失败按「无样本」处理
        logger.warning("judge 打分失败，本条不写入样本: %s", exc)
        return JudgeResult(
            gap=None,
            a_score=None,
            b_score=None,
            shadow_answer=shadow_answer,
            error=f"judge 打分失败: {exc}",
        )


def _user_messages(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question}]


def _to_score(value: Any) -> float | None:
    """把 judge 给出的分数规整到 0~10。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if score != score:  # NaN
        return None
    return max(0.0, min(10.0, score))
