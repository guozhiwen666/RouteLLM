"""影子流量对比评分：LLM-as-judge 逐条打分 + 守门闭环。

链路位置：result_process → comparative_scoring → END

README 的守门流程：5% 请求在后台双跑强模型，用 judge 逐条对比，
**结果不返回给用户**，只作为质量信号进入守门监控器。

本节点只管「编排」：把样本丢到后台线程跑对比、把结果喂给守门监控器。
judge 打分逻辑内联在本节点文件（可单独测试、可换更强的评审模型）。

实现要点：
  - 双跑在**后台线程**执行（生产应换成独立 worker 或消息队列），主链路不等它；
  - judge 打分失败**不写入样本**（脏样本会让守门误判），连续失败打告警 ——
    守门自己瞎了是最危险的静默故障。
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from config.config import RoutingConfig, routing_config
from flow.base import BaseNode
from flow.state import GraphState
from utils.guardian import GuardianMonitor
from utils.llm_client import ChatClient
from utils.metering import Meter

logger = logging.getLogger(__name__)

# judge 连续失败多少次后开始告警（防止守门静默失效）
JUDGE_FAILURE_ALERT_THRESHOLD = 5




class ComparativeScoringNode(BaseNode):
    """影子流量双跑与 judge 评分节点（后台线程）。"""

    name = "comparative_scoring"

    def __init__(
        self,
        *,
        client: ChatClient | None = None,
        guardian: GuardianMonitor | None = None,
        meter: Meter | None = None,
        config: RoutingConfig | None = None,
        executor: ThreadPoolExecutor | None = None,
        max_workers: int = 2,
    ) -> None:
        super().__init__()
        self.client = client
        self.guardian = guardian
        self.meter = meter
        self.cfg = config or routing_config
        self._executor = executor
        self._owns_executor = executor is None
        self._max_workers = max(1, max_workers)
        self._lock = threading.Lock()
        self._pending: list[Future] = []
        self._judge_failures = 0

    # ------------------------------------------------------------------ #

    def process(self, state: GraphState) -> GraphState:
        new_state = dict(state)
        if not new_state.get("shadow"):
            return new_state  # 未被抽中，直接结束
        if self.guardian is None or self.client is None:
            new_state["guardian_action"] = "skipped"
            return new_state

        # 只把不可变的基本类型交给后台线程，避免与主线程共享 state 引发竞态
        snapshot = {
            "request_id": new_state.get("request_id") or "",
            "messages": [dict(m) for m in new_state.get("messages") or []],
            "question": "\n".join(
                str(m.get("content") or "")
                for m in new_state.get("messages") or []
                if m.get("role") == "user"
            ),
            "slm_answer": new_state.get("final_output") or "",
            "temperature": float(new_state.get("temperature", 0.7)),
            "max_tokens": int(new_state.get("max_tokens") or 1024),
        }
        future = self._get_executor().submit(self._evaluate, snapshot)
        with self._lock:
            self._pending.append(future)

        new_state["guardian_action"] = "submitted"
        return new_state

    # ------------------------------------------------------------------ #

    def _evaluate(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """后台执行：强模型双跑 → judge 对比 → 样本写入守门监控器。"""
        result: dict[str, Any] = {
            "request_id": snapshot.get("request_id", ""),
            "gap": None,
            "action": "skipped",
            "error": "",
        }
        try:
            frontier = self.cfg.frontier_tier()
            judge_tier_name = (
                self.cfg.judge_tier if self.cfg.judge_tier in {t.name for t in self.cfg.tiers} else frontier.name
            )
            judge_tier = self.cfg.tier(judge_tier_name)

            judge_result = run_shadow_evaluation(
                self.client,  # type: ignore[arg-type]
                tier=frontier.name,
                model=frontier.model,
                judge_tier=judge_tier.name,
                judge_model=judge_tier.model,
                question=snapshot.get("question") or "",
                answer_a=snapshot.get("slm_answer") or "",
                shadow_messages=snapshot.get("messages") or [],
                temperature=snapshot.get("temperature", 0.7),
                max_tokens=snapshot.get("max_tokens", 1024),
                judge_timeout_s=self.cfg.judge_timeout_s,
                cloud_timeout_s=self.cfg.cloud_timeout_s,
            )

            result["shadow_answer"] = judge_result.shadow_answer
            result["gap"] = judge_result.gap
            result["a_score"] = judge_result.a_score
            result["b_score"] = judge_result.b_score

            if judge_result.gap is None:
                raise ValueError(judge_result.error or "影子评分失败（无有效样本）")

            # 写入守门监控器（窗口聚合 + 自动降级在这里触发）
            action = self.guardian.add_sample(judge_result.gap)  # type: ignore[union-attr]
            result["action"] = action
            if self.meter is not None:
                self.meter.observe_quality_gap(judge_result.gap)

            with self._lock:
                self._judge_failures = 0
            return result
        except Exception as exc:  # noqa: BLE001 - 守门任务失败不能影响主链路
            result["error"] = f"{type(exc).__name__}: {exc}"
            with self._lock:
                self._judge_failures += 1
                failures = self._judge_failures
            logger.warning("影子流量评分失败（%s）：%s", snapshot.get("request_id"), exc)
            if failures >= JUDGE_FAILURE_ALERT_THRESHOLD:
                logger.error(
                    "judge 已连续失败 %d 次，守门机制正在失去监控能力，请立即排查", failures
                )
            return result

    # ------------------------------------------------------------------ #
    # 运维接口
    # ------------------------------------------------------------------ #

    def _get_executor(self) -> ThreadPoolExecutor:
        """懒加载线程池：没有影子流量就一个线程都不起。"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="routellm-shadow"
            )
            self._owns_executor = True
        return self._executor

    def wait_pending(self, timeout: float | None = None) -> list[dict[str, Any]]:
        """等待所有已提交的影子任务完成（测试与优雅退出时使用）。

        :param timeout: 单个任务的超时秒数；None 表示一直等
        """
        with self._lock:
            pending = list(self._pending)
        results: list[dict[str, Any]] = []
        for future in pending:
            try:
                results.append(future.result(timeout=timeout))
            except Exception as exc:  # noqa: BLE001 - 只负责收集，不抛给调用方
                results.append({"gap": None, "action": "error", "error": str(exc)})
        with self._lock:
            self._pending = [future for future in self._pending if not future.done()]
        return results

    def shutdown(self, wait: bool = True) -> None:
        """关闭线程池（仅关闭自己创建的那个）。"""
        if self._executor is not None and self._owns_executor:
            self._executor.shutdown(wait=wait)
            self._executor = None


# ================ judge 影子评分能力（原 utils/judge.py，回滚后内联） ================

import logging
from dataclasses import dataclass, field
from typing import Any

from utils.llm_client import ChatClient
from utils.heuristics import extract_json_object

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
