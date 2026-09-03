"""影子流量对比评分：LLM-as-judge 逐条打分 + 守门闭环。

链路位置：result_process → comparative_scoring → END

README 的守门流程：5% 请求在后台双跑强模型，用 judge 逐条对比，
**结果不返回给用户**，只作为质量信号进入守门监控器。

本节点只管「编排」：把样本丢到后台线程跑对比、把结果喂给守门监控器。
judge 打分能力本身在 `utils/judge.py`（可单独测试、可换更强的评审模型）。

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
from utils.judge import run_shadow_evaluation
from utils.llm_client import ChatClient
from utils.metering import Meter

logger = logging.getLogger(__name__)

# judge 连续失败多少次后开始告警（防止守门静默失效）
JUDGE_FAILURE_ALERT_THRESHOLD = 5


class ScoringState(GraphState):
    """守门阶段独有的状态字段。"""

    shadow: bool  # 本条请求是否被抽中为影子流量
    shadow_output: str  # 影子双跑的强模型结果（不返回给用户）
    judge_score: float  # judge 给出的 SLM 相对强模型的质量差（正=SLM 更差）
    quality_gap: float
    guardian_action: str  # submitted | maintain | raise_threshold | force_frontier


class ComparativeScoringNode(BaseNode):
    """影子流量双跑与 judge 评分节点（后台线程）。"""

    name = "comparative_scoring"
    state_schema = ScoringState

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
