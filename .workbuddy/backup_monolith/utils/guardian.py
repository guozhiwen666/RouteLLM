"""质量守门：影子流量 + LLM-as-judge + 自动降级。

README 把这块称为项目的生命线 —— 没有守门的成本优化就是赌博：
成本省下来了、质量慢慢塌了，而用户不会投诉，只会流失。

闭环逻辑：
  1. 5% 请求在后台双跑强模型，结果与 SLM 结果一起交给 judge 打分；
  2. 逐条质量差汇总成窗口均值；
  3. 连续 N 个窗口质量差超过 2pp → 自动升高路由阈值（更多流量走强模型）；
  4. 单窗口质量差极其离谱 → 一键全量切强模型；
  5. 阈值调整带**变化率限制**和**冷却期**，避免来回震荡。

注意：文档只定义了「升高」和「维持」，没有自动降低。这里严格遵守 ——
阈值降回去属于人工决策（确认质量稳定后手动 reset），自动降低会让系统在
「省钱」和「保质量」之间反复横跳。
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 动作常量
ACTION_MAINTAIN = "maintain"
ACTION_RAISE_THRESHOLD = "raise_threshold"
ACTION_FORCE_FRONTIER = "force_frontier"


@dataclass
class GuardianAlert:
    """一条守门告警。"""

    timestamp: float
    level: str  # info | warning | critical
    action: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "action": self.action,
            "message": self.message,
        }


@dataclass
class GuardianWindow:
    """一个统计窗口的聚合结果。"""

    index: int
    samples: int
    mean_gap: float
    action: str
    closed_at: float


class GuardianMonitor:
    """守门监控器：抽样决策 + 窗口聚合 + 阈值调整。"""

    def __init__(
        self,
        *,
        shadow_ratio: float = 0.05,
        max_quality_drop: float = 0.02,
        consecutive_windows: int = 3,
        auto_rollback: bool = True,
        threshold: float = 0.75,
        threshold_max_step: float = 0.05,
        threshold_min: float = 0.5,
        threshold_max: float = 0.95,
        threshold_cooldown_seconds: int = 300,
        window_min_samples: int = 20,
        extreme_ratio: float = 5.0,
        now: Callable[[], float] = time.time,
    ) -> None:
        if not 0.0 <= shadow_ratio <= 1.0:
            raise ValueError("shadow_ratio 必须落在 [0, 1]")
        if not 0.0 <= max_quality_drop <= 1.0:
            raise ValueError("max_quality_drop 必须落在 [0, 1]")
        if consecutive_windows < 1:
            raise ValueError("consecutive_windows 必须 >= 1")
        if not threshold_min <= threshold <= threshold_max:
            raise ValueError("初始阈值必须落在 [threshold_min, threshold_max] 之间")

        self.shadow_ratio = shadow_ratio
        self.max_quality_drop = max_quality_drop
        self.consecutive_windows = consecutive_windows
        self.auto_rollback = auto_rollback
        self.threshold = threshold
        self.initial_threshold = threshold
        self.threshold_max_step = threshold_max_step
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.threshold_cooldown_seconds = threshold_cooldown_seconds
        self.window_min_samples = max(1, window_min_samples)
        self.extreme_ratio = extreme_ratio
        self._now = now

        self._lock = threading.RLock()
        self._samples: list[float] = []
        self._consecutive_bad = 0
        self._window_index = 0
        self._last_adjust_ts = 0.0
        self.windows: list[GuardianWindow] = []
        self.alerts: list[GuardianAlert] = []
        self.force_all_frontier = False

    # ------------------------------------------------------------------ #
    # 抽样
    # ------------------------------------------------------------------ #

    def should_shadow(self, request_id: str) -> bool:
        """是否把该请求纳入影子流量。

        用 request_id 的稳定哈希抽样：同一个请求重试/重放时结论一致，
        不会因为随机数导致同一条样本被反复评分。
        """
        if self.shadow_ratio <= 0:
            return False
        if self.shadow_ratio >= 1:
            return True
        digest = hashlib.sha1(request_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10000
        return bucket < int(self.shadow_ratio * 10000)

    # ------------------------------------------------------------------ #
    # 样本聚合
    # ------------------------------------------------------------------ #

    def add_sample(self, gap: float) -> str:
        """写入一条质量差样本（正值表示 SLM 比强模型差），返回触发的动作。

        窗口未满时只累计；窗口满时结算一次并判断是否降级。
        """
        if gap != gap:  # NaN 检查，judge 解析失败时可能产生
            logger.warning("忽略非法质量差样本: %r", gap)
            return ACTION_MAINTAIN

        with self._lock:
            self._samples.append(gap)
            if len(self._samples) < self.window_min_samples:
                return ACTION_MAINTAIN
            return self._close_window()

    def _close_window(self) -> str:
        """结算当前窗口。调用方需持有锁。"""
        samples = self._samples
        self._samples = []
        self._window_index += 1
        mean_gap = sum(samples) / len(samples)
        now = self._now()

        action = ACTION_MAINTAIN
        # 极端情况：质量差已经不是"下降"，而是"崩了"，直接全量切强模型
        if mean_gap > self.max_quality_drop * self.extreme_ratio:
            self.force_all_frontier = True
            action = ACTION_FORCE_FRONTIER
            self._alert(
                "critical",
                action,
                f"单窗口质量差 {mean_gap:.4f} 超过阈值的 {self.extreme_ratio} 倍，"
                f"已全量切换到强模型（需人工确认后恢复）",
            )
        elif mean_gap > self.max_quality_drop:
            self._consecutive_bad += 1
            if (
                self._consecutive_bad >= self.consecutive_windows
                and self.auto_rollback
                and (now - self._last_adjust_ts) >= self.threshold_cooldown_seconds
            ):
                action = self._raise_threshold(mean_gap, now)
            elif self._consecutive_bad >= self.consecutive_windows:
                action = ACTION_MAINTAIN
                logger.info(
                    "质量差连续 %d 个窗口超标，但处于冷却期内（距上次调整 %.0fs），暂不调整阈值",
                    self._consecutive_bad,
                    now - self._last_adjust_ts,
                )
        else:
            # 质量达标，连续计数清零（文档没有"自动降低阈值"，这里只复位计数）
            self._consecutive_bad = 0

        window = GuardianWindow(
            index=self._window_index,
            samples=len(samples),
            mean_gap=mean_gap,
            action=action,
            closed_at=now,
        )
        self.windows.append(window)
        logger.info(
            "守门窗口 #%d 结算：样本 %d 条，平均质量差 %.4f（阈值 %.4f），动作 %s",
            window.index,
            window.samples,
            mean_gap,
            self.max_quality_drop,
            action,
        )
        return action

    def _raise_threshold(self, mean_gap: float, now: float) -> str:
        """升高路由阈值：更多请求会因为置信度不够而升级到强模型。

        变化率限制（每次最多 threshold_max_step）+ 冷却期，避免系统震荡。
        """
        old = self.threshold
        new = min(self.threshold_max, round(self.threshold + self.threshold_max_step, 4))
        if new <= old:
            self._alert(
                "warning",
                ACTION_MAINTAIN,
                f"阈值已达上限 {self.threshold_max}，无法继续升高；当前质量差 {mean_gap:.4f}，需人工介入",
            )
            return ACTION_MAINTAIN

        self.threshold = new
        self._last_adjust_ts = now
        self._consecutive_bad = 0  # 调整后重新计数，避免连续触发
        self._alert(
            "warning",
            ACTION_RAISE_THRESHOLD,
            f"连续 {self.consecutive_windows} 个窗口质量差超标（本次 {mean_gap:.4f} > "
            f"{self.max_quality_drop}），路由阈值 {old:.2f} → {new:.2f}",
        )
        return ACTION_RAISE_THRESHOLD

    def _alert(self, level: str, action: str, message: str) -> None:
        alert = GuardianAlert(timestamp=self._now(), level=level, action=action, message=message)
        self.alerts.append(alert)
        logger.warning("[守门告警][%s] %s", level, message)

    # ------------------------------------------------------------------ #
    # 运维接口
    # ------------------------------------------------------------------ #

    def reset(self, *, clear_force_frontier: bool = True) -> None:
        """人工恢复：把阈值复位、解除全量强模型。

        这是文档里唯一的"降回去"路径 —— 属于人工决策，不自动执行。
        """
        with self._lock:
            self.threshold = self.initial_threshold
            self._consecutive_bad = 0
            self._samples = []
            self._last_adjust_ts = 0.0
            if clear_force_frontier:
                self.force_all_frontier = False
            logger.info("守门状态已人工复位，路由阈值回到 %.2f", self.threshold)

    def snapshot(self) -> dict[str, Any]:
        """当前守门状态，供监控面板展示。"""
        with self._lock:
            return {
                "threshold": self.threshold,
                "initial_threshold": self.initial_threshold,
                "shadow_ratio": self.shadow_ratio,
                "max_quality_drop": self.max_quality_drop,
                "consecutive_bad_windows": self._consecutive_bad,
                "current_window_samples": len(self._samples),
                "window_min_samples": self.window_min_samples,
                "windows_closed": len(self.windows),
                "force_all_frontier": self.force_all_frontier,
                "last_window": (
                    {
                        "index": self.windows[-1].index,
                        "samples": self.windows[-1].samples,
                        "mean_gap": round(self.windows[-1].mean_gap, 6),
                        "action": self.windows[-1].action,
                    }
                    if self.windows
                    else None
                ),
                "alerts": [alert.as_dict() for alert in self.alerts[-10:]],
            }
