"""守门机制测试：影子抽样、动态阈值升高、一键全量降级。

共享替身（MockLLM / make_config / make_workflow）在 tests/helpers.py。
运行：cd 项目根 && python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from helpers import *  # noqa: F401,F403 - 测试替身与工具函数

class TestShadowAndGuardian(unittest.TestCase):
    def test_shadow_sampling_is_stable(self) -> None:
        guardian = GuardianMonitor(shadow_ratio=0.5, window_min_samples=2)
        picked = guardian.should_shadow("fixed-request-id")
        # 同一个 request_id 的结论必须稳定，否则同一条样本会被反复评分
        self.assertEqual(picked, guardian.should_shadow("fixed-request-id"))

    def test_shadow_flow_runs_and_scores(self) -> None:
        llm = MockLLM(
            slm_reply="这是一个足够长的本地小模型回答，包含了完整的翻译内容与足够的文字。",
            cloud_reply="这是一个足够长的云端强模型回答，包含了完整的翻译内容与足够的文字。",
            judge={"a_score": 6, "b_score": 9},
        )
        workflow = make_workflow(
            llm,
            make_config(shadow_ratio=1.0, window_min_samples=1, consecutive_windows=1),
        )
        result = workflow.run(user_request("请将这句话改写成正式语气：这份报告逻辑有点乱", request_id="req-shadow"))
        self.assertTrue(result["shadow"])
        self.assertEqual(result["guardian_action"], "submitted")
        outcomes = workflow.comparative_scoring_node.wait_pending(timeout=5)
        self.assertEqual(len(outcomes), 1)
        self.assertAlmostEqual(outcomes[0]["gap"], 0.3)  # (9-6)/10
        workflow.shutdown()

    def test_guardian_raises_threshold_after_bad_windows(self) -> None:
        guardian = GuardianMonitor(
            max_quality_drop=0.02,
            consecutive_windows=3,
            threshold=0.75,
            threshold_max_step=0.05,
            threshold_cooldown_seconds=0,
            window_min_samples=2,
        )
        actions = []
        for _ in range(3):
            actions.append(guardian.add_sample(0.05))  # 每个窗口 2 条样本
            actions.append(guardian.add_sample(0.05))
        self.assertIn(ACTION_RAISE_THRESHOLD, actions)
        self.assertAlmostEqual(guardian.threshold, 0.80)

    def test_guardian_extreme_case_forces_frontier(self) -> None:
        guardian = GuardianMonitor(
            max_quality_drop=0.02, window_min_samples=1, consecutive_windows=3, extreme_ratio=5.0
        )
        action = guardian.add_sample(0.5)  # 50pp 的质量差，属于崩了
        self.assertEqual(action, ACTION_FORCE_FRONTIER)
        self.assertTrue(guardian.force_all_frontier)

    def test_force_frontier_overrides_routing(self) -> None:
        llm = MockLLM()
        workflow = make_workflow(llm)
        workflow.guardian.force_all_frontier = True
        result = workflow.run(user_request("请将这句话改写成正式语气：这份报告逻辑有点乱", request_id="req-forced"))
        self.assertEqual(result["route_tier"], "frontier")
        self.assertIn("全量强模型", result["route_reason"])


# --------------------------------------------------------------------------- #
# 图引擎
# --------------------------------------------------------------------------- #
