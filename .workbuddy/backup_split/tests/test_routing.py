"""路由链路端到端测试：强制升级、快筛、自评、自检升级、缓存集成。

共享替身（MockLLM / make_config / make_workflow）在 tests/helpers.py。
运行：cd 项目根 && python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from helpers import *  # noqa: F401,F403 - 测试替身与工具函数

class TestRouting(unittest.TestCase):
    def test_simple_request_goes_to_slm(self) -> None:
        llm = MockLLM()
        result = make_workflow(llm).run(
            user_request("请将这句话改写成正式语气：这份报告逻辑有点乱", request_id="req-simple")
        )
        self.assertEqual(result["route_tier"], "slm_tiny")
        self.assertEqual(result["final_tier"], "slm_tiny")
        self.assertEqual(result["final_output"], "这是本地小模型的回答。")
        self.assertEqual(result["trace"][:2], ["query_process", "cache_query"])

    def test_math_forces_upgrade(self) -> None:
        llm = MockLLM(self_eval={"can_answer": True, "confidence": 0.99, "difficulty": "simple"})
        result = make_workflow(llm).run(user_request("请计算 123456 × 789012 的结果", request_id="req-math"))
        # 自评再高也没用：数学命中强制升级规则
        self.assertEqual(result["route_tier"], "frontier")
        self.assertIn("contains_math_or_code", result["force_upgrade_hits"])
        self.assertEqual(result["final_output"], "这是云端强模型的回答。")

    def test_code_forces_upgrade(self) -> None:
        llm = MockLLM()
        result = make_workflow(llm).run(
            user_request("用 Python 写一个快排：def quicksort(arr): ...", request_id="req-code")
        )
        self.assertEqual(result["route_tier"], "frontier")
        self.assertIn("contains_math_or_code", result["force_upgrade_hits"])

    def test_low_confidence_upgrades(self) -> None:
        llm = MockLLM(self_eval={"can_answer": True, "confidence": 0.4, "difficulty": "hard"})
        result = make_workflow(llm).run(user_request("请分析一下这份财报的核心风险点", request_id="req-uncertain"))
        self.assertEqual(result["route_tier"], "frontier")
        self.assertLess(result["confidence"], 0.75)

    def test_self_eval_failure_is_fail_safe(self) -> None:
        """自评超时/异常 → 按"不确定"处理 → 保守升级（README 边界处理表）。"""
        llm = MockLLM()
        llm.self_eval_error = TransientLLMError("自评模型超时")
        result = make_workflow(llm).run(
            user_request("请介绍一下人形机器人行业的发展现状", request_id="req-timeout")
        )
        self.assertEqual(result["route_tier"], "frontier")
        self.assertTrue(any("自评调用失败" in w for w in result["warnings"]))

    def test_long_context_upgrades(self) -> None:
        llm = MockLLM()
        long_text = "这是一段很长的文档。" * 6000  # 约 36000 token，超出所有本地档位
        result = make_workflow(llm).run(user_request(long_text, request_id="req-long"))
        self.assertEqual(result["route_tier"], "frontier")
        self.assertTrue(any("context_tokens" in hit for hit in result["force_upgrade_hits"]))

    def test_strict_json_upgrades(self) -> None:
        llm = MockLLM()
        result = make_workflow(llm).run(
            user_request("提取人名和地名，只返回 JSON 格式", request_id="req-json")
        )
        self.assertEqual(result["route_tier"], "frontier")
        self.assertIn("output_format == strict_json", result["force_upgrade_hits"])


class TestOutputSelfCheckUpgrade(unittest.TestCase):
    def test_bad_slm_output_upgrades_to_cloud(self) -> None:
        llm = MockLLM(slm_reply="很抱歉，我无法回答这个问题")
        result = make_workflow(llm).run(user_request("请将这句话改写成正式语气：这份报告逻辑有点乱", request_id="req-bad"))
        self.assertEqual(result["final_tier"], "frontier")
        self.assertEqual(result["final_output"], "这是云端强模型的回答。")
        self.assertEqual(result["upgrade_count"], 1)
        self.assertIn("slm_inference", result["trace"])
        self.assertIn("cloud_inference", result["trace"])

    def test_local_instance_down_falls_back_to_cloud(self) -> None:
        """README 边界处理：本地 vLLM 实例挂了 → 自动切云端。"""
        llm = MockLLM(failure_mode="slm_down")
        workflow = make_workflow(llm)
        result = workflow.run(user_request("请将这句话改写成正式语气：这份报告逻辑有点乱", request_id="req-down"))
        self.assertEqual(result["final_tier"], "frontier")
        self.assertEqual(result["final_output"], "这是云端强模型的回答。")
        # 节点被摘除后应进入冷却期
        snapshot = workflow.client.health_snapshot()
        self.assertFalse(snapshot["slm_tiny"][0]["available"])

    def test_upgrade_limit_prevents_pingpong(self) -> None:
        """云端再失败也不能无限重试。"""
        llm = MockLLM(slm_reply="很抱歉，我无法回答这个问题", failure_mode="cloud_500")
        result = make_workflow(llm, make_config(max_retries=0)).run(
            user_request("请将这句话改写成正式语气：这份报告逻辑有点乱", request_id="req-allfail")
        )
        self.assertEqual(result["upgrade_count"], 1)
        self.assertTrue(result["error"])
        self.assertIn("cloud_inference", result["trace"])


class TestCacheIntegration(unittest.TestCase):
    def test_second_identical_request_hits_cache(self) -> None:
        llm = MockLLM()
        workflow = make_workflow(llm)
        request = user_request("请将这句话改写成正式语气：这份报告逻辑有点乱", request_id="req-cache")
        workflow.run(dict(request))
        second = workflow.run(dict(request))
        self.assertTrue(second["cache_hit"])
        self.assertEqual(second["route_stage"], "cache")
        self.assertEqual(second["final_tier"], "cache")
        self.assertEqual(second["cost_cny"], 0.0)
        self.assertGreater(second["usage"]["cached_input_tokens"], 0)

    def test_pii_request_never_written_to_cache(self) -> None:
        llm = MockLLM()
        workflow = make_workflow(llm)
        request = user_request("我的身份证是 110101199003072316，帮我写封请假信", request_id="req-pii")
        first = workflow.run(dict(request))
        self.assertTrue(first["has_pii"])
        self.assertEqual(workflow.cache.stats()["entries"], 0)
