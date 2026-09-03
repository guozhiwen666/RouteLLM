"""模型客户端（Fallback/退避/流式解析）与计量（计价/Prometheus 导出）测试。

共享替身（MockLLM / make_config / make_workflow）在 tests/helpers.py。
运行：cd 项目根 && python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from helpers import *  # noqa: F401,F403 - 测试替身与工具函数

class TestChatClient(unittest.TestCase):
    def _client(self, responses: list[tuple[int, str]]) -> tuple[ChatClient, list[int]]:
        calls: list[int] = []

        def transport(url, payload, headers, timeout):
            calls.append(1)
            return responses[min(len(calls) - 1, len(responses) - 1)]

        client = ChatClient(transport=transport, max_retries=1, backoff_base_s=0.001, sleep=lambda s: None, jitter=False)
        client.add_endpoint("t", Endpoint(name="primary", base_url="http://a", api_keys=["k1"]))
        client.add_endpoint("t", Endpoint(name="backup", base_url="http://b", api_keys=["k2"]))
        return client, calls

    def test_fallback_to_second_endpoint(self) -> None:
        body = MockLLM._body("m", "ok")
        client, calls = self._client([(500, "boom"), (200, body)])
        response = client.chat("t", "m", [{"role": "user", "content": "hi"}])
        self.assertEqual(response.content, "ok")
        self.assertEqual(len(calls), 2)

    def test_all_backends_fail_raises(self) -> None:
        client, _ = self._client([(500, "boom")])
        with self.assertRaises(AllBackendsFailedError):
            client.chat("t", "m", [{"role": "user", "content": "hi"}])

    def test_permanent_error_is_not_retried(self) -> None:
        client, calls = self._client([(400, "bad request")])
        with self.assertRaises(PermanentLLMError):
            client.chat("t", "m", [{"role": "user", "content": "hi"}])
        self.assertEqual(len(calls), 1)

    def test_cached_tokens_are_parsed(self) -> None:
        body = json.dumps(
            {
                "model": "m",
                "choices": [{"message": {"content": "hi"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 60}},
            }
        )
        client, _ = self._client([(200, body)])
        response = client.chat("t", "m", [{"role": "user", "content": "hi"}])
        self.assertEqual(response.cached_input_tokens, 60)

    def test_missing_tier_raises_no_endpoint(self) -> None:
        client, _ = self._client([(200, MockLLM._body("m", "ok"))])
        with self.assertRaises(NoEndpointError):
            client.chat("不存在的档位", "m", [{"role": "user", "content": "hi"}])


class TestMetering(unittest.TestCase):
    def test_cloud_cost_by_token_type(self) -> None:
        tier = TierConfig("frontier", "m", "upstream", 1000, price_input_per_mtok=70.0, price_output_per_mtok=350.0, price_cached_input_per_mtok=7.0)
        cost = compute_cost(tier, {"input_tokens": 1_000_000, "output_tokens": 0, "cached_input_tokens": 0}, 1.0)
        self.assertAlmostEqual(cost, 70.0)
        cached_cost = compute_cost(
            tier, {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 1_000_000}, 1.0
        )
        self.assertAlmostEqual(cached_cost, 7.0)

    def test_local_cost_includes_gpu_amortization(self) -> None:
        tier = TierConfig("slm_tiny", "m", "http://local", 1000, gpu_hourly_cost_cny=3.0)
        # 占用 1 秒 → 3 元/小时 ÷ 3600
        self.assertAlmostEqual(compute_cost(tier, {}, 1.0), 3.0 / 3600)

    def test_prometheus_export(self) -> None:
        meter = Meter()
        meter.record(
            RequestRecord("r1", "slm_tiny", False, False, {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5}, 0.001, 12.5)
        )
        text = meter.render_prometheus()
        self.assertIn("routellm_requests_total", text)
        self.assertIn("routellm_cost_cny_total", text)
        self.assertIn("routellm_latency_ms", text)
        summary = meter.summary()
        self.assertEqual(summary["requests_total"], 1)
        self.assertAlmostEqual(summary["cost_cny_per_request"], 0.001, places=6)


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
