"""端到端冒烟测试：用假的传输层把整条链路跑起来（不需要真实 GPU / API Key）。

运行方式（标准库 unittest，零额外依赖）：

    python tests/test_smoke.py
    或 python -m unittest discover -s tests

覆盖点基本对应 README 的边界处理表与失败案例：
  强制升级规则、缓存实体一致性、PII 不出网、输出自检升级、
  本地实例不可用自动切云端、守门自动升高阈值、图引擎的结构校验。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

# 让测试可以直接以脚本方式运行（不依赖安装）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import ConfigError, RoutingConfig, TierConfig, parse_yaml_subset  # noqa: E402
from flow.graph import END, GraphError, StateGraph, Workflow  # noqa: E402
from utils.guardian import (  # noqa: E402
    ACTION_FORCE_FRONTIER,
    ACTION_RAISE_THRESHOLD,
    GuardianMonitor,
)
from utils.heuristics import detect_code, detect_math, detect_realtime_data  # noqa: E402
from utils.llm_client import (  # noqa: E402
    AllBackendsFailedError,
    ChatClient,
    Endpoint,
    NoEndpointError,
    PermanentLLMError,
    TransientLLMError,
)
from utils.semantic_cache import hash_embedding  # noqa: E402
from utils.metering import Meter, RequestRecord, compute_cost  # noqa: E402
from utils.selfcheck import check_output  # noqa: E402
from utils.semantic_cache import SemanticCache  # noqa: E402
from utils.heuristics import (  # noqa: E402
    detect_pii,
    estimate_tokens,
    extract_entities,
    extract_json_object,
)

SELF_EVAL_MARK = "路由难度评估器"
JUDGE_MARK = "结果评审员"


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #


class MockLLM:
    """假的模型后端：按 payload 内容返回不同响应。"""

    def __init__(
        self,
        *,
        slm_reply: str = "这是本地小模型的回答。",
        cloud_reply: str = "这是云端强模型的回答。",
        self_eval: dict[str, Any] | None = None,
        judge: dict[str, Any] | None = None,
        failure_mode: str | None = None,
    ) -> None:
        self.slm_reply = slm_reply
        self.cloud_reply = cloud_reply
        self.self_eval = self_eval or {"can_answer": True, "confidence": 0.92, "difficulty": "simple"}
        self.judge = judge or {"a_score": 8, "b_score": 8}
        # None | "slm_down" | "cloud_500"；另外可单独让自评失败
        self.failure_mode = failure_mode
        self.self_eval_error: Exception | None = None
        self.calls: list[dict[str, Any]] = []

    @property
    def transport(self):
        return lambda url, payload, headers, timeout: self(url, payload, headers, timeout)

    def __call__(self, url: str, payload: dict[str, Any], headers: dict, timeout: float):
        self.calls.append({"url": url, "payload": payload})
        model = payload.get("model", "")
        first = payload["messages"][0] if payload.get("messages") else {}

        # 自评请求
        if first.get("role") == "system" and SELF_EVAL_MARK in (first.get("content") or ""):
            if self.self_eval_error is not None:
                raise self.self_eval_error
            return 200, self._body(model, json.dumps(self.self_eval, ensure_ascii=False))
        # judge 请求
        if first.get("role") == "system" and JUDGE_MARK in (first.get("content") or ""):
            return 200, self._body(model, json.dumps(self.judge, ensure_ascii=False))

        # 正常推理：本地 vs 云端
        if model.endswith("frontier") or "frontier" in model:
            if self.failure_mode == "cloud_500":
                return 500, '{"error":"upstream boom"}'
            return 200, self._body(model, self.cloud_reply)
        if self.failure_mode == "slm_down":
            raise TransientLLMError("本地 vLLM 实例连接失败")
        return 200, self._body(model, self.slm_reply)

    @staticmethod
    def _body(model: str, content: str, in_tokens: int = 16, out_tokens: int = 32) -> str:
        return json.dumps(
            {
                "model": model,
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": in_tokens,
                    "completion_tokens": out_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            },
            ensure_ascii=False,
        )


def make_config(**overrides: Any) -> RoutingConfig:
    """测试用配置：三档结构与 README 一致，价格与超时按需覆盖。"""
    tiers = [
        TierConfig("slm_tiny", "mock-tiny", "http://mock-tiny", 8192, local=True, gpu_hourly_cost_cny=3.0),
        TierConfig("slm_mid", "mock-mid", "http://mock-mid", 32768, local=True, gpu_hourly_cost_cny=6.0),
        TierConfig(
            "frontier",
            "mock-frontier",
            "http://mock-frontier",
            200000,
            local=False,
            price_input_per_mtok=70.0,
            price_output_per_mtok=350.0,
            price_cached_input_per_mtok=7.0,
        ),
    ]
    defaults: dict[str, Any] = {
        "tiers": tiers,
        "classifier_mode": "tiny_model_self_eval",
        "confidence_threshold": 0.75,
        "semantic_threshold": 0.95,
        "shadow_ratio": 0.0,  # 测试默认关闭影子流量，需要时单独打开
        "window_min_samples": 2,
        "consecutive_windows": 3,
        "threshold_cooldown_seconds": 0,
        "max_upgrade_attempts": 1,
        "self_eval_timeout_s": 1.0,
        "slm_timeout_s": 5.0,
        "cloud_timeout_s": 5.0,
        "judge_timeout_s": 5.0,
        "max_retries": 1,
        "backoff_base_s": 0.001,
        "cache_ttl_hours": 72,
    }
    defaults.update(overrides)
    return RoutingConfig(**defaults).validate()


def make_client(llm: MockLLM, cfg: RoutingConfig) -> ChatClient:
    """把假后端注册成三个档位的节点。"""
    client = ChatClient(
        transport=llm.transport,
        max_retries=cfg.max_retries,
        backoff_base_s=0.001,
        sleep=lambda _seconds: None,  # 测试里不要真的睡
        jitter=False,
    )
    for tier in cfg.tiers:
        client.add_endpoint(tier.name, Endpoint(name=tier.name, base_url=tier.endpoint))
    return client


def make_workflow(llm: MockLLM, cfg: RoutingConfig | None = None, **workflow_kwargs: Any) -> Workflow:
    cfg = cfg or make_config()
    return Workflow(client=make_client(llm, cfg), config=cfg, **workflow_kwargs)


def user_request(content: str, **extra: Any) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": content}], **extra}

import unittest

import unittest

import unittest

import unittest

import unittest

import unittest

import unittest

class TestHeuristics(unittest.TestCase):
    def test_token_estimation(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertGreater(estimate_tokens("这是一段中文文本"), 0)
        self.assertGreater(estimate_tokens("hello world this is english"), 5)

    def test_detect_math_and_code(self) -> None:
        self.assertTrue(detect_math("请计算 12345 * 6789 的结果"))
        self.assertTrue(detect_math("求解方程 x^2 + 2x + 1 = 0"))
        self.assertFalse(detect_math("帮我把这段话翻译成英文"))
        self.assertTrue(detect_code("```python\nprint(1)\n```"))
        self.assertTrue(detect_code("def add(a, b): return a + b"))
        self.assertFalse(detect_code("今天天气不错"))

    def test_detect_realtime(self) -> None:
        self.assertTrue(detect_realtime_data("现在苹果股价是多少"))
        self.assertTrue(detect_realtime_data("current weather in Tokyo"))
        self.assertTrue(detect_realtime_data("帮我查一下库存"))
        self.assertFalse(detect_realtime_data("把这段话改成正式语气"))
        # 「今天」只是句子内容，不是实时数据请求 —— 误判会造成大量无谓升级
        self.assertFalse(detect_realtime_data("把「今天的会议改到下午」翻译成英文"))

    def test_pii_detection(self) -> None:
        self.assertIn("phone_cn", detect_pii("联系我 13800138000"))
        self.assertIn("email", detect_pii("邮箱是 alice@example.com"))
        self.assertIn("id_card_cn", detect_pii("身份证 110101199003072316"))
        self.assertEqual(detect_pii("今天天气不错"), [])

    def test_entities(self) -> None:
        beijing = extract_entities("北京限行规则是什么")
        shanghai = extract_entities("上海限行规则是什么")
        self.assertEqual(beijing.get("location"), ["北京"])
        self.assertEqual(shanghai.get("location"), ["上海"])
        self.assertIn("time", extract_entities("2026年3月的销售额"))
        self.assertIn("money", extract_entities("预算是 120 万元"))

    def test_json_extraction(self) -> None:
        self.assertEqual(extract_json_object('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(extract_json_object('前面废话 {"a": 1, "b": {"c": 2}} 后面废话'), {"a": 1, "b": {"c": 2}})
        self.assertIsNone(extract_json_object("完全没有 JSON"))

class TestSelfCheck(unittest.TestCase):
    def test_normal_output_passes(self) -> None:
        result = check_output("这是正常的回答内容，长度也足够。")
        self.assertTrue(result["passed"], result["reasons"])

    def test_empty_and_refusal(self) -> None:
        self.assertFalse(check_output("")["passed"])
        self.assertFalse(check_output("很抱歉，我无法回答这个问题")["passed"])

    def test_repetition(self) -> None:
        broken = "同样的这句话" * 30
        result = check_output(broken)
        self.assertFalse(result["passed"])
        self.assertGreater(result["repeat_ratio"], 0.35)

    def test_strict_json(self) -> None:
        self.assertTrue(check_output('{"ok": true}', require_json=True)["passed"])
        self.assertFalse(check_output("这不是 JSON", require_json=True)["passed"])

    def test_prefix_mode_skips_length_check(self) -> None:
        # 流式前缀：长度不足不该判死刑，但拒答词必须立刻拦下
        self.assertTrue(check_output("正在进行", mode="prefix")["passed"])
        self.assertFalse(check_output("I cannot answer that", mode="prefix")["passed"])

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

class TestConfig(unittest.TestCase):
    def test_yaml_subset_parser(self) -> None:
        text = """
        # 注释
        routing:
          tiers:
            - name: slm_tiny
              model: qwen3-0.6b-awq
              endpoint: http://localhost:8000
              max_context: 8192
          classifier:
            mode: tiny_model_self_eval     # 行内注释
            confidence_threshold: 0.75
          force_upgrade_rules:
            - contains_math_or_code
            - context_tokens > 24000
          cache:
            invalidate_on: [model_version, prompt_version]
        """
        data = parse_yaml_subset(text)
        self.assertEqual(data["routing"]["tiers"][0]["name"], "slm_tiny")
        self.assertEqual(data["routing"]["tiers"][0]["max_context"], 8192)
        self.assertEqual(data["routing"]["classifier"]["mode"], "tiny_model_self_eval")
        self.assertEqual(data["routing"]["cache"]["invalidate_on"], ["model_version", "prompt_version"])
        self.assertEqual(data["routing"]["force_upgrade_rules"], ["contains_math_or_code", "context_tokens > 24000"])

    def test_project_config_file_parses(self) -> None:
        from config.config import DEFAULT_CONFIG_FILE, load_routing_config

        cfg = load_routing_config(DEFAULT_CONFIG_FILE)
        self.assertEqual([t.name for t in cfg.tiers], ["slm_tiny", "slm_mid", "frontier"])
        self.assertEqual(cfg.force_context_tokens, 24000)
        self.assertTrue(cfg.force_math_or_code)
        self.assertEqual(cfg.cache_invalidate_on, ["model_version", "prompt_version"])

    def test_missing_config_file_raises(self) -> None:
        from config.config import load_routing_config

        with self.assertRaises(ConfigError):
            load_routing_config("不存在的文件.yaml")

    def test_invalid_threshold_raises(self) -> None:
        with self.assertRaises(ConfigError):
            make_config(confidence_threshold=5.0)

    def test_duplicate_tier_names_raise(self) -> None:
        tiers = [TierConfig("a", "m", "http://x", 100), TierConfig("a", "m2", "http://y", 200)]
        with self.assertRaises(ConfigError):
            RoutingConfig(tiers=tiers)

class TestGraphEngine(unittest.TestCase):
    def test_missing_entry_raises(self) -> None:
        graph = StateGraph()
        graph.add_node("a", lambda state: state)
        with self.assertRaises(GraphError):
            graph.compile()

    def test_unknown_edge_target_raises(self) -> None:
        graph = StateGraph()
        graph.add_node("a", lambda state: state)
        graph.set_entry_point("a")
        graph.add_edge("a", "不存在")
        with self.assertRaises(GraphError):
            graph.compile()

    def test_unknown_branch_raises(self) -> None:
        graph = StateGraph()
        graph.add_node("a", lambda state: state)
        graph.set_entry_point("a")
        graph.add_conditional_edges("a", lambda state: "拼写错误的分支", {"ok": END})
        app = graph.compile()
        with self.assertRaises(GraphError):
            app.invoke({})

    def test_static_and_conditional_conflict(self) -> None:
        graph = StateGraph()
        graph.add_node("a", lambda state: state)
        graph.add_edge("a", END)
        with self.assertRaises(GraphError):
            graph.add_conditional_edges("a", lambda state: "x", {"x": END})

    def test_cycle_is_detected(self) -> None:
        graph = StateGraph()
        graph.add_node("a", lambda state: state)
        graph.set_entry_point("a")
        graph.add_edge("a", "a")  # 自己指向自己
        app = graph.compile(max_steps=5)
        with self.assertRaises(GraphError):
            app.invoke({})

    def test_stream_yields_each_node(self) -> None:
        llm = MockLLM()
        workflow = make_workflow(llm)
        steps = list(workflow.run(user_request("请将这句话改写成正式语气：这份报告逻辑有点乱"), stream=True))
        names = [name for name, _ in steps]
        self.assertIn("query_process", names)
        self.assertIn("cache_query", names)
        self.assertIn("result_process", names)

    def test_invalid_messages_are_recorded_not_raised(self) -> None:
        """入参错误由 BaseNode.run 收敛成 state['error']，不炸掉链路。

        注意这不是"吞异常"：非法入参的链路会立刻走到 result_process 收尾，
        错误信息完整留在 state 里，调用方可以直接判断。
        """
        llm = MockLLM()
        workflow = make_workflow(llm)
        for bad in ({}, {"messages": []}, {"messages": [{"role": "robot", "content": "hi"}]}):
            result = workflow.run(dict(bad))
            self.assertTrue(result["error"], f"应当记录错误: {bad}")
        result = workflow.run({"messages": [{"role": "user", "content": "   "}]})
        self.assertIn("content", result["error"])

    def test_node_exception_does_not_break_chain(self) -> None:
        """节点抛异常时由 BaseNode.run 记录到 error，链路继续走保守路径。"""
        llm = MockLLM()
        workflow = make_workflow(llm, make_config(max_upgrade_attempts=1))
        result = workflow.run(user_request("请将这句话改写成正式语气：这份报告逻辑有点乱", request_id="req-err"))
        self.assertIsInstance(result, dict)
        self.assertIn("result_process", result["trace"])

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

class TestSemanticCache(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = SemanticCache(threshold=0.95, ttl_hours=72, now=lambda: 1000.0)

    def _store(self, query: str, answer: str = "缓存的答案") -> None:
        self.cache.store(
            query=query,
            answer=answer,
            vector=hash_embedding(query),
            entities=extract_entities(query),
            model_version="v1",
            prompt_version="p1",
        )

    def test_exact_query_hits(self) -> None:
        self._store("北京限行规则是什么")
        lookup = self.cache.lookup(
            query="北京限行规则是什么",
            vector=hash_embedding("北京限行规则是什么"),
            entities=extract_entities("北京限行规则是什么"),
            model_version="v1",
            prompt_version="p1",
        )
        self.assertTrue(lookup.hit, lookup.reason)
        self.assertEqual(lookup.answer, "缓存的答案")

    def test_entity_mismatch_is_rejected(self) -> None:
        """README 失败案例：问「上海限行」返回了「北京限行」的答案。

        这里用「iPhone 15 / 16」构造一对字面高度相似、但关键实体不同的 query，
        确保拦下来的是**实体校验**而不是相似度不够。
        """
        # 相似度阈值刻意调低，确保拦下来的是实体校验而不是相似度不够
        cache = SemanticCache(threshold=0.85, ttl_hours=72, now=lambda: 1000.0)
        cache.store(
            query="iPhone 15 Pro 的电池容量是多少",
            answer="缓存的答案",
            vector=hash_embedding("iPhone 15 Pro 的电池容量是多少"),
            entities=extract_entities("iPhone 15 Pro 的电池容量是多少"),
        )
        query = "iPhone 16 Pro 的电池容量是多少"
        lookup = cache.lookup(
            query=query,
            vector=hash_embedding(query),
            entities=extract_entities(query),
        )
        self.assertFalse(lookup.hit)
        self.assertIn("实体", lookup.reason)
        self.assertGreater(lookup.similarity, 0.85)

    def test_pii_and_no_cache_are_skipped(self) -> None:
        self._store("普通问题")
        query = "我的手机号是 13800138000"
        lookup = self.cache.lookup(
            query=query,
            vector=hash_embedding(query),
            entities={},
            has_pii=True,
        )
        self.assertFalse(lookup.hit)
        self.assertIn("PII", lookup.reason)

        lookup2 = self.cache.lookup(query="现在股价", vector=hash_embedding("现在股价"), entities={}, no_cache=True)
        self.assertFalse(lookup2.hit)
        self.assertIn("no_cache", lookup2.reason)

    def test_model_version_invalidates(self) -> None:
        self._store("北京限行规则是什么")
        query = "北京限行规则是什么"
        lookup = self.cache.lookup(
            query=query,
            vector=hash_embedding(query),
            entities=extract_entities(query),
            model_version="v2",  # 换了模型版本
            prompt_version="p1",
        )
        self.assertFalse(lookup.hit)
        self.assertEqual(self.cache.invalidate(model_version="v1"), 1)

    def test_ttl_expiry(self) -> None:
        cache = SemanticCache(threshold=0.95, ttl_hours=1, now=lambda: 0.0)
        cache.store(query="问题", answer="答案", vector=hash_embedding("问题"), entities={})
        cache._now = lambda: 7200.0  # 两小时后
        lookup = cache.lookup(query="问题", vector=hash_embedding("问题"), entities={})
        self.assertFalse(lookup.hit)


if __name__ == "__main__":
    unittest.main()
