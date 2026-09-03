"""启发式规则与输出自检的单元测试。

共享替身（MockLLM / make_config / make_workflow）在 tests/helpers.py。
运行：cd 项目根 && python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from helpers import *  # noqa: F401,F403 - 测试替身与工具函数

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


# --------------------------------------------------------------------------- #
# 语义缓存
# --------------------------------------------------------------------------- #
