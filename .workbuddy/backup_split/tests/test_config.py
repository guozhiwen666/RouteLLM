"""配置模型与 YAML 子集解析器的测试。

共享替身（MockLLM / make_config / make_workflow）在 tests/helpers.py。
运行：cd 项目根 && python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from helpers import *  # noqa: F401,F403 - 测试替身与工具函数

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
