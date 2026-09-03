"""语义缓存（命中/实体不一致/PII/版本失效/TTL）的单元测试。

共享替身（MockLLM / make_config / make_workflow）在 tests/helpers.py。
运行：cd 项目根 && python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from helpers import *  # noqa: F401,F403 - 测试替身与工具函数

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


# --------------------------------------------------------------------------- #
# 路由链路
# --------------------------------------------------------------------------- #
