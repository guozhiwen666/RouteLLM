"""语义缓存：收益最大、也最危险的一招。

README 的四条铁律在这里落地：相似度 ≥0.95 才可能命中；命中后二次校验
（关键实体完全一致 + 长度比合理）；实时数据不缓存、PII 不查不写；
按「模型版本 + prompt 版本 + 时间」三维失效。

拆分：向量能力在 utils/embedding.py，实体/长度校验在 utils/text_utils.py，
拒答词在 utils/selfcheck.py，本模块只管存储结构与四道闸。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from utils.embedding import EmbedFn, cosine_similarity, hash_embedding, normalize_for_embedding
from utils.selfcheck import contains_refusal
from utils.text_utils import entity_consistent, length_ratio_ok

@dataclass
class CacheEntry:
    """一条缓存记录。"""

    text: str  # 归一化后的 query
    answer: str
    vector: list[float]
    entities: dict[str, list[str]]
    created_at: float
    tier: str = ""
    hit_count: int = 0
    usage: dict[str, int] = field(default_factory=dict)

@dataclass
class CacheLookup:
    """缓存查询结果。

    ``hit`` 为真时才可以使用 ``answer``；未命中时 ``reason`` 说明原因，
    这是排查「为什么缓存没生效」的第一手材料。
    """

    hit: bool
    answer: str = ""
    entry: CacheEntry | None = None
    similarity: float = 0.0
    entity_consistent: bool = False
    reason: str = ""

    @property
    def usage(self) -> dict[str, int]:
        """命中时复用的 token 计量（作为 cached_input_tokens 计入成本）。"""
        return dict(self.entry.usage) if self.entry else {}

class SemanticCache:
    """语义缓存主体（进程内实现，可替换为 Redis 后端，接口保持不变）。"""

    def __init__(
        self,
        *,
        threshold: float = 0.95,
        ttl_hours: int = 72,
        entity_check: bool = True,
        exclude_pii: bool = True,
        invalidate_on: list[str] | None = None,
        length_ratio_min: float = 0.5,
        length_ratio_max: float = 2.0,
        embed_fn: EmbedFn | None = None,
        max_entries: int = 10000,
        now: Callable[[], float] = time.time,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("semantic_threshold 必须落在 (0, 1]")
        if ttl_hours <= 0:
            raise ValueError("ttl_hours 必须为正数")

        self.threshold = threshold
        self.ttl_seconds = ttl_hours * 3600
        self.entity_check = entity_check
        self.exclude_pii = exclude_pii
        # 失效维度：README 指定 model_version + prompt_version
        self.invalidate_on = list(invalidate_on or ["model_version", "prompt_version"])
        self.length_ratio_min = length_ratio_min
        self.length_ratio_max = length_ratio_max
        self._embed_fn = embed_fn or hash_embedding
        self._max_entries = max_entries
        self._now = now
        self._lock = threading.RLock()
        # 命名空间 → (query 文本 → 条目)，LRU 顺序由 OrderedDict 维护
        self._store: dict[str, OrderedDict[str, CacheEntry]] = {}
        # 便于统计的全局计数
        self.lookups = 0
        self.hits = 0
        self.stores = 0
        self.evictions = 0

    # ---- 基础能力 ----

    def embed(self, text: str) -> list[float]:
        """生成 query 向量；失败返回空向量（上层按未命中处理）。"""
        try:
            return self._embed_fn(text)
        except Exception:  # noqa: BLE001 - embedding 失败不能影响主链路
            return []

    def namespace(self, model_version: str = "", prompt_version: str = "") -> str:
        """按失效维度拼命名空间：模型版本一变命名空间就变，旧缓存自然失效。"""
        parts: list[str] = []
        for dim in self.invalidate_on:
            if dim == "model_version":
                parts.append(f"m={model_version or '-'}")
            elif dim == "prompt_version":
                parts.append(f"p={prompt_version or '-'}")
        return "|".join(parts) or "default"

    # ---- 读 ----

    def lookup(
        self,
        *,
        query: str,
        vector: list[float],
        entities: dict[str, list[str]],
        model_version: str = "",
        prompt_version: str = "",
        has_pii: bool = False,
        no_cache: bool = False,
    ) -> CacheLookup:
        """查询缓存。任何一个环节不通过都按未命中处理，并记录原因。"""
        with self._lock:
            self.lookups += 1

            if no_cache:
                return CacheLookup(hit=False, reason="请求标记为 no_cache（涉实时数据）")
            if has_pii and self.exclude_pii:
                # PII 绝不缓存：A 用户的答案不能因为相似就返回给 B 用户
                return CacheLookup(hit=False, reason="请求含 PII，跳过缓存查询")
            if not vector:
                return CacheLookup(hit=False, reason="embedding 生成失败")

            bucket = self._bucket(model_version, prompt_version)
            self._purge(bucket)

            best_entry: CacheEntry | None = None
            best_similarity = 0.0
            for entry in bucket.values():
                similarity = cosine_similarity(vector, entry.vector)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry

            if best_entry is None:
                return CacheLookup(hit=False, reason="缓存为空")
            if best_similarity < self.threshold:
                return CacheLookup(
                    hit=False,
                    similarity=best_similarity,
                    reason=f"最高相似度 {best_similarity:.4f} 低于阈值 {self.threshold}",
                )
            if self.entity_check and not entity_consistent(entities, best_entry.entities):
                return CacheLookup(
                    hit=False,
                    similarity=best_similarity,
                    entity_consistent=False,
                    reason="二次校验未通过：关键实体不一致",
                )
            if not length_ratio_ok(
                query, best_entry.text, self.length_ratio_min, self.length_ratio_max
            ):
                return CacheLookup(
                    hit=False,
                    similarity=best_similarity,
                    entity_consistent=True,
                    reason="二次校验未通过：query 长度比超出范围",
                )

            # 命中
            best_entry.hit_count += 1
            self.hits += 1
            return CacheLookup(
                hit=True,
                answer=best_entry.answer,
                entry=best_entry,
                similarity=best_similarity,
                entity_consistent=True,
                reason="命中缓存并通过二次校验",
            )

    # ---- 写 ----

    def store(
        self,
        *,
        query: str,
        answer: str,
        vector: list[float],
        entities: dict[str, list[str]],
        model_version: str = "",
        prompt_version: str = "",
        has_pii: bool = False,
        no_cache: bool = False,
        tier: str = "",
        usage: dict[str, int] | None = None,
    ) -> bool:
        """写入缓存，返回是否真的写入。

        写之前的这几道闸很重要：一条错误答案被缓存，会被反复返回给后续查询
        （README 风险清单第 2 条：缓存污染）。
        """
        if not query or not answer:
            return False
        # 写缓存前必须自己做一遍质量校验 —— 拒答/空答案一律不入库
        if not answer.strip() or contains_refusal(answer):
            return False
        if no_cache or (has_pii and self.exclude_pii) or not vector:
            return False

        with self._lock:
            bucket = self._bucket(model_version, prompt_version)
            self._purge(bucket)
            key = normalize_for_embedding(query)
            bucket[key] = CacheEntry(
                text=key,
                answer=answer,
                vector=vector,
                entities=entities,
                created_at=self._now(),
                tier=tier,
                usage=dict(usage or {}),
            )
            bucket.move_to_end(key)
            self.stores += 1
            self._evict_if_needed(bucket)
            return True

    # ---- 失效与运维 ----

    def invalidate(self, *, model_version: str | None = None, prompt_version: str | None = None) -> int:
        """按维度失效缓存；不传参数则清空全部。返回被清除的条目数。"""
        with self._lock:
            if model_version is None and prompt_version is None:
                removed = sum(len(bucket) for bucket in self._store.values())
                self._store.clear()
                return removed

            removed = 0
            for namespace in list(self._store):
                matched = True
                if model_version is not None and f"m={model_version}" not in namespace:
                    matched = False
                if prompt_version is not None and f"p={prompt_version}" not in namespace:
                    matched = False
                if matched:
                    removed += len(self._store.pop(namespace))
            return removed

    def hot_entries(self, min_hits: int = 50) -> list[CacheEntry]:
        """命中次数异常高的条目 —— 需要人工抽检，防止缓存污染被放大（README 风险 2）。"""
        with self._lock:
            found: list[CacheEntry] = []
            for bucket in self._store.values():
                found.extend(entry for entry in bucket.values() if entry.hit_count >= min_hits)
            return sorted(found, key=lambda e: e.hit_count, reverse=True)

    def stats(self) -> dict[str, Any]:
        """缓存统计，供监控面板使用。"""
        with self._lock:
            entries = sum(len(bucket) for bucket in self._store.values())
            return {
                "entries": entries,
                "namespaces": len(self._store),
                "lookups": self.lookups,
                "hits": self.hits,
                "stores": self.stores,
                "evictions": self.evictions,
                "hit_rate": (self.hits / self.lookups) if self.lookups else 0.0,
            }

    # ---- 内部实现 ----

    def _bucket(self, model_version: str, prompt_version: str) -> OrderedDict[str, CacheEntry]:
        namespace = self.namespace(model_version, prompt_version)
        return self._store.setdefault(namespace, OrderedDict())

    def _purge(self, bucket: OrderedDict[str, CacheEntry]) -> int:
        """清除过期条目（时间窗口维度）。"""
        now = self._now()
        expired = [key for key, entry in bucket.items() if now - entry.created_at > self.ttl_seconds]
        for key in expired:
            bucket.pop(key, None)
        return len(expired)

    def _evict_if_needed(self, bucket: OrderedDict[str, CacheEntry]) -> None:
        """容量上限保护：按 LRU 淘汰，避免缓存无限膨胀把内存吃光。"""
        while len(bucket) > self._max_entries:
            bucket.popitem(last=False)
            self.evictions += 1
