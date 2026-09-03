"""向量工具：embedding 兜底实现与相似度计算。

依赖说明：默认用**哈希向量**（hashing trick，纯标准库）作为兜底 embedding，
可运行、可测试、确定性（相同输入永远得到相同向量）。但它只捕捉**词面相似**，
覆盖不了「语义等价的不同问法」。生产环境务必注入真实 embedding 模型
（README 里的 BGE-small，通过 ``SemanticCache(embed_fn=...)`` 替换），
本模块的余弦相似度与归一化逻辑可原样复用。
"""

from __future__ import annotations

import hashlib
import math
from typing import Callable

EmbedFn = Callable[[str], list[float]]


def normalize_for_embedding(text: str) -> str:
    """生成向量前的输入归一化：小写化 + 折叠空白（标点保留，它是语义的一部分）。"""
    if not text:
        return ""
    return " ".join(text.lower().split())


def hash_embedding(text: str, dim: int = 256, ngram: int = 2) -> list[float]:
    """确定性哈希向量：对字符 n-gram 做带符号哈希累加，再 L2 归一化。"""
    vector = [0.0] * dim
    cleaned = normalize_for_embedding(text)
    if not cleaned:
        return vector

    grams = [cleaned[i : i + ngram] for i in range(len(cleaned) - ngram + 1)] or [cleaned]
    for gram in grams:
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % dim
        sign = 1.0 if (value >> 32) & 1 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0.0:
        return vector
    return [component / norm for component in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """余弦相似度。向量为空或维度不一致时返回 0（视为不命中，保守）。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = math.sqrt(sum(a * a for a in left))
    norm_r = math.sqrt(sum(b * b for b in right))
    if norm_l == 0.0 or norm_r == 0.0:
        return 0.0
    # 浮点误差可能让结果略微越界，夹紧到 [-1, 1]
    return max(-1.0, min(1.0, dot / (norm_l * norm_r)))
