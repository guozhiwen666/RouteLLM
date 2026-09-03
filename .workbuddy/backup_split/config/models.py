"""配置数据模型：把 README 里的档位、阈值、缓存与守门参数变成类型化对象。

本模块只负责"长什么样"和"取值是否合法"，不关心从哪里读 ——
读取与合并逻辑在 `config/loader.py`，两者通过 `RoutingConfig` 对接。

计费口径统一为 **CNY / 百万 token**（混用币种会让成本账变成自我安慰）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.env import env_float, env_str

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "routing.yaml"


class ConfigError(ValueError):
    """配置错误：取值缺失、类型非法、或配置文件语法超出支持范围。"""


@dataclass
class LLMConfig:
    """通用 LLM 连接信息（沿用仓库原有字段，补齐默认值与容错解析）。

    原实现的坑：`float(os.getenv("LLM_DEFAULT_TEMPERATURE"))` 在变量缺失时
    会得到 None 并直接 TypeError —— 服务连启动都启动不了。
    """

    base_url: str = ""
    api_key: str = ""
    vl_model: str = ""
    llm_model: str = ""
    item_model: str = ""
    llm_temperature: float = 0.7

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=env_str("OPENAI_API_BASE"),
            api_key=env_str("OPENAI_API_KEY"),
            vl_model=env_str("VL_MODEL"),
            llm_model=env_str("LLM_DEFAULT_MODEL"),
            item_model=env_str("ITEM_MODEL"),
            llm_temperature=env_float("LLM_DEFAULT_TEMPERATURE", 0.7),
        )

    @property
    def configured(self) -> bool:
        """是否具备调用云端模型的最低条件。"""
        return bool(self.base_url and self.api_key)


# 模块级单例：沿用仓库原有「导入即配置」的用法，保证 .env 缺失也不会炸
llm_config: LLMConfig = LLMConfig.from_env()


@dataclass(frozen=True)
class TierConfig:
    """一个模型档位（本地 vLLM 实例或云端供应商）。

    本地档位没有按量费用，但**必须把 GPU 折旧算进去** ——
    GPU 是固定开支不是按量付费，不算折旧则「本地=免费」这个账是错的。
    """

    name: str
    model: str
    endpoint: str
    max_context: int
    price_input_per_mtok: float = 0.0
    price_output_per_mtok: float = 0.0
    price_cached_input_per_mtok: float = 0.0
    gpu_hourly_cost_cny: float = 0.0  # GPU 折旧，元/小时（仅本地档位需要）
    # 是否本地档位；None 表示按 endpoint 推断（README 用 "upstream" 表示云端）。
    # 注意别靠 endpoint 长什么样去猜 —— 云端供应商地址同样是 URL，早晚出事。
    local: bool | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ConfigError("档位缺少 name")
        if not self.model:
            raise ConfigError(f"档位 {self.name} 缺少 model")
        if not self.endpoint:
            raise ConfigError(f"档位 {self.name} 缺少 endpoint")
        if self.max_context <= 0:
            raise ConfigError(f"档位 {self.name} 的 max_context 必须为正整数")
        for price_name in (
            "price_input_per_mtok",
            "price_output_per_mtok",
            "price_cached_input_per_mtok",
            "gpu_hourly_cost_cny",
        ):
            if getattr(self, price_name) < 0:
                raise ConfigError(f"档位 {self.name} 的 {price_name} 不能为负")

    @property
    def is_local(self) -> bool:
        """是否为本地推理档位（数据不出内网）；优先取显式声明的 `local`。"""
        if self.local is not None:
            return self.local
        return self.endpoint.strip().lower() != "upstream"


@dataclass
class RoutingConfig:
    """README routing.yaml 的代码化表示（默认值即文档给出的取值）。"""

    tiers: list[TierConfig] = field(default_factory=list)

    # ---- 路由判定 ----
    classifier_mode: str = "tiny_model_self_eval"  # heuristic | tiny_model_self_eval | trained
    confidence_threshold: float = 0.75  # 自评置信度低于此值 → 保守升级

    # ---- 强制升级规则（命中即走 frontier，不看自评结果）----
    force_math_or_code: bool = True
    force_context_tokens: int = 24000  # context_tokens > 24000
    force_realtime_data: bool = True  # 涉及实时数据
    force_strict_json: bool = True  # output_format == strict_json

    # ---- 语义缓存 ----
    semantic_threshold: float = 0.95
    entity_consistency_check: bool = True
    exclude_pii: bool = True
    cache_ttl_hours: int = 72
    # 失效维度：模型一升级，缓存必须整体失效（README 点名的坑）
    cache_invalidate_on: list[str] = field(default_factory=lambda: ["model_version", "prompt_version"])
    cache_length_ratio_min: float = 0.5  # 二次校验：query 长度比下限
    cache_length_ratio_max: float = 2.0  # 二次校验：query 长度比上限

    # ---- 质量守门 ----
    shadow_ratio: float = 0.05
    max_quality_drop: float = 0.02  # 2pp
    consecutive_windows: int = 3
    auto_rollback: bool = True
    # 阈值调整的变化率限制与冷却期（README：避免震荡）
    threshold_max_step: float = 0.05
    threshold_min: float = 0.5
    threshold_max: float = 0.95
    threshold_cooldown_seconds: int = 300
    # 单窗口质量差超过阈值的这个倍数 → 一键全量切强模型（极端情况）
    shadow_extreme_ratio: float = 5.0
    window_min_samples: int = 20  # 一个统计窗口的最小样本量

    # ---- 时间与延迟预算 ----
    route_budget_ms: float = 15.0  # 路由总开销预算，超了要告警
    self_eval_timeout_s: float = 0.30
    slm_timeout_s: float = 10.0
    cloud_timeout_s: float = 30.0
    judge_timeout_s: float = 20.0
    max_retries: int = 2
    backoff_base_s: float = 0.5

    # ---- 输出自检 ----
    min_output_tokens: int = 4
    max_repeat_ngram_ratio: float = 0.35
    max_upgrade_attempts: int = 1  # 自检失败后最多升级重试 1 次，避免死循环

    # ---- 启发式快筛 ----
    simple_template_max_tokens: int = 50  # 长度 < 50 token
    simple_template_max_instructions: int = 2

    # ---- 其它 ----
    judge_tier: str = "frontier"
    currency: str = "CNY"

    # ------------------------------------------------------------------ #
    # 档位查询
    # ------------------------------------------------------------------ #

    def __post_init__(self) -> None:
        if not self.tiers:
            raise ConfigError("routing.tiers 不能为空")
        names = [t.name for t in self.tiers]
        if len(names) != len(set(names)):
            raise ConfigError(f"routing.tiers 存在重复档位名: {names}")
        if self.judge_tier not in names:
            logger.warning("judge_tier=%s 不在档位列表中，将退化为最后一个档位", self.judge_tier)

    def tier(self, name: str) -> TierConfig:
        for item in self.tiers:
            if item.name == name:
                return item
        raise ConfigError(f"未定义的档位: {name}")

    @property
    def local_tiers(self) -> list[TierConfig]:
        """本地档位，按上下文容量升序 —— 优先用最小的、最便宜的那个。"""
        return sorted((item for item in self.tiers if item.is_local), key=lambda t: t.max_context)

    @property
    def frontier_tiers(self) -> list[TierConfig]:
        return [item for item in self.tiers if not item.is_local]

    def frontier_tier(self) -> TierConfig:
        if not self.frontier_tiers:
            raise ConfigError("未配置 frontier（云端）档位，无法升级")
        return self.frontier_tiers[0]

    def pick_local_tier(self, context_tokens: int) -> TierConfig | None:
        """挑能装下该上下文的最小本地档位；装不下返回 None（意味着必须上云）。"""
        for item in self.local_tiers:
            if context_tokens <= item.max_context:
                return item
        return None

    def validate(self) -> "RoutingConfig":
        """关键参数取值校验：越界直接报错，而不是等线上出事。"""
        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ConfigError("classifier.confidence_threshold 必须落在 (0, 1]")
        if not 0.0 < self.semantic_threshold <= 1.0:
            raise ConfigError("cache.semantic_threshold 必须落在 (0, 1]")
        if not 0.0 <= self.shadow_ratio <= 1.0:
            raise ConfigError("guardian.shadow_ratio 必须落在 [0, 1]")
        if not 0.0 <= self.max_quality_drop <= 1.0:
            raise ConfigError("guardian.max_quality_drop 必须落在 [0, 1]")
        if self.consecutive_windows < 1:
            raise ConfigError("guardian.consecutive_windows 必须 >= 1")
        if self.cache_ttl_hours <= 0:
            raise ConfigError("cache.ttl_hours 必须为正数")
        if self.classifier_mode not in {"heuristic", "tiny_model_self_eval", "trained"}:
            raise ConfigError(f"不支持的 classifier.mode: {self.classifier_mode}")
        return self

    def to_dict(self) -> dict[str, Any]:
        """导出为普通字典（便于日志/调试打印，不含敏感字段之外的加工）。"""
        return {
            "classifier_mode": self.classifier_mode,
            "confidence_threshold": self.confidence_threshold,
            "semantic_threshold": self.semantic_threshold,
            "shadow_ratio": self.shadow_ratio,
            "max_quality_drop": self.max_quality_drop,
            "tiers": [
                {
                    "name": item.name,
                    "model": item.model,
                    "endpoint": item.endpoint,
                    "is_local": item.is_local,
                    "max_context": item.max_context,
                }
                for item in self.tiers
            ],
        }
