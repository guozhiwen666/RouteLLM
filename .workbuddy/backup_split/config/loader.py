"""配置加载：把「默认值 → 配置文件 → 环境变量」三级覆盖装配成 RoutingConfig。

取值优先级：**环境变量 > 配置文件 > 代码默认值**。
代码默认值全部取自 README 的 routing.yaml 示例 —— 默认值被改动的唯一正当理由应该是「文档改了」。

本模块只做装配，数据结构定义在 `config/models.py`，
YAML 解析与环境读取分别由 `utils/yaml_subset.py` 与 `utils/env.py` 提供。
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from config.models import (
    DEFAULT_CONFIG_FILE,
    ConfigError,
    LLMConfig,
    RoutingConfig,
    TierConfig,
    llm_config,
)
from utils.env import env_float, env_int, env_str
from utils.yaml_subset import YamlParseError, parse_yaml_subset

logger = logging.getLogger(__name__)


def load_routing_config(
    path: str | os.PathLike[str] | None = None,
    llm: LLMConfig | None = None,
) -> RoutingConfig:
    """加载路由配置。

    :param path: 配置文件路径；None 时取 ROUTELLM_CONFIG 或默认 config/routing.yaml
    :param llm: LLM 连接信息；None 时取模块级单例
    :raises ConfigError: 文件不存在、语法错误、或参数越界
    """
    llm_cfg = llm or llm_config
    file_path = Path(path) if path else Path(env_str("ROUTELLM_CONFIG", str(DEFAULT_CONFIG_FILE)))

    defaults = RoutingConfig(tiers=_default_tiers(llm_cfg))
    routing_raw = _read_routing_file(file_path)

    def pick(key: str, env_key: str, default: Any, cast: type) -> Any:
        """文件取值优先，其次环境变量，最后默认值。"""
        if key in routing_raw:
            return routing_raw[key]
        env_value = os.getenv(env_key)
        if env_value is not None and env_value.strip() != "":
            try:
                return cast(env_value)  # type: ignore[operator]
            except (TypeError, ValueError):
                logger.warning("环境变量 %s=%r 非法，回落 %s", env_key, env_value, default)
        return default

    classifier_raw = routing_raw.get("classifier") or {}
    cache_raw = routing_raw.get("cache") or {}
    guardian_raw = routing_raw.get("guardian") or {}

    cfg = RoutingConfig(
        tiers=_tiers_from_dict(routing_raw.get("tiers"), defaults.tiers),
        classifier_mode=str(classifier_raw.get("mode", defaults.classifier_mode)),
        confidence_threshold=float(
            classifier_raw.get("confidence_threshold")
            if "confidence_threshold" in classifier_raw
            else pick(
                "confidence_threshold",
                "ROUTELLM_CONFIDENCE_THRESHOLD",
                defaults.confidence_threshold,
                float,
            )
        ),
        semantic_threshold=float(cache_raw.get("semantic_threshold", defaults.semantic_threshold)),
        entity_consistency_check=bool(cache_raw.get("entity_consistency_check", defaults.entity_consistency_check)),
        exclude_pii=bool(cache_raw.get("exclude_pii", defaults.exclude_pii)),
        cache_ttl_hours=int(cache_raw.get("ttl_hours", defaults.cache_ttl_hours)),
        shadow_ratio=float(guardian_raw.get("shadow_ratio", defaults.shadow_ratio)),
        max_quality_drop=float(guardian_raw.get("max_quality_drop", defaults.max_quality_drop)),
        consecutive_windows=int(guardian_raw.get("consecutive_windows", defaults.consecutive_windows)),
        auto_rollback=bool(guardian_raw.get("auto_rollback", defaults.auto_rollback)),
        # 以下为工程参数，只从环境变量覆盖（配置文件里没定义）
        threshold_max_step=env_float("ROUTELLM_THRESHOLD_MAX_STEP", defaults.threshold_max_step),
        threshold_cooldown_seconds=env_int("ROUTELLM_THRESHOLD_COOLDOWN_S", defaults.threshold_cooldown_seconds),
        route_budget_ms=env_float("ROUTELLM_ROUTE_BUDGET_MS", defaults.route_budget_ms),
        self_eval_timeout_s=env_float("ROUTELLM_SELF_EVAL_TIMEOUT_S", defaults.self_eval_timeout_s),
        slm_timeout_s=env_float("ROUTELLM_SLM_TIMEOUT_S", defaults.slm_timeout_s),
        cloud_timeout_s=env_float("ROUTELLM_CLOUD_TIMEOUT_S", defaults.cloud_timeout_s),
        judge_timeout_s=env_float("ROUTELLM_JUDGE_TIMEOUT_S", defaults.judge_timeout_s),
        max_retries=env_int("ROUTELLM_MAX_RETRIES", defaults.max_retries),
        backoff_base_s=env_float("ROUTELLM_BACKOFF_BASE_S", defaults.backoff_base_s),
        min_output_tokens=env_int("ROUTELLM_MIN_OUTPUT_TOKENS", defaults.min_output_tokens),
        max_repeat_ngram_ratio=env_float("ROUTELLM_MAX_REPEAT_RATIO", defaults.max_repeat_ngram_ratio),
        window_min_samples=env_int("ROUTELLM_WINDOW_MIN_SAMPLES", defaults.window_min_samples),
        cache_length_ratio_min=env_float("ROUTELLM_CACHE_LEN_RATIO_MIN", defaults.cache_length_ratio_min),
        cache_length_ratio_max=env_float("ROUTELLM_CACHE_LEN_RATIO_MAX", defaults.cache_length_ratio_max),
    )

    # 强制升级规则：配置文件给的是规则名列表，列表即事实来源（没写的规则 = 关闭）
    if "force_upgrade_rules" in routing_raw:
        _apply_force_upgrade_rules(cfg, routing_raw["force_upgrade_rules"])
    if "invalidate_on" in cache_raw:
        cfg.cache_invalidate_on = _parse_invalidate_on(cache_raw["invalidate_on"])

    return cfg.validate()


def _read_routing_file(file_path: Path) -> dict[str, Any]:
    """读取配置文件并返回 routing 节点内容；文件不存在时返回空字典。"""
    if not file_path.is_file():
        # 只有"调用方显式指定了一个不存在的路径"才算错误；默认路径缺失则走全默认值
        if str(file_path) != str(DEFAULT_CONFIG_FILE):
            raise ConfigError(f"配置文件不存在: {file_path}")
        return {}

    try:
        file_data = parse_yaml_subset(file_path.read_text(encoding="utf-8"))
    except YamlParseError as exc:
        raise ConfigError(f"配置文件解析失败: {file_path} ({exc})") from exc
    except OSError as exc:
        raise ConfigError(f"读取配置文件失败: {file_path} ({exc})") from exc

    routing_raw = file_data.get("routing", {}) if isinstance(file_data, dict) else {}
    if not isinstance(routing_raw, dict):
        raise ConfigError("routing 节点必须是键值映射")
    return routing_raw


def _default_tiers(llm: LLMConfig) -> list[TierConfig]:
    """README 的三档定义；本地地址与云端模型优先取环境变量。"""
    return [
        TierConfig(
            name="slm_tiny",
            model=env_str("SLM_TINY_MODEL", "qwen3-0.6b-awq"),
            endpoint=env_str("MODEL_URL", "http://localhost:8000"),
            max_context=8192,
            local=True,
            gpu_hourly_cost_cny=env_float("SLM_TINY_GPU_HOURLY_CNY", 3.0),
        ),
        TierConfig(
            name="slm_mid",
            model=env_str("SLM_MID_MODEL", "qwen3-8b-awq"),
            endpoint=env_str("SLM_MID_URL", "http://localhost:8001"),
            max_context=32768,
            local=True,
            gpu_hourly_cost_cny=env_float("SLM_MID_GPU_HOURLY_CNY", 6.0),
        ),
        TierConfig(
            name="frontier",
            model=llm.llm_model or "claude-frontier-2026-08-01",
            endpoint=llm.base_url or "upstream",
            max_context=200000,
            local=False,
            # 占位价：务必按真实供应商报价调整，否则成本报表是自我安慰
            price_input_per_mtok=env_float("FRONTIER_PRICE_IN_CNY", 70.0),
            price_output_per_mtok=env_float("FRONTIER_PRICE_OUT_CNY", 350.0),
            # 缓存命中的 input 有折扣（README：cached_input_tokens 必须单独统计）
            price_cached_input_per_mtok=env_float("FRONTIER_PRICE_CACHED_IN_CNY", 7.0),
        ),
    ]


def _tiers_from_dict(raw: Any, base: list[TierConfig]) -> list[TierConfig]:
    """把配置文件里的 tiers 合并到默认档位上（同 name 覆盖，未提及的保留默认）。"""
    if not raw:
        return base
    if not isinstance(raw, list):
        raise ConfigError("routing.tiers 必须是列表")

    by_name = {item.name: item for item in base}
    merged: list[TierConfig] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ConfigError(f"routing.tiers[{index}] 必须是键值映射，实际为 {type(item).__name__}")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ConfigError(f"routing.tiers[{index}] 缺少 name")
        known = by_name.get(name)
        defaults: dict[str, Any] = (
            {
                "model": known.model,
                "endpoint": known.endpoint,
                "max_context": known.max_context,
                "price_input_per_mtok": known.price_input_per_mtok,
                "price_output_per_mtok": known.price_output_per_mtok,
                "price_cached_input_per_mtok": known.price_cached_input_per_mtok,
                "gpu_hourly_cost_cny": known.gpu_hourly_cost_cny,
                "local": known.local,
            }
            if known
            else {"model": "", "endpoint": "", "max_context": 4096, "local": None}
        )
        payload = dict(defaults)
        payload.update({key: value for key, value in item.items() if value is not None})
        merged.append(TierConfig(**payload))  # type: ignore[arg-type]
    return merged


def _apply_force_upgrade_rules(cfg: RoutingConfig, raw: Any) -> None:
    """解析 README 的 force_upgrade_rules 列表。

    支持：`contains_math_or_code` / `context_tokens > 24000` /
    `requires_realtime_data` / `output_format == strict_json`。
    先全关再逐条打开，保证「配置里没写的规则 = 关闭」。
    """
    if raw is None:
        return
    if not isinstance(raw, list):
        raise ConfigError("routing.force_upgrade_rules 必须是列表")

    cfg.force_math_or_code = False
    cfg.force_realtime_data = False
    cfg.force_strict_json = False
    for item in raw:
        rule = str(item).strip()
        if rule == "contains_math_or_code":
            cfg.force_math_or_code = True
        elif rule == "requires_realtime_data":
            cfg.force_realtime_data = True
        elif rule == "output_format == strict_json":
            cfg.force_strict_json = True
        elif rule.startswith("context_tokens"):
            matched = re.search(r"(\d+)", rule)
            if not matched:
                raise ConfigError(f"无法解析强制升级规则: {rule!r}")
            cfg.force_context_tokens = int(matched.group(1))
        else:
            logger.warning("忽略未知的强制升级规则: %r", rule)


def _parse_invalidate_on(raw: Any) -> list[str]:
    """缓存失效维度，仅支持 model_version / prompt_version。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raise ConfigError("cache.invalidate_on 必须是列表")

    allowed = {"model_version", "prompt_version"}
    dims: list[str] = []
    for item in raw:
        dim = str(item).strip()
        if dim not in allowed:
            raise ConfigError(f"不支持的缓存失效维度: {dim!r}，可选 {sorted(allowed)}")
        if dim not in dims:
            dims.append(dim)
    return dims
