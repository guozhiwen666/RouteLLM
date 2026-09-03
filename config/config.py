"""RouteLLM 配置层（单文件版）。

由环境变量读取 + YAML 子集解析器 + 配置对象 + 加载逻辑四部分合并而成
（回滚后的布局）。默认值取自 config/routing.yaml 与 README，
优先级：环境变量 > 配置文件 > 默认值。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def env_str(key: str, default: str = "") -> str:
    """读取字符串；缺失或空串都回落到默认值。"""
    value = os.getenv(key)
    return default if value is None or value == "" else value


def env_float(key: str, default: float) -> float:
    """读取浮点数；解析失败时告警并回落默认值。"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("环境变量 %s=%r 不是合法浮点数，回落默认值 %s", key, raw, default)
        return default


def env_int(key: str, default: int) -> int:
    """读取整数；解析失败时告警并回落默认值。"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("环境变量 %s=%r 不是合法整数，回落默认值 %s", key, raw, default)
        return default


def env_bool(key: str, default: bool) -> bool:
    """读取开关；1/true/yes/y/on 视为真，其余视为假。"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in _TRUE_VALUES

from typing import Any

_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}
_NULL = {"null", "~", ""}


class YamlParseError(ValueError):
    """YAML 子集解析失败。"""


def parse_yaml_subset(text: str) -> dict[str, Any]:
    """解析 YAML 子集文本，返回嵌套字典；顶层不是映射时返回空字典。"""
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = _strip_comment(raw_line)
        if not stripped.strip():
            continue
        if "\t" in stripped[: len(stripped) - len(stripped.lstrip())]:
            raise YamlParseError("配置文件不支持 Tab 缩进，请改用空格")
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))
    if not lines:
        return {}
    value, _ = _parse_block(lines, 0, lines[0][0])
    return value if isinstance(value, dict) else {}


def _strip_comment(line: str) -> str:
    """去掉行注释（不处理引号内的 #，本项目配置里没有这种写法）。"""
    index = line.find("#")
    if index == -1:
        return line
    # 只在 # 位于行首或前面有空白时视为注释，避免误伤 http://a#b 之类的值
    if index == 0 or line[index - 1] in " \t":
        return line[:index]
    return line


def _coerce_scalar(raw: str) -> Any:
    """把字符串标量还原成 Python 基础类型（引号 / 行内数组 / 布尔 / 空 / 数字 / 字符串）。"""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_coerce_scalar(part) for part in inner.split(",")]
    lowered = text.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    if lowered in _NULL:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _looks_like_key(line: str) -> bool:
    """判断一行是否为 `key: value`（用 ': ' 切分，避免误伤 http:// 这类值）。"""
    return ": " in line or line.endswith(":")


def _parse_block(lines: list[tuple[int, str]], pos: int, indent: int) -> tuple[Any, int]:
    """按缩进递归解析一个块，返回 (值, 下一行位置)。"""
    if pos >= len(lines):
        return None, pos
    current = lines[pos][1]
    if current.startswith("-") and (len(current) == 1 or current[1] in " \t"):
        return _parse_sequence(lines, pos, indent)
    return _parse_mapping(lines, pos, indent)


def _parse_sequence(lines: list[tuple[int, str]], pos: int, indent: int) -> tuple[list[Any], int]:
    """解析 `- ` 开头的列表块，支持列表项为键值映射的写法。"""
    result: list[Any] = []
    while pos < len(lines) and lines[pos][0] == indent and lines[pos][1].startswith("-"):
        content = lines[pos][1][1:].strip()
        pos += 1
        if not content:
            # "- " 单独一行，值在后续缩进块里
            value, pos = (
                _parse_block(lines, pos, lines[pos][0])
                if pos < len(lines) and lines[pos][0] > indent
                else (None, pos)
            )
            result.append(value)
            continue

        if _looks_like_key(content):
            item: dict[str, Any] = {}
            key, _, value_text = content.partition(":")
            key, value_text = key.strip(), value_text.strip()
            item[key] = _coerce_scalar(value_text) if value_text else None
            if pos < len(lines) and lines[pos][0] > indent:
                sub, pos = _parse_block(lines, pos, lines[pos][0])
                if isinstance(sub, dict) and value_text:
                    item.update(sub)  # 后续行是同级的其它键
                elif not value_text:
                    item[key] = sub  # 值本身是个嵌套块
            result.append(item)
        else:
            result.append(_coerce_scalar(content))
    return result, pos


def _parse_mapping(lines: list[tuple[int, str]], pos: int, indent: int) -> tuple[dict[str, Any], int]:
    """解析 `key: value` 组成的映射块。"""
    result: dict[str, Any] = {}
    while pos < len(lines) and lines[pos][0] == indent:
        line = lines[pos][1]
        if line.startswith("-"):
            break
        if ": " not in line and not line.endswith(":"):
            raise YamlParseError(f"配置文件语法错误（第 {pos + 1} 个有效行）: 缺少冒号 -> {line!r}")
        key, _, value_text = line.partition(":")
        key, value_text = key.strip(), value_text.strip()
        pos += 1
        if value_text:
            result[key] = _coerce_scalar(value_text)
        elif pos < len(lines) and lines[pos][0] > indent:
            sub, pos = _parse_block(lines, pos, lines[pos][0])
            result[key] = sub
        else:
            result[key] = {}
    return result, pos

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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

import logging
import os
import re
from pathlib import Path
from typing import Any


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


# ---- 模块级单例：沿用「导入即配置」的用法（.env 缺失不会炸） ----
routing_config = load_routing_config()
