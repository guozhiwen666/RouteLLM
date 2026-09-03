"""环境变量读取工具：把「取不到」和「取到了但格式不对」都处理掉。

设计原则：**配置写错不应该让服务起不来**。
显式写错的环境变量一律告警并回落默认值，而不是抛异常 ——
但也不是静默忽略，告警日志是排查「为什么配置没生效」的唯一线索。
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
