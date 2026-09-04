"""配置模块入口：对外统一从这里导入，内部实现已按职责拆开。

模块拆分（均在 `config/` 同级目录下）：

  - `config/models.py`       —— 数据结构与取值校验（长什么样、值合不合法）
  - `config/loader.py`       —— 默认值 / 配置文件 / 环境变量的三级装配
  - `config/yaml_subset.py`  —— 受限 YAML 解析（零依赖）
  - `config/env.py`          —— 环境变量安全读取

对外用法保持不变（调用方无需关心内部拆成几块）：

    from config.config import load_routing_config, routing_config, TierConfig
"""

from config.loader import load_routing_config
from config.models import (
    DEFAULT_CONFIG_FILE,
    PROJECT_ROOT,
    ConfigError,
    LLMConfig,
    RoutingConfig,
    TierConfig,
    llm_config,
)
from config.yaml_subset import parse_yaml_subset

# 模块级单例：沿用「导入即配置」的用法，但保证 .env 缺失也不会炸掉
routing_config: RoutingConfig = load_routing_config()

__all__ = [
    "ConfigError",
    "DEFAULT_CONFIG_FILE",
    "LLMConfig",
    "PROJECT_ROOT",
    "RoutingConfig",
    "TierConfig",
    "llm_config",
    "load_routing_config",
    "parse_yaml_subset",
    "routing_config",
]
