"""图状态定义：一次请求在整条链路上流转时共享的数据总线。

设计约定（节点实现必须遵守，否则链路会变得不可推理）：

1. 节点**不得原地修改**传入的 state，必须先 `dict(state)` 拷贝、再写字段、最后返回新对象；
2. 所有会跨节点流转的字段都集中声明在这里，按「谁写入」分组注释，
   方便一眼看出每个字段的生命周期归属；
3. 时间统一用秒（float，time.perf_counter 相对值），延迟统一用毫秒；
4. 图引擎会拿这里的字段集合做拼写检查——拼错字段名会被日志告警，
   而不是悄悄埋雷。

（注：本文件为回滚后的单文件布局——节点不再各自声明 state_schema，
全部字段回归 GraphState 统一管理。）
"""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """一次请求在图链路上流转的全部状态字段。

    分组规则：按写入方（哪个节点/环节负责填这些字段）归类。
    """

    # ---------- 入口契约（调用方传入 / query_process_node 补齐） ----------
    request_id: str  # 请求唯一 ID，影子流量抽样、追踪、排障都依赖它
    messages: list[dict[str, str]]  # OpenAI 格式对话列表 [{role, content}]
    query: str  # 主文本（取最后一条 user 消息），缓存与自评都用它
    normalized_query: str  # 归一化后的文本（折叠空白、去噪）

    # ---------- 输入分析（query_process_node 写，全下游读） ----------
    prompt_version: str  # prompt 版本（缓存失效维度之一）
    model_version: str  # 模型版本（缓存失效维度之一）
    output_format: str  # text | strict_json
    temperature: float  # 推理温度
    max_tokens: int  # 输出上限
    stream: bool  # 是否流式返回（影响输出自检的严格程度）

    context_tokens: int  # 估算的上下文 token 数
    features: dict  # 启发式特征：code / math / realtime / instructions / language / template
    has_pii: bool  # 命中 PII → 不写缓存、优先本地推理
    pii_types: list[str]  # 命中的 PII 类型列表
    entities: dict  # 关键实体（地名/时间/金额/型号），缓存二次校验用
    no_cache: bool  # 涉实时数据 → 不缓存

    # ---------- 语义缓存（cache_query_node 查 / result_process_node 写） ----------
    embedding: list[float]  # 请求向量（哈希兜底，生产可注入真实 embedding）
    cache_hit: bool  # 是否命中缓存
    cache_similarity: float  # 与缓存条目最高相似度
    cache_entity_consistent: bool  # 实体二次校验是否通过
    cache_reason: str  # 未命中原因（排查「为什么没省到钱」用）
    cached_answer: str  # 命中时的缓存答案

    # ---------- 路由决策（cache_query_node 写，推理节点与计量读） ----------
    route_stage: str  # cache | force_upgrade | heuristic | self_eval | self_check_upgrade
    route_tier: str  # 档位名（slm_tiny / slm_mid / frontier）
    route_reason: str  # 路由理由（留痕）
    difficulty: str  # simple | moderate | hard
    confidence: float  # 路由置信度（自评结果 / 快筛给 0.99）
    self_eval_raw: str  # 小模型自评的原始输出（排障用）
    force_upgrade_hits: list[str]  # 命中的强制升级规则名
    upgrade_count: int  # 升级次数（防乒乓：自检升级 + 云端累加）

    # ---------- 推理执行（slm/cloud 节点写） ----------
    slm_response: str  # 本地 SLM 原始输出
    cloud_response: str  # 云端模型原始输出
    self_check: dict  # 输出自检结果 {passed, mode, reasons, repeat_ratio}

    # ---------- 质量守门（comparative_scoring_node 写） ----------
    shadow: bool  # 本条请求是否被抽中为影子流量
    shadow_output: str  # 影子双跑的强模型结果（不返回给用户）
    judge_score: float  # judge 给出的 SLM 相对强模型的质量分
    quality_gap: float  # 质量差（正 = SLM 更差）
    guardian_action: str  # submitted | maintain | raise_threshold | force_frontier

    # ---------- 出口契约（result_process_node 定稿后交还调用方） ----------
    final_output: str  # 最终返回给用户的答案
    final_tier: str  # 实际产出答案的档位（slm_tiny / slm_mid / frontier / cache）

    # ---------- 计量（result_process_node 写，监控与账单读） ----------
    usage: dict[str, int]  # input_tokens / cached_input_tokens / output_tokens
    cost_cny: float  # 本次请求成本（CNY）
    latency_ms: dict[str, float]  # 各阶段耗时（毫秒）

    # ---------- 引擎维护（BaseNode.run 与图引擎写入，路由函数读取） ----------
    error: str  # 最近一次错误描述；非空时上层路由器走保守路径
    warnings: list[str]  # 不影响主流程但需要留痕的异常
    trace: list[str]  # 实际执行过的节点顺序


def state_fields(*schemas: type) -> set[str]:
    """汇总一批 TypedDict 声明的字段名，供图引擎做拼写检查。"""
    fields: set[str] = set()
    for schema in schemas:
        if schema is None:
            continue
        fields.update(getattr(schema, "__annotations__", {}) or {})
    return fields
