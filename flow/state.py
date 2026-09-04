"""图状态定义：一次请求在整条链路上流转时共享的**公共数据总线**。

设计约定（节点实现必须遵守，否则链路会变得不可推理）：

1. 节点**不得原地修改**传入的 state，必须先 `dict(state)` 拷贝、再写字段、最后返回新对象；
2. 这里只放**多个节点都会读写、或调用方需要拿到的公共字段**；
   某个节点独有的中间产物，一律在该节点自己的模块里用
   `class XxxState(GraphState)` 扩展声明，不要塞进公共总线；
3. 时间统一用秒（float，time.perf_counter 相对值），延迟统一用毫秒；
4. 节点可以声明 `state_schema` 类属性，图引擎会汇总所有 schema 做字段拼写检查。

为什么拆分：字段全堆在一处会让「谁能改哪个字段」失去约束，
最终演变成谁都敢写、谁都不敢删的公共泥潭。拆开之后，
节点的输入输出契约就写在节点旁边，读一个文件就能看懂一个节点。
"""

from __future__ import annotations

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """所有节点共享的公共字段。

    判断标准（满足任一即可留在这里）：
      - 入口/出口契约：调用方传进来、或最终要交还给调用方的；
      - 跨节点：两个以上节点会读写的；
      - 引擎维护：图引擎自己会写入或依赖的。
    """

    # ---- 入口契约（调用方传入 / query_process_node 补齐） ----
    request_id: str  # 请求唯一 ID，影子流量抽样、追踪、排障都依赖它
    messages: list[dict[str, str]]  # OpenAI 格式对话列表 [{role, content}]
    query: str  # 主文本（取最后一条 user 消息），缓存与自评都用它
    normalized_query: str  # 归一化后的文本（折叠空白、去噪）

    # ---- 出口契约（result_process_node 定稿后交还调用方） ----
    final_output: str  # 最终返回给用户的答案
    final_tier: str  # 实际产出答案的档位（slm_tiny / slm_mid / frontier / cache）

    # ---- 输出自检（slm/cloud 两个推理节点都会写，路由器据此决定升级） ----
    self_check: dict  # 输出自检结果 {passed, mode, reasons, repeat_ratio}

    # ---- 计量（result_process_node 写，监控与账单读） ----
    usage: dict[str, int]  # input_tokens / cached_input_tokens / output_tokens
    cost_cny: float  # 本次请求成本（CNY）
    latency_ms: dict[str, float]  # 各阶段耗时

    # ---- 引擎维护（BaseNode.run 与图引擎写入，路由函数读取） ----
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
