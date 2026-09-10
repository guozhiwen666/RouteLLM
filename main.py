"""RouteLLM 网关演示入口。

用法：
    python main.py                        # 跑一组演示请求，打印每条的路由决策
    python main.py --query "帮我总结这段话"  # 只跑单条请求

模型接入方式：
  - 本地 SLM：设置 MODEL_URL（vLLM 实例，OpenAI 兼容）
  - 云端 LLM：设置 OPENAI_API_BASE / OPENAI_API_KEY / LLM_DEFAULT_MODEL
  - 什么都没配：自动使用内置假后端，把整条链路（缓存/路由/守门/计量）完整演示出来，
    方便在不占 GPU 的情况下跑通代码 —— 真后端只是把传输层换成 urllib 而已。

注意：本项目按设计文档只实现「核心路由链路」，README 中的 FastAPI 网关、
Redis 向量缓存、Prometheus 拉取等属于周边设施（需要额外第三方依赖），
不在本次代码范围内；语义缓存的进程内实现可用 Redis 版替换，接口保持不变。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# 保证从任意目录执行都能 import 到本项目
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_routing_config  # noqa: E402
from config.models import llm_config  # noqa: E402
from flow.graph import Workflow, build_chat_client  # noqa: E402

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


DEMO_BATCH = [
    # (请求文本, 期望走的路径注释)
    ("把这句话翻译成英文：好的，我明天上午十点前把报告发你邮箱", "启发式快筛 → 本地 SLM"),
    ("把这句话翻译成英文：好的，我明天上午十点前把报告发你邮箱", "（同上，第二次应命中语义缓存）"),
    ("请用 Python 写一个快速排序函数，要求处理边界情况", "含代码 → 强制升级云端"),
    ("请计算 12345 × 6789 等于多少", "数学 → 强制升级云端（量化重灾区）"),
    ("帮我查一下北京的实时路况", "实时数据（强信号）→ 云端且不缓存"),
    ("请帮我写一段话，向朋友推荐一款降噪耳机", "普通生成任务 → 自评通过 → 本地 SLM"),
    ("这是很长的一段文档。" * 6000, "超长上下文（约 3.6 万 token）→ 超出本地档位，强制升级"),
]


def build_workflow(config_path: str | None) -> tuple[Workflow, bool]:
    """构建工作流。返回 (workflow, 是否真实后端)。"""
    cfg = load_routing_config(config_path)

    # 判断后端可用性：只看环境变量有没有真的配，而不是配置里的 endpoint
    # 长得像 http —— localhost 默认地址没人监听时照样会连接拒绝
    local_configured = bool(os.getenv("MODEL_URL"))
    cloud_configured = bool(os.getenv("OPENAI_API_BASE")) and bool(os.getenv("OPENAI_API_KEY"))
    real_available = local_configured or cloud_configured

    if real_available:
        logging.info("检测到模型端点配置，使用真实 HTTP 后端")
        api_keys = [llm_config.api_key] if llm_config.api_key else []
        client = build_chat_client(cfg, api_keys=api_keys)
        workflow = Workflow(client=client, config=cfg)
        return workflow, True

    # 假后端：任何模型调用都返回固定的长回答
    logging.warning("未检测到 MODEL_URL / OPENAI_API_BASE 配置，使用内置假后端演示链路")
    from utils.llm_client import ChatClient, Endpoint

    def fake_transport(url: str, payload: dict, headers: dict, timeout: float):
        # 让自评返回高置信，这样假后端也能演示"自评通过走本地"的分支
        first = payload.get("messages", [{}])[0]
        system = first.get("content") or ""
        if "路由难度评估器" in system:
            content = '{"can_answer": true, "confidence": 0.9, "difficulty": "simple", "reason": "demo"}'
        elif "结果评审员" in system:
            content = '{"a_score": 8, "b_score": 9, "reason": "demo judge"}'
        else:
            content = f"【{payload['model']} 的演示回答】这条内容足够长，可以通过输出自检，不会被误判为过短。"
        body = json.dumps(
            {
                "model": payload["model"],
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 32, "completion_tokens": 48, "prompt_tokens_details": {"cached_tokens": 0}},
            }
        )
        return 200, body

    client = ChatClient(transport=fake_transport, max_retries=cfg.max_retries, jitter=False)
    for tier in cfg.tiers:
        # 云端档位的 endpoint 是 upstream 哨兵值，假后端里给它一个虚拟地址即可
        base_url = tier.endpoint if tier.endpoint.startswith("http") else "http://mock-frontier.local"
        client.add_endpoint(tier.name, Endpoint(name=tier.name, base_url=base_url))
    workflow = Workflow(client=client, config=cfg)
    return workflow, False


def _short(text: str, limit: int = 30) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def run_one(workflow: Workflow, text: str, note: str = "") -> None:
    """跑一条请求并打印决策摘要。"""
    state = workflow.run(
        {
            "messages": [{"role": "user", "content": text}],
            "request_id": f"demo-{abs(hash(text)):x}",
        }
    )
    latency = state.get("latency_ms") or {}
    total_ms = latency.get("total_ms", 0)
    cache_tag = "缓存命中" if state.get("cache_hit") else "-"
    print(
        f"  输入: {_short(text)}"
        f"\n  路径: {_short(' → '.join(state.get('trace') or []), 90)}"
        f"\n  档位: {state.get('final_tier', '?')!s:<10} 命中: {cache_tag:<6}"
        f" 耗时: {total_ms:>6.1f}ms  成本: {state.get('cost_cny', 0):.6f} CNY"
        f"\n  理由: {state.get('route_reason', '')}"
    )
    if state.get("warnings"):
        for warning in state["warnings"]:
            print(f"  警告: {warning}")
    if note:
        print(f"  预期: {note}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RouteLLM 分级路由演示")
    parser.add_argument("--config", default=None, help="routing.yaml 配置文件路径")
    parser.add_argument("--query", default=None, help="只跑单条请求")
    args = parser.parse_args()

    workflow, real = build_workflow(args.config)
    print(f"RouteLLM 演示（后端: {'真实 HTTP' if real else '内置假后端'}）")
    print("=" * 72)

    if args.query:
        run_one(workflow, args.query)
    else:
        for text, note in DEMO_BATCH:
            run_one(workflow, text, note)

    print("=" * 72)
    print("计量汇总:")
    print(json.dumps(workflow.summary(), ensure_ascii=False, indent=2))
    print("\nPrometheus 指标（前 12 行示例）:")
    for line in workflow.metrics().splitlines()[:12]:
        print(" ", line)

    workflow.shutdown()


if __name__ == "__main__":
    main()
