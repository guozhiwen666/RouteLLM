"""图引擎结构校验与执行语义测试（自写 StateGraph/CompiledGraph）。

共享替身（MockLLM / make_config / make_workflow）在 tests/helpers.py。
运行：cd 项目根 && python -m unittest discover -s tests
"""

from __future__ import annotations

import unittest

from helpers import *  # noqa: F401,F403 - 测试替身与工具函数

class TestGraphEngine(unittest.TestCase):
    def test_missing_entry_raises(self) -> None:
        graph = StateGraph()
        graph.add_node("a", lambda state: state)
        with self.assertRaises(GraphError):
            graph.compile()

    def test_unknown_edge_target_raises(self) -> None:
        graph = StateGraph()
        graph.add_node("a", lambda state: state)
        graph.set_entry_point("a")
        graph.add_edge("a", "不存在")
        with self.assertRaises(GraphError):
            graph.compile()

    def test_unknown_branch_raises(self) -> None:
        graph = StateGraph()
        graph.add_node("a", lambda state: state)
        graph.set_entry_point("a")
        graph.add_conditional_edges("a", lambda state: "拼写错误的分支", {"ok": END})
        app = graph.compile()
        with self.assertRaises(GraphError):
            app.invoke({})

    def test_static_and_conditional_conflict(self) -> None:
        graph = StateGraph()
        graph.add_node("a", lambda state: state)
        graph.add_edge("a", END)
        with self.assertRaises(GraphError):
            graph.add_conditional_edges("a", lambda state: "x", {"x": END})

    def test_cycle_is_detected(self) -> None:
        graph = StateGraph()
        graph.add_node("a", lambda state: state)
        graph.set_entry_point("a")
        graph.add_edge("a", "a")  # 自己指向自己
        app = graph.compile(max_steps=5)
        with self.assertRaises(GraphError):
            app.invoke({})

    def test_stream_yields_each_node(self) -> None:
        llm = MockLLM()
        workflow = make_workflow(llm)
        steps = list(workflow.run(user_request("请将这句话改写成正式语气：这份报告逻辑有点乱"), stream=True))
        names = [name for name, _ in steps]
        self.assertIn("query_process", names)
        self.assertIn("cache_query", names)
        self.assertIn("result_process", names)

    def test_invalid_messages_are_recorded_not_raised(self) -> None:
        """入参错误由 BaseNode.run 收敛成 state['error']，不炸掉链路。

        注意这不是"吞异常"：非法入参的链路会立刻走到 result_process 收尾，
        错误信息完整留在 state 里，调用方可以直接判断。
        """
        llm = MockLLM()
        workflow = make_workflow(llm)
        for bad in ({}, {"messages": []}, {"messages": [{"role": "robot", "content": "hi"}]}):
            result = workflow.run(dict(bad))
            self.assertTrue(result["error"], f"应当记录错误: {bad}")
        result = workflow.run({"messages": [{"role": "user", "content": "   "}]})
        self.assertIn("content", result["error"])

    def test_node_exception_does_not_break_chain(self) -> None:
        """节点抛异常时由 BaseNode.run 记录到 error，链路继续走保守路径。"""
        llm = MockLLM()
        workflow = make_workflow(llm, make_config(max_upgrade_attempts=1))
        result = workflow.run(user_request("请将这句话改写成正式语气：这份报告逻辑有点乱", request_id="req-err"))
        self.assertIsInstance(result, dict)
        self.assertIn("result_process", result["trace"])


# --------------------------------------------------------------------------- #
# 客户端与计量
# --------------------------------------------------------------------------- #
