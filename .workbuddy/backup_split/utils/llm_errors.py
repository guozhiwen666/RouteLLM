"""模型调用异常分层：区分「重试有用」和「重试没用」。

这是重试策略能写对的前提 —— 把所有失败都当成一回事，
要么会把 400（请求本身写错了）反复重试浪费时间，
要么会把 429（限流，等一下就好）直接判死导致不必要的升级。

判据只有一条：**换个 Key 或换个节点再试一次，结果会不会不一样？**
"""

from __future__ import annotations


class LLMError(Exception):
    """所有模型调用异常的基类。"""


class TransientLLMError(LLMError):
    """可恢复错误：限流 / 5xx / 网络超时 —— 换 Key、换节点、退避重试。"""


class AuthLLMError(LLMError):
    """鉴权失败（401/403）—— 换下一个 Key 试，重试同一个 Key 没意义。"""


class PermanentLLMError(LLMError):
    """不可恢复错误（400/404/422，请求本身有问题）—— 重试多少次都是同一个结果。"""


class NoEndpointError(LLMError):
    """该档位没有可用节点（配置漏了或全部在冷却中）。"""


class AllBackendsFailedError(LLMError):
    """Fallback 链上所有节点都试过了，全部失败。"""
