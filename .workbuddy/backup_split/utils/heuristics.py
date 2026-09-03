"""输入侧启发式分析：零延迟、零成本、可解释的第一道快速通道。

README 的取舍很明确：启发式规则覆盖不全、容易被绕过，但它的成本是 0 且完全可解释，
所以定位是**第一道快速通道**，而不是最终裁判 —— 判不了的请求交给小模型自评。

本模块只做"特征检测"，不做"决策"（决策在 cache_query_node），规则可以单独测试和替换。
通用文本工具（token 估算 / 实体 / PII / JSON 提取）已下沉到 `utils/text_utils.py`。
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------- #
# 强制升级规则用到的特征检测
# --------------------------------------------------------------------------- #

_CODE_LANGUAGES = (
    "python", "javascript", "typescript", "java", "golang", "rust", "c++", "c#",
    "sql", "bash", "shell", "powershell", "scala", "kotlin", "swift", "php", "ruby",
    "html", "css", "vue", "react", "django", "flask",
)


def detect_code(text: str) -> bool:
    """是否含代码。

    量化对代码生成的伤害中等但明确（语法错误率上升），命中即强制升级。
    信号按强度排序：代码块围栏 > 语句关键字 > 结构标点 > 编程语言点名。
    """
    if not text:
        return False
    hints = (
        "```",  # 代码块围栏
        "def ", "class ", "import ", "function", "return ", "const ", "var ", "let ",
        "SELECT ", "INSERT ", "UPDATE ", "<?php", "#include", "public static",
        "async def", "lambda", "print(", "console.log",
    )
    lowered = text.lower()
    if any(hint.lower() in lowered for hint in hints):
        return True
    # 代码味很重的标点组合：连续出现花括号 / 箭头 / 分号
    if re.search(r"(=>|::|\{\s*\n|\}\s*;|;\s*\n\s*[a-z_]+\s*\()", text):
        return True
    # 明确点名的编程语言（"用 Python 写个快排" 也算）
    return any(re.search(rf"\b{language}\b", lowered) for language in _CODE_LANGUAGES)


def detect_math(text: str) -> bool:
    """是否含数学/多步推理。

    README 的失败案例第一条就是它：量化对数学推理伤害 10–20pp，但自评往往判为「简单」。
    """
    if not text:
        return False
    keywords = (
        "计算", "求解", "证明", "推导", "解方程", "积分", "导数", "概率", "矩阵", "增长率",
        "复利", "百分比", "平均值", "方差", "最大公约数", "最小公倍数",
        "calculate", "solve", "prove", "derivative", "integral", "probability",
    )
    if any(word in text for word in keywords) or any(word in text.lower() for word in keywords):
        return True
    # 明显的算式：数字 + 运算符 + 数字
    if re.search(r"\d+\s*[\+\-\*/×÷]\s*\d+\s*=", text):
        return True
    # LaTeX 痕迹
    return bool(re.search(r"\\frac|\\sqrt|\^\{|\$[^$]+\$", text))


# 强时间信号：单独出现就说明要"此刻的数据"
_REALTIME_STRONG = (
    "现在", "当前", "实时", "最新", "此刻", "目前", "当下", "此时", "刚刚", "正在",
    "now", "current", "latest", "right now", "at the moment", "live", "up to date",
)

# 数据类名词：天然依赖最新数据
_REALTIME_DATA_NOUNS = (
    "天气", "气温", "股价", "股票", "报价", "库存", "余额", "汇率", "销量", "新闻",
    "热搜", "榜单", "排队", "余票", "航班", "快递", "物流", "油价", "金价",
    "weather", "stock price", "inventory", "balance", "exchange rate", "headline",
)


def detect_realtime_data(text: str) -> bool:
    """是否依赖实时数据（「今天股价」「当前库存」）—— 命中即打 no_cache 标记。

    判定分三档，从强到弱：
      1. 强时间信号（现在 / 当前 / 实时 / 最新 …）单独成立；
      2. 数据类名词（股价 / 天气 / 库存 / 汇率 …）单独成立 ——
         这类问题天然依赖最新数据，宁可错杀也不要把昨天的答案缓存下来；
      3. 弱时间词（今天 / 本周 …）单独**不成立** ——
         「把今天天气很好翻译成英文」里的"今天"只是句子内容，
         只凭它判定为实时请求会造成大量无谓升级。
    """
    if not text:
        return False
    lowered = text.lower()
    if any(marker in text or marker in lowered for marker in _REALTIME_STRONG):
        return True
    return any(noun in text or noun in lowered for noun in _REALTIME_DATA_NOUNS)


def detect_strict_json(text: str) -> bool:
    """是否要求严格的 JSON 输出（命中即强制升级，量化对格式化输出不可靠）。"""
    if not text:
        return False
    patterns = (
        r"严格.*json", r"只(?:返回|输出).*json", r"仅(?:返回|输出).*json",
        r"json\s*(?:格式|对象|schema)", r"strict\s+json", r"only\s+(?:return|output)\s+json",
        r"respond\s+(?:in|with)\s+json", r"json\s+schema",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def count_instructions(text: str) -> int:
    """统计指令条数：指令越多，越容易超出小模型的指令跟随能力。

    粗略规则：祈使动词与编号列表条目分别计数，取较大者（避免重复计数）。
    """
    if not text:
        return 0
    imperative = len(
        re.findall(r"(?:请|帮我|需要你|你要|务必|先|然后|接着|最后|另外|同时)", text)
    )
    numbered = len(re.findall(r"(?:^|\n)\s*(?:\d+[.、)]|[-*·])\s*\S", text))
    sentences = len(re.findall(r"[。!！?？;；]", text))
    return max(numbered, min(imperative, sentences) if sentences else imperative)


def detect_language(text: str) -> str:
    """语种粗判：中文 / 英文 / 混排 / 其它（小语种是量化的重灾区）。"""
    if not text:
        return "unknown"
    cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]", text))
    total_alpha = len(re.findall(r"[A-Za-z]", text))
    if cjk and total_alpha:
        return "mixed"
    if cjk:
        return "zh"
    if total_alpha:
        return "en"
    return "other"


# --------------------------------------------------------------------------- #
# 简单模板：启发式快筛的「极高置信简单」判定依据
# --------------------------------------------------------------------------- #

SIMPLE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "rewrite": ("改写", "润色", "重写", "换个说法", "精简", "扩写", "rewrite", "rephrase", "polish"),
    "translate": ("翻译", "译成", "英文怎么说", "translate"),
    "classify": ("分类", "归类", "打标", "判断属于", "属于哪", "classify", "categorize"),
    "extract": ("提取", "抽取", "抽出来", "列出所有", "extract", "list all"),
    "format": ("格式化", "转成表格", "转成列表", "转为", "format", "convert to"),
    "summarize": ("总结一下", "简要总结", "摘要", "summarize", "tl;dr"),
}


def match_simple_template(text: str) -> str | None:
    """命中已知的简单任务模板则返回模板名，否则返回 None。

    这些任务（格式转换 / 简单改写 / 分类 / 抽取）正是 README 说的
    「根本不需要强模型」的那 70%，也是 4bit 量化后几乎无损的场景。
    """
    if not text:
        return None
    lowered = text.lower()
    for name, keywords in SIMPLE_TEMPLATES.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            return name
    return None


def build_features(text: str, messages: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """汇总一次请求的全部启发式特征，供快筛与强制升级规则共用。"""
    return {
        "contains_code": detect_code(text),
        "contains_math": detect_math(text),
        "requires_realtime_data": detect_realtime_data(text),
        "strict_json": detect_strict_json(text),
        "instruction_count": count_instructions(text),
        "language": detect_language(text),
        "simple_template": match_simple_template(text),
        "turns": len(messages or []),
    }
