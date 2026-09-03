from __future__ import annotations

import json
import math
import re
from typing import Any

# --------------------------------------------------------------------------- #
# 归一化与 token 估算
# --------------------------------------------------------------------------- #

_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_WORD_PATTERN = re.compile(r"[A-Za-z0-9_'\-]+")
_NOISE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """归一化：去首尾空白、统一换行、折叠连续空白（缓存匹配前的必要步骤）。"""
    if not text:
        return ""
    return _NOISE_PATTERN.sub(" ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def count_cjk(text: str) -> int:
    """统计中日韩文字符数（语种与 token 估算都要用）。"""
    return len(_CJK_PATTERN.findall(text)) if text else 0


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（加权字符法）：
      - 中文/日文/韩文字符 ≈ 0.6 token
      - 拉丁单词/数字 ≈ 1.3 token
      - 其余标点/符号 ≈ 0.5 token
    """
    if not text:
        return 0
    cjk = count_cjk(text)
    rest = _CJK_PATTERN.sub(" ", text)
    words = len(_WORD_PATTERN.findall(rest))
    symbols = len(re.sub(r"[A-Za-z0-9_'\-\s]", "", rest))
    return max(1, math.ceil(cjk * 0.6 + words * 1.3 + symbols * 0.5))


def estimate_messages_tokens(messages: list[dict[str, str]]) -> int:
    """估算整段对话的上下文长度：内容 + 每条消息约 4 token 的角色/分隔开销。"""
    if not messages:
        return 0
    total = 0
    for message in messages:
        total += estimate_tokens(message.get("content") or "") + 4
    return total


# --------------------------------------------------------------------------- #
# 关键实体抽取：语义缓存的二次校验依赖它
# --------------------------------------------------------------------------- #

_ENTITY_PATTERNS: dict[str, tuple[str, ...]] = {
    # 地名：带行政后缀的，或常见城市名（「北京限行」这种不带后缀的也得认）
    "location": (
        r"[\u4e00-\u9fff]{2,8}(?:市|省|自治区|特别行政区|区|县|镇|路|街道)",
        r"(?:北京|上海|广州|深圳|杭州|成都|南京|武汉|西安|重庆|天津|苏州|长沙|青岛|郑州)",
        r"\b(?:Beijing|Shanghai|Shenzhen|Hangzhou|Tokyo|New York|London)\b",
    ),
    # 时间：绝对时间点与相对时间都算，避免「去年的答案」命中「今年的问题」
    "time": (
        r"\d{4}\s*年(?:\s*\d{1,2}\s*月)?(?:\s*\d{1,2}\s*日)?",
        r"\d{4}-\d{1,2}-\d{1,2}",
        r"\d{1,2}\s*月\s*\d{1,2}\s*日",
        r"(?:今天|明天|昨天|后天|本周|上周|下周|本月|上月|今年|去年|明年|当前|最新)",
        r"\b(?:today|tomorrow|yesterday|now|current|latest|this year|last year)\b",
    ),
    # 金额
    "money": (
        r"[¥￥$]\s?\d[\d,]*(?:\.\d+)?",
        r"\d[\d,]*(?:\.\d+)?\s?(?:元|块|万元|亿元|美元|人民币|欧元)",
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:USD|CNY|RMB|EUR)\b",
    ),
    # 产品型号：字母+数字组合，如 iPhone 15 Pro / GPT-4 / Qwen3-8B / A100
    "model_no": (r"\b[A-Za-z][A-Za-z]*(?:[-\s]?\d+)[A-Za-z0-9\-\.]*\b",),
}


def extract_entities(text: str) -> dict[str, list[str]]:
    """抽取关键实体（地名 / 时间 / 金额 / 型号）。

    README 的缓存事故就是「北京限行」命中「上海限行」—— 相似度 0.93 但实体不同。
    实体一致性检查是缓存的**强制二次校验**，不是可选优化。
    """
    if not text:
        return {}
    entities: dict[str, list[str]] = {}
    for category, patterns in _ENTITY_PATTERNS.items():
        found: list[str] = []
        for pattern in patterns:
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                value = match if isinstance(match, str) else "".join(match)
                value = re.sub(r"\s+", "", value)
                # 型号统一小写比较，避免 iPhone / iphone 被判为不一致
                key = value.lower() if category == "model_no" else value
                if key and key not in found:
                    found.append(key)
        if found:
            entities[category] = sorted(found)
    return entities


def entity_consistent(left: dict[str, list[str]], right: dict[str, list[str]]) -> bool:
    """两次请求的实体集合必须**完全一致**才算一致。

    注意是集合相等而不是「有交集」：一边有地名、另一边没有，属于典型的风险命中。
    """
    if not left and not right:
        return True
    keys = set(left) | set(right)
    for key in keys:
        if set(left.get(key) or []) != set(right.get(key) or []):
            return False
    return True


def length_ratio_ok(left: str, right: str, min_ratio: float = 0.5, max_ratio: float = 2.0) -> bool:
    """长度比校验：长度差一倍以上的 query，语义上通常不是同一个问题。"""
    a, b = len(left or ""), len(right or "")
    if a == 0 or b == 0:
        return a == b
    ratio = a / b
    return min_ratio <= ratio <= max_ratio


# --------------------------------------------------------------------------- #
# PII 检测：命中的请求既不写缓存，也优先本地推理
# --------------------------------------------------------------------------- #

_PII_PATTERNS: dict[str, str] = {
    "phone_cn": r"(?<!\d)1[3-9]\d{9}(?!\d)",
    "id_card_cn": r"(?<!\d)\d{17}[\dXx](?!\d)",
    "email": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "bank_card": r"(?<!\d)\d{16,19}(?!\d)",
}


def detect_pii(text: str) -> list[str]:
    """检测个人敏感信息。漏检的代价远大于误检（多花点钱而已）。"""
    if not text:
        return []
    hits = [name for name, pattern in _PII_PATTERNS.items() if re.search(pattern, text)]
    # 身份证 18 位必然命中银行卡的 16–19 位规则，去重避免同一条信息报两种类型
    if "id_card_cn" in hits and "bank_card" in hits:
        hits.remove("bank_card")
    return hits


# --------------------------------------------------------------------------- #
# JSON 提取：自评与 judge 都要求模型输出 JSON，模型却经常夹带解释
# --------------------------------------------------------------------------- #

def extract_json_object(text: str) -> dict[str, Any] | None:
    """从可能夹带废话的文本里抠出第一个完整 JSON 对象。

    做了三件事：剥 ```json 围栏 → 整段尝试 → 括号配对扫描。
    失败返回 None，调用方按「解析失败 = 未知」处理（路由侧一律保守升级）。
    """
    if not text:
        return None
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
    for candidate in _json_candidates(cleaned):
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _json_candidates(text: str) -> list[str]:
    """产出可能合法的 JSON 片段，按优先级排序。"""
    candidates = [text]
    start = text.find("{")
    if start == -1:
        return candidates

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidates.append(text[start : index + 1])
                break
    return candidates


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
