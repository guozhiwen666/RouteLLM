"""文本基础工具：归一化、token 估算、实体抽取、PII 检测、JSON 提取。

这里放的都是「跟文本打交道的基础能力」，被启发式、自评、judge、语义缓存
等多个模块共用。归到这里后，各调用方不再各自维护一份正则，规则改一处生效全局。

依赖说明：token 估算是**估算**（加权字符法，无分词器依赖），
用于路由阈值与成本粗算；精确计量一律以后端返回的 usage 为准。
"""

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
