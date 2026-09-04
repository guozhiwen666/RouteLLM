"""受限 YAML 子集解析器（零依赖，仅用于本项目自己的配置文件）。

为什么自己写而不引 PyYAML：项目只需要读一个结构固定的 routing.yaml，
为它引入一个完整 YAML 实现（连带锚点、多文档、标签等用不上的语法）不划算。

支持范围（超出即报错，绝不静默兜底）：
  - 行注释（`#` 位于行首或前面有空白）
  - `key: value` 键值
  - `- item` 列表，以及 `- key: value` 形式的列表项（后续缩进行同属该项）
  - 缩进嵌套块（不支持 Tab 缩进）
  - 行内数组 `[a, b]`
不支持的写法会抛 `YamlParseError` —— 配置文件语法错了比没配置更危险，
静默按默认值继续跑，等于让一个错误配置悄悄上线。
"""

from __future__ import annotations

from typing import Any

_TRUE = {"true", "yes", "on"}
_FALSE = {"false", "no", "off"}
_NULL = {"null", "~", ""}


class YamlParseError(ValueError):
    """YAML 子集解析失败。"""


def parse_yaml_subset(text: str) -> dict[str, Any]:
    """解析 YAML 子集文本，返回嵌套字典；顶层不是映射时返回空字典。"""
    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = _strip_comment(raw_line)
        if not stripped.strip():
            continue
        if "\t" in stripped[: len(stripped) - len(stripped.lstrip())]:
            raise YamlParseError("配置文件不支持 Tab 缩进，请改用空格")
        indent = len(stripped) - len(stripped.lstrip(" "))
        lines.append((indent, stripped.strip()))
    if not lines:
        return {}
    value, _ = _parse_block(lines, 0, lines[0][0])
    return value if isinstance(value, dict) else {}


def _strip_comment(line: str) -> str:
    """去掉行注释（不处理引号内的 #，本项目配置里没有这种写法）。"""
    index = line.find("#")
    if index == -1:
        return line
    # 只在 # 位于行首或前面有空白时视为注释，避免误伤 http://a#b 之类的值
    if index == 0 or line[index - 1] in " \t":
        return line[:index]
    return line


def _coerce_scalar(raw: str) -> Any:
    """把字符串标量还原成 Python 基础类型（引号 / 行内数组 / 布尔 / 空 / 数字 / 字符串）。"""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_coerce_scalar(part) for part in inner.split(",")]
    lowered = text.lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    if lowered in _NULL:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _looks_like_key(line: str) -> bool:
    """判断一行是否为 `key: value`（用 ': ' 切分，避免误伤 http:// 这类值）。"""
    return ": " in line or line.endswith(":")


def _parse_block(lines: list[tuple[int, str]], pos: int, indent: int) -> tuple[Any, int]:
    """按缩进递归解析一个块，返回 (值, 下一行位置)。"""
    if pos >= len(lines):
        return None, pos
    current = lines[pos][1]
    if current.startswith("-") and (len(current) == 1 or current[1] in " \t"):
        return _parse_sequence(lines, pos, indent)
    return _parse_mapping(lines, pos, indent)


def _parse_sequence(lines: list[tuple[int, str]], pos: int, indent: int) -> tuple[list[Any], int]:
    """解析 `- ` 开头的列表块，支持列表项为键值映射的写法。"""
    result: list[Any] = []
    while pos < len(lines) and lines[pos][0] == indent and lines[pos][1].startswith("-"):
        content = lines[pos][1][1:].strip()
        pos += 1
        if not content:
            # "- " 单独一行，值在后续缩进块里
            value, pos = (
                _parse_block(lines, pos, lines[pos][0])
                if pos < len(lines) and lines[pos][0] > indent
                else (None, pos)
            )
            result.append(value)
            continue

        if _looks_like_key(content):
            item: dict[str, Any] = {}
            key, _, value_text = content.partition(":")
            key, value_text = key.strip(), value_text.strip()
            item[key] = _coerce_scalar(value_text) if value_text else None
            if pos < len(lines) and lines[pos][0] > indent:
                sub, pos = _parse_block(lines, pos, lines[pos][0])
                if isinstance(sub, dict) and value_text:
                    item.update(sub)  # 后续行是同级的其它键
                elif not value_text:
                    item[key] = sub  # 值本身是个嵌套块
            result.append(item)
        else:
            result.append(_coerce_scalar(content))
    return result, pos


def _parse_mapping(lines: list[tuple[int, str]], pos: int, indent: int) -> tuple[dict[str, Any], int]:
    """解析 `key: value` 组成的映射块。"""
    result: dict[str, Any] = {}
    while pos < len(lines) and lines[pos][0] == indent:
        line = lines[pos][1]
        if line.startswith("-"):
            break
        if ": " not in line and not line.endswith(":"):
            raise YamlParseError(f"配置文件语法错误（第 {pos + 1} 个有效行）: 缺少冒号 -> {line!r}")
        key, _, value_text = line.partition(":")
        key, value_text = key.strip(), value_text.strip()
        pos += 1
        if value_text:
            result[key] = _coerce_scalar(value_text)
        elif pos < len(lines) and lines[pos][0] > indent:
            sub, pos = _parse_block(lines, pos, lines[pos][0])
            result[key] = sub
        else:
            result[key] = {}
    return result, pos
