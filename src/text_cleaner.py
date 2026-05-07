from __future__ import annotations

import regex as re


PAGE_NUM_RE = re.compile(r"^\s*(第\s*)?\d+\s*(页|/\s*\d+)?\s*$")
SPACE_RE = re.compile(r"[ \t\u3000]+")
CHINESE_OR_NUM_RE = re.compile(r"[\p{Han}\d%％）)]$")
LINE_START_RE = re.compile(r"^[\p{Han}\d（(]")
SENTENCE_END_RE = re.compile(r"[。！？；：.!?;:]$")


def _strip_page_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if not PAGE_NUM_RE.match(line.strip())]


def _merge_chinese_linebreaks(lines: list[str]) -> str:
    merged: list[str] = []
    current = ""
    for raw_line in lines:
        line = SPACE_RE.sub(" ", raw_line.strip())
        if not line:
            if current:
                merged.append(current)
                current = ""
            merged.append("")
            continue

        if not current:
            current = line
            continue

        should_merge = (
            not SENTENCE_END_RE.search(current)
            and CHINESE_OR_NUM_RE.search(current) is not None
            and LINE_START_RE.search(line) is not None
        )
        if should_merge:
            current += line
        else:
            merged.append(current)
            current = line

    if current:
        merged.append(current)
    return "\n".join(merged)


def clean_text(text: str, min_length: int = 1) -> str:
    """金融文本清洗：保留数字、日期、金额、百分比，只处理明显噪声。"""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = _strip_page_lines(text.split("\n"))
    text = _merge_chinese_linebreaks(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(SPACE_RE.sub(" ", line).strip() for line in text.split("\n"))
    text = text.strip()

    if len(re.sub(r"\s+", "", text)) < min_length:
        return ""
    return text
