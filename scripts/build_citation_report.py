from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import resolve_path
from src.utils import ensure_dir


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    file_path = Path(path)
    if not file_path.exists():
        return records
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def render_report(records: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for record in records:
        citations = record.get("citations") or []
        source_items = []
        evidence_items = []
        for idx, citation in enumerate(citations, start=1):
            source_id = citation.get("source_id") or f"S{idx}"
            file_name = citation.get("file_name", "")
            page = citation.get("page", "无")
            chunk_id = citation.get("chunk_id", "")
            text = citation.get("text", "")
            source_items.append(
                f"<li><strong>[{html.escape(source_id)}]</strong> "
                f"{html.escape(file_name)} - 第 {html.escape(str(page))} 页 - "
                f"<code>{html.escape(chunk_id)}</code></li>"
            )
            evidence_items.append(f"<blockquote><mark>{html.escape(text)}</mark></blockquote>")

        if not source_items:
            source_items.append("<li>无引用来源</li>")
            evidence_items.append("<p class=\"empty\">当前回答没有引用片段。</p>")

        cards.append(
            "<section class=\"card\">"
            f"<div class=\"meta\">question_id: {html.escape(str(record.get('question_id', '')))}</div>"
            f"<h2>{html.escape(record.get('question', ''))}</h2>"
            "<h3>系统回答</h3>"
            f"<pre>{html.escape(record.get('answer', ''))}</pre>"
            "<h3>引用来源</h3>"
            f"<ul>{''.join(source_items)}</ul>"
            "<h3>原文引用片段</h3>"
            f"{''.join(evidence_items)}"
            "</section>"
        )

    if not cards:
        cards.append("<section class=\"card\"><h2>暂无问答记录</h2><p>请先运行 RAG 问答生成 qa_results.jsonl。</p></section>")

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>中文金融 RAG 引用溯源报告</title>
  <style>
    body {{ margin: 0; padding: 32px; background: #f6f7f9; color: #1f2937; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    header {{ max-width: 1080px; margin: 0 auto 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .sub {{ color: #6b7280; }}
    .card {{ max-width: 1080px; margin: 0 auto 18px; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 22px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06); }}
    .meta {{ color: #6b7280; font-size: 13px; margin-bottom: 8px; }}
    h2 {{ font-size: 21px; margin: 0 0 16px; }}
    h3 {{ font-size: 15px; margin: 18px 0 8px; color: #374151; }}
    pre {{ white-space: pre-wrap; line-height: 1.7; background: #f9fafb; border: 1px solid #eef0f3; border-radius: 6px; padding: 12px; }}
    li {{ margin: 8px 0; }}
    code {{ background: #f3f4f6; padding: 2px 5px; border-radius: 4px; }}
    blockquote {{ margin: 10px 0; padding: 12px 14px; border-left: 4px solid #2563eb; background: #f8fafc; }}
    mark {{ background: #fff3bf; padding: 2px 4px; border-radius: 4px; }}
    .empty {{ color: #6b7280; }}
  </style>
</head>
<body>
  <header>
    <h1>中文金融 RAG 引用溯源报告</h1>
    <div class="sub">每个卡片展示一个问题、系统回答、引用来源和原文片段。</div>
  </header>
  {''.join(cards)}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/qa_results.jsonl")
    parser.add_argument("--output", default="outputs/citation_report.html")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    records = load_jsonl(input_path)
    ensure_dir(output_path.parent)
    output_path.write_text(render_report(records), encoding="utf-8")
    print(f"citation report saved: {output_path}, records={len(records)}")


if __name__ == "__main__":
    main()
