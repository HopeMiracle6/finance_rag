from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import resolve_path
from src.utils import ensure_dir


NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:亿元|万元|元|%|年|月|日)?")
ENTITY_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]{2,}(?:公司|集团|股份|银行|证券|科技|设计)")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def sample_category_map(samples: list[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in samples:
        question_id = item.get("question_id") or item.get("id")
        if question_id:
            mapping[str(question_id)] = item.get("category") or item.get("question_type") or "unknown"
        if item.get("question"):
            mapping[item["question"]] = item.get("category") or item.get("question_type") or "unknown"
    return mapping


def is_refusal_answer(answer: str) -> bool:
    normalized = answer.replace(" ", "")
    terms = ["不能提供投资建议", "不构成投资建议", "无法提供", "仅凭当前资料无法判断", "无法判断"]
    return any(term in normalized for term in terms)


def extract_claim_tokens(question: str, answer: str) -> set[str]:
    tokens = {item.strip() for item in NUMBER_RE.findall(answer) if item.strip()}
    tokens.update(item.strip() for item in ENTITY_RE.findall(question) if item.strip())
    return {token for token in tokens if len(token) >= 2}


def is_possible_unfaithful(record: dict[str, Any]) -> bool:
    answer = record.get("answer", "")
    citations = record.get("citations", [])
    if not citations and is_refusal_answer(answer):
        return False
    evidence = " ".join(
        f"{citation.get('file_name', '')} {citation.get('text', '')}"
        for citation in citations
    )
    tokens = extract_claim_tokens(record.get("question", ""), answer)
    if not tokens:
        return False
    missing = [token for token in tokens if token not in evidence]
    return bool(missing)


def evaluate(records: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    category_lookup = sample_category_map(samples)
    total = len(records)
    answer_lengths = [len(item.get("answer", "")) for item in records]
    citation_counts = [len(item.get("citations") or []) for item in records]
    latencies = [float(item.get("latency_seconds") or 0.0) for item in records]
    possible_unfaithful = [item for item in records if is_possible_unfaithful(item)]

    by_category: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "citation_count": 0, "latency": 0.0, "possible_unfaithful_count": 0})
    for item in records:
        category = category_lookup.get(item.get("question_id", ""), category_lookup.get(item.get("question", ""), "unknown"))
        bucket = by_category[category]
        bucket["count"] += 1
        bucket["citation_count"] += len(item.get("citations") or [])
        bucket["latency"] += float(item.get("latency_seconds") or 0.0)
        if is_possible_unfaithful(item):
            bucket["possible_unfaithful_count"] += 1

    by_category_result = {}
    for category, item in by_category.items():
        count = max(1, item["count"])
        by_category_result[category] = {
            "count": item["count"],
            "average_citation_count": round(item["citation_count"] / count, 4),
            "average_latency_seconds": round(item["latency"] / count, 4),
            "possible_unfaithful_count": item["possible_unfaithful_count"],
        }

    return {
        "total_questions": total,
        "average_answer_length": round(sum(answer_lengths) / total, 4) if total else 0,
        "average_citation_count": round(sum(citation_counts) / total, 4) if total else 0,
        "no_citation_ratio": round(sum(1 for count in citation_counts if count == 0) / total, 4) if total else 0,
        "average_latency_seconds": round(sum(latencies) / total, 4) if total else 0,
        "possible_unfaithful_count": len(possible_unfaithful),
        "possible_unfaithful_question_ids": [item.get("question_id") for item in possible_unfaithful],
        "by_category": by_category_result,
        "ragas_ready": False,
        "ragas_note": "当前为本地规则评估；后续可接入 RAGAS 计算 faithfulness、answer_relevancy、context_precision 等指标。",
    }


def write_json_report(path: str | Path, metrics: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown_report(path: str | Path, metrics: dict[str, Any]) -> None:
    lines = [
        "# RAG 本地评估报告",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 问题总数 | {metrics['total_questions']} |",
        f"| 平均回答长度 | {metrics['average_answer_length']} |",
        f"| 平均引用数量 | {metrics['average_citation_count']} |",
        f"| 无引用回答比例 | {metrics['no_citation_ratio']} |",
        f"| 平均延迟秒数 | {metrics['average_latency_seconds']} |",
        f"| possible_unfaithful 数量 | {metrics['possible_unfaithful_count']} |",
        "",
        "## 按问题类型统计",
        "",
        "| 类型 | 数量 | 平均引用数量 | 平均延迟秒数 | possible_unfaithful |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, item in metrics.get("by_category", {}).items():
        lines.append(
            f"| {category} | {item['count']} | {item['average_citation_count']} | "
            f"{item['average_latency_seconds']} | {item['possible_unfaithful_count']} |"
        )
    lines.extend(["", f"> {metrics['ragas_note']}"])
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-results", default="outputs/qa_results.jsonl")
    parser.add_argument("--sample-questions", default="outputs/sample_questions.jsonl")
    parser.add_argument("--json-output", default="outputs/eval_report.json")
    parser.add_argument("--md-output", default="outputs/eval_report.md")
    args = parser.parse_args()

    qa_results = load_jsonl(resolve_path(args.qa_results))
    samples = load_jsonl(resolve_path(args.sample_questions))
    metrics = evaluate(qa_results, samples)
    write_json_report(resolve_path(args.json_output), metrics)
    write_markdown_report(resolve_path(args.md_output), metrics)
    print(f"eval report saved: {resolve_path(args.json_output)}")
    print(f"eval report saved: {resolve_path(args.md_output)}")


if __name__ == "__main__":
    main()
