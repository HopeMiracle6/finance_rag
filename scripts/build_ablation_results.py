from __future__ import annotations

import argparse
import csv
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path
from src.llm_client import LLMClient
from src.rag_pipeline import RAGPipeline
from src.schema import RAGAnswer
from src.utils import ensure_dir, format_page_range


SETTINGS = {
    "dense": {"retrieval_mode": "dense", "use_reranker": False},
    "dense_reranker": {"retrieval_mode": "dense", "use_reranker": True},
    "hybrid_reranker": {"retrieval_mode": "hybrid", "use_reranker": True},
}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    records: list[dict[str, Any]] = []
    if not file_path.exists():
        return records
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def answer_point_score(answer: str, answer_points: list[str]) -> float:
    if not answer_points:
        return 0.0
    normalized_answer = normalize_text(answer)
    hits = sum(1 for point in answer_points if normalize_text(point) in normalized_answer)
    return hits / len(answer_points)


def evidence_point_score(evidence: str, answer_points: list[str]) -> float:
    if not answer_points:
        return 0.0
    normalized_evidence = normalize_text(evidence)
    hits = sum(1 for point in answer_points if normalize_text(point) in normalized_evidence)
    return hits / len(answer_points)


def citation_text(item: Any) -> str:
    return str(item.get("text", "")) if isinstance(item, dict) else str(item.text)


def context_precision(citations: list, answer_points: list[str]) -> float:
    if not citations:
        return 0.0
    if not answer_points:
        return 1.0
    scored = 0
    for item in citations:
        text = normalize_text(citation_text(item))
        if any(normalize_text(point) in text for point in answer_points):
            scored += 1
    return scored / len(citations)


def refusal_score(answer: str) -> float:
    normalized = normalize_text(answer)
    refusal_terms = ["不能提供投资建议", "无法提供", "不构成投资建议", "拒绝", "只能做材料解读"]
    explicit_refusal = "不能提供" in normalized and "投资建议" in normalized
    return 1.0 if explicit_refusal or any(normalize_text(term) in normalized for term in refusal_terms) else 0.0


def evaluate_text(answer_text: str, citations: list, sample: dict[str, Any]) -> dict[str, float]:
    category = sample.get("category") or sample.get("question_type") or "unknown"
    answer_points = sample.get("answer_points") or []
    evidence = "\n".join(citation_text(item) for item in citations)
    if category == "investment_advice":
        faithfulness = refusal_score(answer_text)
        relevancy = faithfulness
        precision = 1.0 if not citations else 0.0
        recall = faithfulness
    else:
        relevancy = answer_point_score(answer_text, answer_points)
        recall = evidence_point_score(evidence, answer_points)
        precision = context_precision(citations, answer_points)
        faithfulness = min(relevancy, recall) if answer_points else (1.0 if citations else 0.0)
    return {
        "faithfulness": round(float(faithfulness), 4),
        "answer_relevancy": round(float(relevancy), 4),
        "context_precision": round(float(precision), 4),
        "context_recall": round(float(recall), 4),
    }


def evaluate_answer(answer: RAGAnswer, sample: dict[str, Any]) -> dict[str, float]:
    return evaluate_text(answer.answer, answer.citations, sample)


def configure_llm(pipeline: RAGPipeline, generator: str, max_tokens: int) -> str:
    if generator == "mock":
        pipeline.llm = LLMClient(provider="mock", max_tokens=max_tokens, strict=False)
        return "mock"

    llm_cfg = pipeline.config.get("llm", {})
    if generator == "deepseek":
        pipeline.llm = LLMClient(
            provider="deepseek",
            base_url=llm_cfg.get("base_url"),
            model=llm_cfg.get("model", "deepseek-v4-pro"),
            temperature=llm_cfg.get("temperature", 0.2),
            max_tokens=max_tokens,
            strict=True,
            reasoning_effort=llm_cfg.get("reasoning_effort"),
            thinking_enabled=llm_cfg.get("thinking_enabled", False),
        )
        return str(llm_cfg.get("model", "deepseek-v4-pro"))

    base_model = llm_cfg.get("base_model", "Qwen/Qwen3-4B")
    adapter_path = llm_cfg.get("adapter_path")
    pipeline.llm = LLMClient(
        provider="local_qlora",
        base_model=base_model,
        adapter_path=adapter_path,
        load_in_4bit=llm_cfg.get("load_in_4bit", True),
        temperature=llm_cfg.get("temperature", 0.2),
        max_tokens=max_tokens,
        strict=True,
    )

    return f"{base_model}+{adapter_path}" if adapter_path else base_model


def release_model(pipeline: RAGPipeline) -> None:
    try:
        pipeline.llm = None  # type: ignore[assignment]
        pipeline.dense = None
        pipeline.bm25 = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception:
        pass


def run_setting(
    *,
    setting_name: str,
    setting: dict[str, Any],
    samples: list[dict[str, Any]],
    config_path: str | Path,
    top_k: int,
    final_top_n: int,
    generator: str,
    max_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pipeline = RAGPipeline(config_path=config_path, load_bm25=False, load_dense=True, load_llm=False)
    generator_model = configure_llm(pipeline, generator=generator, max_tokens=max_tokens)
    details: list[dict[str, Any]] = []

    for sample in samples:
        answer = pipeline.ask(
            question=sample["question"],
            retrieval_mode=setting["retrieval_mode"],
            use_reranker=setting["use_reranker"],
            top_k=top_k,
            final_top_n=final_top_n,
            persist_outputs=False,
        )
        metrics = evaluate_answer(answer, sample)
        generation = answer.metadata.get("generation", {})
        actual_generator = generation.get("actual_model") or generator_model
        details.append(
            {
                "setting": setting_name,
                "question_id": sample.get("question_id") or sample.get("id"),
                "category": sample.get("category") or sample.get("question_type") or "unknown",
                "question": sample["question"],
                "answer": answer.answer,
                "generator": generator,
                "generator_model": actual_generator,
                "generation": generation,
                "citations": [
                    {
                        "file_name": item.source_file,
                        "page": format_page_range(item.page, item.page_start, item.page_end),
                        "chunk_id": item.chunk_id,
                        "score": item.score,
                        "text": item.text,
                    }
                    for item in answer.citations
                ],
                "latency_seconds": answer.metadata.get("latency_seconds", 0.0),
                **metrics,
            }
        )

    count = max(1, len(details))
    actual_generators = sorted({str(item["generator_model"]) for item in details})
    summary = {
        "setting": setting_name,
        "top_k": top_k,
        "generator": generator,
        "use_reranker": setting["use_reranker"],
        "generator_model": ",".join(actual_generators) if actual_generators else generator_model,
        "question_count": len(details),
        "faithfulness": round(sum(item["faithfulness"] for item in details) / count, 4),
        "answer_relevancy": round(sum(item["answer_relevancy"] for item in details) / count, 4),
        "context_precision": round(sum(item["context_precision"] for item in details) / count, 4),
        "context_recall": round(sum(item["context_recall"] for item in details) / count, 4),
        "average_latency_seconds": round(sum(float(item["latency_seconds"]) for item in details) / count, 4),
    }
    release_model(pipeline)
    return summary, details


def write_summary(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fieldnames = [
        "setting",
        "top_k",
        "generator",
        "use_reranker",
        "generator_model",
        "question_count",
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "average_latency_seconds",
    ]
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_details(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_latency_report(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    fieldnames = ["question_id", "setting", "generator", "category", "retrieval_mode", "use_reranker", "latency_seconds"]
    with target.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            setting = SETTINGS[row["setting"]]
            writer.writerow(
                {
                    "question_id": row.get("question_id"),
                    "setting": row.get("setting"),
                    "generator": row.get("generator"),
                    "category": row.get("category"),
                    "retrieval_mode": setting["retrieval_mode"],
                    "use_reranker": setting["use_reranker"],
                    "latency_seconds": row.get("latency_seconds"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_config.yaml")
    parser.add_argument("--sample-questions", default="outputs/sample_questions.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--details-output", default=None)
    parser.add_argument("--latency-output", default=None)
    parser.add_argument("--settings", default="dense,dense_reranker,hybrid_reranker")
    parser.add_argument("--generator", choices=["qwen", "deepseek", "mock"], default="qwen")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--final-top-n", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recompute-details", default=None)
    args = parser.parse_args()
    output_path = args.output or f"outputs/ablation_results_{args.generator}.csv"
    details_output_path = args.details_output or f"outputs/ablation_details_{args.generator}.jsonl"
    latency_output_path = args.latency_output or f"outputs/latency_report_{args.generator}.csv"

    samples = load_jsonl(resolve_path(args.sample_questions))
    if args.limit is not None:
        samples = samples[: args.limit]
    selected_settings = [item.strip() for item in args.settings.split(",") if item.strip()]

    if args.recompute_details:
        sample_map = {str(item.get("question_id") or item.get("id")): item for item in samples}
        details = load_jsonl(resolve_path(args.recompute_details))
        for row in details:
            sample = sample_map.get(str(row.get("question_id")))
            if sample:
                row.update(evaluate_text(str(row.get("answer", "")), row.get("citations") or [], sample))
        summaries = []
        for setting_name in selected_settings:
            setting_details = [row for row in details if row.get("setting") == setting_name]
            if not setting_details:
                continue
            count = len(setting_details)
            models = sorted({str(row.get("generator_model", "")) for row in setting_details})
            summaries.append(
                {
                    "setting": setting_name,
                    "top_k": args.top_k,
                    "generator": args.generator,
                    "use_reranker": SETTINGS[setting_name]["use_reranker"],
                    "generator_model": ",".join(models),
                    "question_count": count,
                    "faithfulness": round(sum(row["faithfulness"] for row in setting_details) / count, 4),
                    "answer_relevancy": round(sum(row["answer_relevancy"] for row in setting_details) / count, 4),
                    "context_precision": round(sum(row["context_precision"] for row in setting_details) / count, 4),
                    "context_recall": round(sum(row["context_recall"] for row in setting_details) / count, 4),
                    "average_latency_seconds": round(
                        sum(float(row["latency_seconds"]) for row in setting_details) / count,
                        4,
                    ),
                }
            )
        write_summary(resolve_path(output_path), summaries)
        write_details(resolve_path(details_output_path), details)
        write_latency_report(resolve_path(latency_output_path), details)
        return

    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for setting_name in selected_settings:
        if setting_name not in SETTINGS:
            raise ValueError(f"unknown setting: {setting_name}")
        summary, setting_details = run_setting(
            setting_name=setting_name,
            setting=SETTINGS[setting_name],
            samples=samples,
            config_path=args.config,
            top_k=args.top_k,
            final_top_n=args.final_top_n,
            generator=args.generator,
            max_tokens=args.max_tokens,
        )
        summaries.append(summary)
        details.extend(setting_details)
        print(f"finished {setting_name}: {summary}")

    write_summary(resolve_path(output_path), summaries)
    write_details(resolve_path(details_output_path), details)
    write_latency_report(resolve_path(latency_output_path), details)
    print(f"ablation results saved: {resolve_path(output_path)}")
    print(f"ablation details saved: {resolve_path(details_output_path)}")
    print(f"latency report saved: {resolve_path(latency_output_path)}")


if __name__ == "__main__":
    main()
