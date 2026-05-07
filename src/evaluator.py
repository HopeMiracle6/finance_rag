from __future__ import annotations

from pathlib import Path

from src.rag_pipeline import RAGPipeline
from src.reranker import Reranker
from src.schema import EvalQuestion, RetrievalResult
from src.utils import ensure_dir, read_jsonl


def load_eval_questions(path: str | Path) -> list[EvalQuestion]:
    return read_jsonl(path, model=EvalQuestion)  # type: ignore[return-value]


def recall_at_k(results: list[RetrievalResult], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    retrieved = {result.chunk_id for result in results[:k]}
    return len(retrieved & set(gold_ids)) / len(set(gold_ids))


def hit_at_k(results: list[RetrievalResult], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    retrieved = {result.chunk_id for result in results[:k]}
    return 1.0 if retrieved & set(gold_ids) else 0.0


def mrr(results: list[RetrievalResult], gold_ids: list[str]) -> float:
    gold = set(gold_ids)
    if not gold:
        return 0.0
    for idx, result in enumerate(results, start=1):
        if result.chunk_id in gold:
            return 1.0 / idx
    return 0.0


def evaluate_retrieval_method(
    pipeline: RAGPipeline,
    questions: list[EvalQuestion],
    method: str,
    top_k: int = 30,
    final_top_n: int = 10,
) -> dict[str, float]:
    valid_questions = [question for question in questions if question.gold_chunk_ids]
    if not valid_questions:
        return {"Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0}

    metrics = {"Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0}
    for question in valid_questions:
        if method == "hybrid_reranker":
            results = pipeline.retrieve(question.question, retrieval_mode="hybrid", top_k=top_k)
            reranker_cfg = pipeline.config.get("reranker", {})
            reranker = Reranker(
                model_name=reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"),
                fallback_model_name=reranker_cfg.get("fallback_model_name", "BAAI/bge-reranker-base"),
                device=reranker_cfg.get("device", "auto"),
                enabled=reranker_cfg.get("enabled", True),
            )
            results = reranker.rerank(question.question, results, top_n=final_top_n)
        else:
            results = pipeline.retrieve(question.question, retrieval_mode=method, top_k=top_k)
        metrics["Recall@1"] += recall_at_k(results, question.gold_chunk_ids, 1)
        metrics["Recall@3"] += recall_at_k(results, question.gold_chunk_ids, 3)
        metrics["Recall@5"] += recall_at_k(results, question.gold_chunk_ids, 5)
        metrics["Recall@10"] += recall_at_k(results, question.gold_chunk_ids, 10)
        metrics["MRR"] += mrr(results, question.gold_chunk_ids)

    count = len(valid_questions)
    return {key: value / count for key, value in metrics.items()}


def write_retrieval_report(path: str | Path, rows: list[dict[str, object]]) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    headers = ["Method", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR"]
    lines = [
        "# Retrieval Evaluation Report",
        "",
        "| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {Method} | {Recall@1:.4f} | {Recall@3:.4f} | {Recall@5:.4f} | {Recall@10:.4f} | {MRR:.4f} |".format(
                **{key: row.get(key, 0.0) for key in headers}
            )
        )
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


REQUIRED_FIELDS = ("结论：", "依据：", "风险提示：", "引用来源：", "无法判断的部分：")


def evaluate_rag_answers(
    pipeline: RAGPipeline,
    questions: list[EvalQuestion],
    retrieval_mode: str = "hybrid",
    use_reranker: bool = True,
    top_k: int = 30,
    final_top_n: int = 5,
) -> dict[str, float]:
    if not questions:
        return {
            "format_rate": 0.0,
            "citation_rate": 0.0,
            "refusal_accuracy": 0.0,
            "keyword_coverage": 0.0,
            "evidence_hit_rate": 0.0,
        }

    format_hits = 0
    citation_hits = 0
    refusal_total = 0
    refusal_hits = 0
    keyword_scores: list[float] = []
    evidence_scores: list[float] = []

    for question in questions:
        answer = pipeline.ask(
            question.question,
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker,
            top_k=top_k,
            final_top_n=final_top_n,
        )
        text = answer.answer
        if all(field in text for field in REQUIRED_FIELDS):
            format_hits += 1
        if "引用来源：" in text and (answer.citations or "无" in text):
            citation_hits += 1

        if question.question_type in {"out_of_context", "investment_advice"}:
            refusal_total += 1
            if "仅凭当前资料无法判断" in text or "无法提供" in text or "不能提供投资建议" in text:
                refusal_hits += 1

        if question.answer_points:
            keyword_scores.append(sum(1 for point in question.answer_points if point in text) / len(question.answer_points))

        if question.gold_chunk_ids:
            retrieved = {item.chunk_id for item in answer.citations}
            evidence_scores.append(1.0 if retrieved & set(question.gold_chunk_ids) else 0.0)

    return {
        "format_rate": format_hits / len(questions),
        "citation_rate": citation_hits / len(questions),
        "refusal_accuracy": refusal_hits / refusal_total if refusal_total else 0.0,
        "keyword_coverage": sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0,
        "evidence_hit_rate": sum(evidence_scores) / len(evidence_scores) if evidence_scores else 0.0,
    }


def write_rag_report(path: str | Path, metrics: dict[str, float]) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    lines = [
        "# RAG Evaluation Report",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| 格式遵循率 | {metrics.get('format_rate', 0.0):.4f} |",
        f"| 引用存在率 | {metrics.get('citation_rate', 0.0):.4f} |",
        f"| 拒答准确率 | {metrics.get('refusal_accuracy', 0.0):.4f} |",
        f"| 关键词覆盖率 | {metrics.get('keyword_coverage', 0.0):.4f} |",
        f"| 证据命中率 | {metrics.get('evidence_hit_rate', 0.0):.4f} |",
    ]
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
