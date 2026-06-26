from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from src.schema import RetrievalResult
from src.utils import ensure_dir, format_page_range, model_dump, project_root


def _outputs_dir() -> Path:
    return project_root() / "outputs"


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _citation_payload(result: RetrievalResult, index: int) -> dict[str, Any]:
    return {
        "source_id": f"S{index}",
        "file_name": result.source_file,
        "page": format_page_range(result.page, result.page_start, result.page_end),
        "chunk_id": result.chunk_id,
        "text": result.text,
    }


def _retrieved_chunk_payload(result: RetrievalResult) -> dict[str, Any]:
    payload = model_dump(result)
    payload["page"] = format_page_range(result.page, result.page_start, result.page_end)
    return payload


def save_qa_result(
    *,
    question_id: str,
    question: str,
    answer: str,
    citations: list[RetrievalResult],
    retrieved_chunks: list[RetrievalResult],
    retrieval_top_k: int,
    embedding_model: str,
    reranker_model: str,
    generator_model: str,
    generation: dict[str, Any],
    latency_seconds: float,
    created_at: str | None = None,
    output_path: str | Path | None = None,
) -> None:
    target = Path(output_path) if output_path else _outputs_dir() / "qa_results.jsonl"
    ensure_dir(target.parent)
    record = {
        "question_id": question_id,
        "question": question,
        "answer": answer,
        "citations": [_citation_payload(item, idx) for idx, item in enumerate(citations, start=1)],
        "retrieved_chunks": [_retrieved_chunk_payload(item) for item in retrieved_chunks],
        "retrieval_top_k": retrieval_top_k,
        "embedding_model": embedding_model,
        "reranker_model": reranker_model,
        "generator_model": generator_model,
        "generation": generation,
        "latency_seconds": round(float(latency_seconds), 4),
        "created_at": created_at or datetime.now().isoformat(timespec="seconds"),
    }
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_retrieved_contexts(
    *,
    question_id: str,
    question: str,
    retrieved_chunks: list[RetrievalResult],
    output_path: str | Path | None = None,
) -> None:
    target = Path(output_path) if output_path else _outputs_dir() / "retrieved_contexts.csv"
    ensure_dir(target.parent)
    fieldnames = ["question_id", "question", "rank", "score", "file_name", "page", "chunk_id", "chunk_text"]
    if target.exists() and target.stat().st_size > 0:
        with target.open("rb") as f:
            has_bom = f.read(3) == b"\xef\xbb\xbf"
        if not has_bom:
            content = target.read_text(encoding="utf-8")
            target.write_text(content, encoding="utf-8-sig")
    file_exists = target.exists() and target.stat().st_size > 0
    mode = "a" if file_exists else "w"
    encoding = "utf-8" if file_exists else "utf-8-sig"
    with target.open(mode, encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for item in retrieved_chunks:
            writer.writerow(
                {
                    "question_id": question_id,
                    "question": question,
                    "rank": item.rank,
                    "score": round(float(item.score), 6),
                    "file_name": item.source_file,
                    "page": format_page_range(item.page, item.page_start, item.page_end),
                    "chunk_id": item.chunk_id,
                    "chunk_text": _compact_text(item.text),
                }
            )
