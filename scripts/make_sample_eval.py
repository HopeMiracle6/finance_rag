from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path
from src.schema import TextChunk
from src.utils import read_jsonl, write_jsonl


def _find_chunk_ids(chunks: list[TextChunk], *keywords: str) -> list[str]:
    ids = [chunk.chunk_id for chunk in chunks if all(keyword in chunk.text for keyword in keywords)]
    if ids:
        return ids[:2]
    return [chunks[0].chunk_id] if chunks else []


def main() -> None:
    config = load_config("configs/rag_config.yaml")
    chunks = read_jsonl(resolve_path(config["paths"]["chunks_path"]), model=TextChunk)
    chunks = list(chunks)  # type: ignore[assignment]

    reason_ids = _find_chunk_ids(chunks, "订单增加")
    profit_ids = _find_chunk_ids(chunks, "净利润", "1.4")
    risk_ids = _find_chunk_ids(chunks, "风险")
    general_ids = _find_chunk_ids(chunks, "业绩预告")

    questions = [
        {
            "id": "q001",
            "question": "公司预计净利润同比增长的原因是什么？",
            "answer_points": ["订单增加", "生产效率提升", "原材料采购成本下降"],
            "gold_chunk_ids": reason_ids,
            "question_type": "fact_qa",
        },
        {
            "id": "q002",
            "question": "公司预计 2025 年净利润区间是多少？",
            "answer_points": ["1.4 亿元", "1.6 亿元", "40%", "60%"],
            "gold_chunk_ids": profit_ids,
            "question_type": "extraction",
        },
        {
            "id": "q003",
            "question": "这份公告的核心结论是什么？",
            "answer_points": ["业绩增长", "净利润"],
            "gold_chunk_ids": general_ids,
            "question_type": "summary",
        },
        {
            "id": "q004",
            "question": "这份公告提示了哪些风险？",
            "answer_points": ["经营环境变化", "原材料价格波动", "市场竞争加剧"],
            "gold_chunk_ids": risk_ids,
            "question_type": "fact_qa",
        },
        {
            "id": "q005",
            "question": "公司 2026 年现金分红方案是什么？",
            "answer_points": [],
            "gold_chunk_ids": [],
            "question_type": "out_of_context",
        },
        {
            "id": "q006",
            "question": "这家公司值得买入吗？",
            "answer_points": [],
            "gold_chunk_ids": [],
            "question_type": "investment_advice",
        },
    ]

    eval_dir = resolve_path("data/eval")
    write_jsonl(eval_dir / "eval_questions.jsonl", questions)
    write_jsonl(eval_dir / "gold_evidence.jsonl", [{"id": item["id"], "gold_chunk_ids": item["gold_chunk_ids"]} for item in questions])
    print(f"sample eval questions saved: {eval_dir / 'eval_questions.jsonl'}")


if __name__ == "__main__":
    main()
