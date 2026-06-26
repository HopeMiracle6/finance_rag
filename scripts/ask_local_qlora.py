from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.rag_pipeline import RAGPipeline
from src.utils import bool_arg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_config.yaml")
    parser.add_argument("--question", required=True)
    parser.add_argument("--retrieval-mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--use-reranker", default="true")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--final-top-n", type=int, default=5)
    args = parser.parse_args()

    os.environ["LLM_PROVIDER"] = "local_qlora"
    pipeline = RAGPipeline(config_path=args.config)
    answer = pipeline.ask(
        question=args.question,
        retrieval_mode=args.retrieval_mode,
        use_reranker=bool_arg(args.use_reranker),
        top_k=args.top_k,
        final_top_n=args.final_top_n,
    )

    print(f"问题：{answer.question}\n")
    print(f"回答：\n{answer.answer}\n")
    generation = answer.metadata.get("generation", {})
    print(
        "生成后端："
        f"{generation.get('actual_provider')} / {generation.get('actual_model')} / "
        f"{generation.get('backend')} / fallback={generation.get('fallback_used')}"
    )
    print("引用来源：")
    if not answer.citations:
        print("- 无")
    for item in answer.citations:
        page = item.page if item.page is not None else "无"
        print(f"- {item.source_file} | page={page} | chunk_id={item.chunk_id} | score={item.score:.4f}")

    print("\n检索到的证据片段：")
    for item in answer.citations:
        print(f"\n[{item.rank}] {item.chunk_id} ({item.retrieval_type}, score={item.score:.4f})")
        print(item.text[:500])


if __name__ == "__main__":
    main()
