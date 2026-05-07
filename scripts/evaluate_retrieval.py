from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config, load_yaml, resolve_path
from src.evaluator import evaluate_retrieval_method, load_eval_questions, write_retrieval_report
from src.rag_pipeline import RAGPipeline


def ensure_eval_questions(path: Path) -> None:
    if not path.exists():
        subprocess.run([sys.executable, "scripts/make_sample_eval.py"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_config.yaml")
    parser.add_argument("--eval-config", default="configs/eval_config.yaml")
    args = parser.parse_args()

    rag_config = load_config(args.config)
    eval_config = load_yaml(args.eval_config)
    questions_path = resolve_path(eval_config["paths"]["questions_path"])
    ensure_eval_questions(questions_path)
    questions = load_eval_questions(questions_path)
    pipeline = RAGPipeline(config_path=args.config, load_llm=False)

    rows = []
    for method in ["bm25", "dense", "hybrid", "hybrid_reranker"]:
        metrics = evaluate_retrieval_method(pipeline, questions, method=method)
        rows.append({"Method": method.replace("_", " + ").title(), **metrics})

    report_path = resolve_path(eval_config["paths"]["retrieval_report_path"])
    write_retrieval_report(report_path, rows)
    print(f"retrieval report saved: {report_path}")


if __name__ == "__main__":
    main()
