from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path
from src.dense_retriever import DenseRetriever
from src.schema import TextChunk
from src.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_config.yaml")
    parser.add_argument("--chunks", default=None)
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--chroma-batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    emb_cfg = config["embedding"]

    chunks_path = resolve_path(args.chunks or paths["chunks_path"])
    persist_dir = resolve_path(args.persist_dir or paths["chroma_persist_dir"])
    model_name = args.embedding_model or emb_cfg.get("model_name", "BAAI/bge-m3")
    chunks = read_jsonl(chunks_path, model=TextChunk)

    retriever = DenseRetriever(
        persist_dir=persist_dir,
        embedding_model_name=model_name,
        device=emb_cfg.get("device", "auto"),
        batch_size=emb_cfg.get("batch_size", 16),
        chroma_batch_size=args.chroma_batch_size or emb_cfg.get("chroma_batch_size", 512),
    )
    retriever.build_index(chunks)  # type: ignore[arg-type]
    print(f"Dense index saved: {persist_dir}, chunks={len(chunks)}, embedding_backend={retriever.embedder.backend}")


if __name__ == "__main__":
    main()
