from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bm25_retriever import BM25Retriever, tokenize
from src.config import load_config, resolve_path
from src.schema import TextChunk
from src.utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_config.yaml")
    parser.add_argument("--chunks", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    chunks_path = resolve_path(args.chunks or paths["chunks_path"])
    output_path = resolve_path(args.output or paths["bm25_index_path"])

    chunks = read_jsonl(chunks_path, model=TextChunk)
    retriever = BM25Retriever(chunks)  # type: ignore[arg-type]
    retriever.save(output_path)
    write_json(
        resolve_path(paths["bm25_corpus_path"]),
        {"chunk_ids": [chunk.chunk_id for chunk in chunks], "tokens": [tokenize(chunk.text) for chunk in chunks]},
    )
    print(f"BM25 index saved: {output_path}, chunks={len(chunks)}")


if __name__ == "__main__":
    main()
