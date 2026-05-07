from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.bm25_retriever import tokenize
from src.chunker import chunk_and_save
from src.config import load_config, resolve_path
from src.document_loader import load_and_save
from src.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_config.yaml")
    parser.add_argument("--raw-docs-dir", default=None)
    parser.add_argument("--documents-output", default=None)
    parser.add_argument("--chunks-output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    chunk_cfg = config["chunking"]

    raw_docs_dir = resolve_path(args.raw_docs_dir or paths["raw_docs_dir"])
    documents_output = resolve_path(args.documents_output or paths["documents_path"])
    chunks_output = resolve_path(args.chunks_output or paths["chunks_path"])
    bm25_corpus_output = resolve_path(paths["bm25_corpus_path"])

    documents = load_and_save(raw_docs_dir, documents_output)
    chunks = chunk_and_save(
        documents,
        str(chunks_output),
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"],
        min_chunk_size=chunk_cfg["min_chunk_size"],
    )
    write_json(
        bm25_corpus_output,
        {"chunk_ids": [chunk.chunk_id for chunk in chunks], "tokens": [tokenize(chunk.text) for chunk in chunks]},
    )
    print(f"documents: {len(documents)} -> {documents_output}")
    print(f"chunks: {len(chunks)} -> {chunks_output}")


if __name__ == "__main__":
    main()
