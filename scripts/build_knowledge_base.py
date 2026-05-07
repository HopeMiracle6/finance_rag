from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_path
from src.knowledge_base_builder import build_knowledge_base, load_cninfo_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_config.yaml")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--source-base-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pages-output", default=None)
    parser.add_argument("--kb-chunks-output", default=None)
    parser.add_argument("--chunks-output", default=None)
    parser.add_argument("--documents-output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    chunk_cfg = config["chunking"]

    metadata_path = resolve_path(args.metadata or paths["cninfo_metadata_path"])
    records = load_cninfo_metadata(metadata_path)
    stats = build_knowledge_base(
        records,
        pages_output=resolve_path(args.pages_output or paths["kb_pages_path"]),
        kb_chunks_output=resolve_path(args.kb_chunks_output or paths["kb_chunks_path"]),
        chunks_output=resolve_path(args.chunks_output or paths["chunks_path"]),
        documents_output=resolve_path(args.documents_output or paths["documents_path"]),
        source_base_dir=args.source_base_dir,
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"],
        min_chunk_size=chunk_cfg["min_chunk_size"],
        limit=args.limit,
    )
    print(f"knowledge base built from: {metadata_path}")
    print(stats)


if __name__ == "__main__":
    main()
