from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.cninfo_client import copy_pdf_to_raw_dir, normalize_cninfo_item
from src.config import load_config, resolve_path
from src.knowledge_base_builder import build_knowledge_base
from src.schema import CninfoDocumentMetadata
from src.utils import read_jsonl, write_jsonl


DEFAULT_QLORA_PROJECT = Path(__file__).resolve().parents[3] / "中文金融公告解读助手"


def resolve_source_pdf(raw_pdf_path: str | None, qlora_project: Path) -> Path | None:
    if not raw_pdf_path:
        return None
    path = Path(raw_pdf_path)
    if path.is_absolute():
        return path
    return qlora_project / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_config.yaml")
    parser.add_argument("--qlora-project", default=str(DEFAULT_QLORA_PROJECT))
    parser.add_argument("--source-jsonl", default="data/raw/downloaded_announcements.jsonl")
    parser.add_argument("--metadata-output", default=None)
    parser.add_argument("--raw-pdf-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-copy-pdfs", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    chunk_cfg = config["chunking"]
    qlora_project = Path(args.qlora_project)
    source_jsonl = resolve_path(args.source_jsonl, qlora_project)
    metadata_output = resolve_path(args.metadata_output or paths["cninfo_metadata_path"])
    raw_pdf_dir = resolve_path(args.raw_pdf_dir or paths["raw_pdf_dir"])

    raw_items = read_jsonl(source_jsonl)
    records: list[CninfoDocumentMetadata] = []
    for index, item in enumerate(raw_items):
        if args.limit is not None and index >= args.limit:
            break
        source_pdf = resolve_source_pdf(item.get("pdf_path"), qlora_project)
        if not source_pdf or not source_pdf.exists():
            continue
        metadata = normalize_cninfo_item(item, pdf_path=source_pdf)
        if not args.no_copy_pdfs:
            metadata = copy_pdf_to_raw_dir(metadata, source_pdf, raw_pdf_dir)
        records.append(metadata)

    write_jsonl(metadata_output, records)
    print(f"cninfo metadata imported: {metadata_output}, records={len(records)}")
    print(f"raw pdf dir: {raw_pdf_dir}")

    if args.skip_build:
        return

    stats = build_knowledge_base(
        records,
        pages_output=resolve_path(paths["kb_pages_path"]),
        kb_chunks_output=resolve_path(paths["kb_chunks_path"]),
        chunks_output=resolve_path(paths["chunks_path"]),
        documents_output=resolve_path(paths["documents_path"]),
        chunk_size=chunk_cfg["chunk_size"],
        chunk_overlap=chunk_cfg["chunk_overlap"],
        min_chunk_size=chunk_cfg["min_chunk_size"],
    )
    print("knowledge base built")
    print(stats)


if __name__ == "__main__":
    main()
