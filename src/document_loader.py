from __future__ import annotations

from pathlib import Path

from src.schema import RawDocument
from src.text_cleaner import clean_text
from src.logging_utils import logger
from src.utils import stable_id, write_jsonl


SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def _load_pdf(path: Path, min_length: int) -> list[RawDocument]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("读取 PDF 需要安装 pymupdf") from exc

    records: list[RawDocument] = []
    with fitz.open(path) as pdf:
        for idx, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"), min_length=min_length)
            if not text:
                continue
            doc_id = stable_id(path.name, idx, text[:80])
            records.append(
                RawDocument(
                    doc_id=doc_id,
                    source_file=path.name,
                    file_type="pdf",
                    page=idx,
                    text=text,
                    metadata={"path": str(path)},
                )
            )
    return records


def _load_text_file(path: Path, min_length: int) -> list[RawDocument]:
    text = path.read_text(encoding="utf-8")
    text = clean_text(text, min_length=min_length)
    if not text:
        return []
    doc_id = stable_id(path.name, text[:120])
    return [
        RawDocument(
            doc_id=doc_id,
            source_file=path.name,
            file_type=path.suffix.lower().lstrip("."),
            page=None,
            text=text,
            metadata={"path": str(path)},
        )
    ]


def load_documents(raw_docs_dir: str | Path, min_length: int = 20) -> list[RawDocument]:
    raw_dir = Path(raw_docs_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"文档目录不存在: {raw_dir}")

    documents: list[RawDocument] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        logger.info(f"加载文档: {path}")
        if path.suffix.lower() == ".pdf":
            documents.extend(_load_pdf(path, min_length=min_length))
        else:
            documents.extend(_load_text_file(path, min_length=min_length))
    return documents


def load_and_save(raw_docs_dir: str | Path, output_path: str | Path, min_length: int = 20) -> list[RawDocument]:
    documents = load_documents(raw_docs_dir, min_length=min_length)
    write_jsonl(output_path, documents)
    logger.info(f"已写入 RawDocument: {output_path}, count={len(documents)}")
    return documents
