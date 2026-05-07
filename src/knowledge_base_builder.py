from __future__ import annotations

from pathlib import Path
from typing import Iterable

import regex as re

from src.chunker import chunk_document, estimate_token_count
from src.logging_utils import logger
from src.schema import CninfoDocumentMetadata, KnowledgeChunk, KnowledgeDocumentPage, RawDocument, TextChunk
from src.text_cleaner import clean_text
from src.utils import project_root, read_jsonl, write_jsonl


PAGE_MARK_RE = re.compile(r"\[\[PAGE=(\d+)]]")


def resolve_pdf_path(pdf_path: str | Path | None, source_base_dir: str | Path | None = None) -> Path | None:
    if not pdf_path:
        return None
    raw_path = Path(pdf_path)
    if raw_path.is_absolute():
        return raw_path

    candidates: list[Path] = []
    if source_base_dir:
        candidates.append(Path(source_base_dir) / raw_path)
    candidates.append(project_root() / raw_path)
    candidates.append(raw_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else raw_path


def load_pdf_pages(
    metadata: CninfoDocumentMetadata,
    pdf_path: str | Path,
    min_length: int = 20,
) -> list[KnowledgeDocumentPage]:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("pymupdf is required for PDF parsing") from exc

    pages: list[KnowledgeDocumentPage] = []
    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"), min_length=min_length)
            if not text:
                continue
            pages.append(
                KnowledgeDocumentPage(
                    doc_id=metadata.doc_id,
                    file_name=metadata.file_name,
                    company_name=metadata.company_name,
                    stock_code=metadata.stock_code,
                    report_type=metadata.report_type,
                    publish_date=metadata.publish_date,
                    source_url=metadata.source_url,
                    page=page_number,
                    text=text,
                    metadata={
                        **metadata.metadata,
                        "title": metadata.title,
                        "source": metadata.source,
                        "pdf_path": str(pdf_path),
                    },
                )
            )
    return pages


def page_to_raw_document(page: KnowledgeDocumentPage) -> RawDocument:
    return RawDocument(
        doc_id=f"{page.doc_id}_p{page.page}",
        source_file=page.file_name,
        file_type="pdf",
        page=page.page,
        text=page.text,
        metadata={
            **page.metadata,
            "file_name": page.file_name,
            "company_name": page.company_name,
            "stock_code": page.stock_code,
            "report_type": page.report_type,
            "publish_date": page.publish_date,
            "source_url": page.source_url,
        },
    )


def pages_to_raw_document(pages: list[KnowledgeDocumentPage]) -> RawDocument:
    first = pages[0]
    page_numbers = [page.page for page in pages if page.page is not None]
    text = "\n\n".join(f"[[PAGE={page.page}]]\n{page.text}" for page in pages if page.page is not None)
    page_start = min(page_numbers) if page_numbers else None
    page_end = max(page_numbers) if page_numbers else None
    return RawDocument(
        doc_id=first.doc_id,
        source_file=first.file_name,
        file_type="pdf",
        page=page_start,
        text=text,
        metadata={
            **first.metadata,
            "file_name": first.file_name,
            "company_name": first.company_name,
            "stock_code": first.stock_code,
            "report_type": first.report_type,
            "publish_date": first.publish_date,
            "source_url": first.source_url,
            "page_start": page_start,
            "page_end": page_end,
        },
    )


def _page_boundaries(document_text: str) -> list[tuple[int, int, int]]:
    matches = list(PAGE_MARK_RE.finditer(document_text))
    boundaries: list[tuple[int, int, int]] = []
    for index, match in enumerate(matches):
        page = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document_text)
        boundaries.append((page, start, end))
    return boundaries


def _infer_page_range(start: int, end: int, boundaries: list[tuple[int, int, int]]) -> tuple[int | None, int | None]:
    pages = [page for page, page_start, page_end in boundaries if page_end > start and page_start < end]
    if not pages:
        return None, None
    return min(pages), max(pages)


def _strip_page_marks(text: str) -> str:
    return PAGE_MARK_RE.sub("", text).strip()


def chunk_continuous_document(
    document: RawDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_size: int = 80,
) -> list[TextChunk]:
    chunks = chunk_document(
        document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )
    enriched: list[TextChunk] = []
    current_page = document.metadata.get("page_start") or document.page
    for chunk in chunks:
        marker_matches = list(PAGE_MARK_RE.finditer(chunk.text))
        marker_pages = [int(match.group(1)) for match in marker_matches]
        if marker_pages:
            prefix = chunk.text[: marker_matches[0].start()].strip()
            page_start = current_page if prefix and current_page else marker_pages[0]
            page_end = marker_pages[-1]
            current_page = page_end
        else:
            page_start = current_page
            page_end = current_page

        if page_start and page_end and page_end - page_start > 5:
            page_start = page_end

        clean_chunk_text = _strip_page_marks(chunk.text)
        if len(clean_chunk_text) < min_chunk_size:
            continue
        metadata = {
            **chunk.metadata,
            "page_start": page_start,
            "page_end": page_end,
            "page_range": f"{page_start}-{page_end}" if page_start and page_end and page_start != page_end else page_start,
        }
        enriched.append(
            TextChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                source_file=chunk.source_file,
                page=page_start,
                page_start=page_start,
                page_end=page_end,
                section_title=chunk.section_title,
                text=clean_chunk_text,
                token_count=estimate_token_count(clean_chunk_text),
                metadata=metadata,
            )
        )
    return enriched


def text_chunk_to_knowledge_chunk(chunk: TextChunk) -> KnowledgeChunk:
    metadata = chunk.metadata
    return KnowledgeChunk(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        file_name=str(metadata.get("file_name") or chunk.source_file),
        page=chunk.page,
        page_start=chunk.page_start or metadata.get("page_start"),
        page_end=chunk.page_end or metadata.get("page_end"),
        company_name=metadata.get("company_name"),
        stock_code=metadata.get("stock_code"),
        source_url=metadata.get("source_url"),
        report_type=metadata.get("report_type"),
        publish_date=metadata.get("publish_date"),
        section_title=chunk.section_title,
        token_count=chunk.token_count,
        metadata=metadata.copy(),
    )


def build_knowledge_base(
    metadata_records: Iterable[CninfoDocumentMetadata],
    pages_output: str | Path,
    kb_chunks_output: str | Path,
    chunks_output: str | Path,
    documents_output: str | Path | None = None,
    source_base_dir: str | Path | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_size: int = 80,
    min_text_length: int = 20,
    limit: int | None = None,
) -> dict[str, int]:
    pages: list[KnowledgeDocumentPage] = []
    raw_documents: list[RawDocument] = []
    chunks: list[TextChunk] = []
    skipped = 0

    for index, metadata in enumerate(metadata_records):
        if limit is not None and index >= limit:
            break
        pdf_path = resolve_pdf_path(metadata.pdf_path, source_base_dir=source_base_dir)
        if not pdf_path or not pdf_path.exists():
            skipped += 1
            logger.warning(f"PDF not found, skipped: {metadata.pdf_path}")
            continue
        try:
            doc_pages = load_pdf_pages(metadata, pdf_path, min_length=min_text_length)
        except Exception as exc:
            skipped += 1
            logger.warning(f"PDF parse failed, skipped: {pdf_path}: {exc}")
            continue
        pages.extend(doc_pages)
        if not doc_pages:
            continue
        raw_document = pages_to_raw_document(doc_pages)
        raw_documents.append(raw_document)
        chunks.extend(
            chunk_continuous_document(
                raw_document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size,
            )
        )
    kb_chunks = [text_chunk_to_knowledge_chunk(chunk) for chunk in chunks]

    write_jsonl(pages_output, pages)
    if documents_output:
        write_jsonl(documents_output, raw_documents)
    write_jsonl(chunks_output, chunks)
    write_jsonl(kb_chunks_output, kb_chunks)

    return {
        "documents": len(raw_documents),
        "pages": len(pages),
        "chunks": len(chunks),
        "skipped": skipped,
    }


def load_cninfo_metadata(path: str | Path) -> list[CninfoDocumentMetadata]:
    return read_jsonl(path, model=CninfoDocumentMetadata)  # type: ignore[return-value]
