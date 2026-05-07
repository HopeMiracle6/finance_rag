from __future__ import annotations

import regex as re

from src.schema import RawDocument, TextChunk
from src.utils import stable_id, write_jsonl


TITLE_RE = re.compile(
    r"^\s*((第?[一二三四五六七八九十]+[章节部分项、.．])|([一二三四五六七八九十]+[、.．])|(\d+(\.\d+)*[、.．]))\s*(.+)$"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；;!?])")


def estimate_token_count(text: str) -> int:
    return max(1, len(text.strip()) // 2)


def detect_title(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    match = TITLE_RE.match(line)
    if match and len(line) <= 60:
        return line
    if len(line) <= 32 and not re.search(r"[。！？；;!?]", line):
        title_words = ("摘要", "风险", "业绩", "原因", "说明", "结论", "财务", "提示")
        if any(word in line for word in title_words):
            return line
    return None


def split_by_sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        title = detect_title(line)
        if title:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = title
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def split_long_text(text: str, chunk_size: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
            continue
        sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]
        for sentence in sentences:
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                for start in range(0, len(sentence), chunk_size):
                    units.append(sentence[start : start + chunk_size])
    return units


def pack_units(units: list[str], chunk_size: int, chunk_overlap: int, min_chunk_size: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for unit in units:
        sep = "\n\n" if "\n" in unit else ""
        candidate = f"{current}{sep}{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(unit) > chunk_size:
            for start in range(0, len(unit), max(1, chunk_size - chunk_overlap)):
                piece = unit[start : start + chunk_size].strip()
                if len(piece) >= min_chunk_size:
                    chunks.append(piece)
            current = ""
        else:
            overlap = current[-chunk_overlap:] if current and chunk_overlap > 0 else ""
            current = f"{overlap}{unit}".strip() if overlap else unit

    if current:
        chunks.append(current)

    if len(chunks) > 1:
        return [chunk for chunk in chunks if len(chunk) >= min_chunk_size]
    return chunks if not chunks or len(chunks[0]) >= min_chunk_size else []


def chunk_document(
    document: RawDocument,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_size: int = 80,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    index = 0
    for section_title, section_text in split_by_sections(document.text):
        units = split_long_text(section_text, chunk_size=chunk_size)
        for text in pack_units(units, chunk_size=chunk_size, chunk_overlap=chunk_overlap, min_chunk_size=min_chunk_size):
            chunk_id = stable_id(document.doc_id, document.page, index, text[:80])
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    source_file=document.source_file,
                    page=document.page,
                    page_start=document.metadata.get("page_start", document.page),
                    page_end=document.metadata.get("page_end", document.page),
                    section_title=section_title,
                    text=text,
                    token_count=estimate_token_count(text),
                    metadata=document.metadata.copy(),
                )
            )
            index += 1
    return chunks


def chunk_documents(
    documents: list[RawDocument],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_size: int = 80,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                min_chunk_size=min_chunk_size,
            )
        )
    return chunks


def chunk_and_save(
    documents: list[RawDocument],
    output_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    min_chunk_size: int = 80,
) -> list[TextChunk]:
    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )
    write_jsonl(output_path, chunks)
    return chunks
