from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RawDocument(BaseModel):
    doc_id: str
    source_file: str
    file_type: str
    page: int | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextChunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_file: str
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_title: str | None = None
    text: str
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CninfoDocumentMetadata(BaseModel):
    doc_id: str
    file_name: str
    company_name: str | None = None
    stock_code: str | None = None
    report_type: str | None = None
    publish_date: str | None = None
    source_url: str | None = None
    pdf_path: str | None = None
    title: str | None = None
    source: str = "cninfo"
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeDocumentPage(BaseModel):
    doc_id: str
    file_name: str
    company_name: str | None = None
    stock_code: str | None = None
    report_type: str | None = None
    publish_date: str | None = None
    source_url: str | None = None
    page: int | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    file_name: str
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    company_name: str | None = None
    stock_code: str | None = None
    source_url: str | None = None
    report_type: str | None = None
    publish_date: str | None = None
    section_title: str | None = None
    token_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    doc_id: str
    source_file: str
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    text: str
    score: float
    retrieval_type: str
    rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGAnswer(BaseModel):
    question: str
    answer: str
    citations: list[RetrievalResult]
    retrieval_mode: str
    used_reranker: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalQuestion(BaseModel):
    id: str
    question: str
    answer_points: list[str] = Field(default_factory=list)
    gold_chunk_ids: list[str] = Field(default_factory=list)
    question_type: str = "fact_qa"
