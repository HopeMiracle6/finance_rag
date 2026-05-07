from __future__ import annotations

import pickle
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import jieba
except Exception:
    jieba = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

from src.schema import RetrievalResult, TextChunk
from src.utils import ensure_dir, model_dump, normalize_scores


def tokenize(text: str) -> list[str]:
    if jieba is not None:
        return [token.strip() for token in jieba.lcut(text) if token.strip()]
    tokens: list[str] = []
    for match in re.finditer(r"[A-Za-z0-9.%-]+|[\u4e00-\u9fff]", text):
        tokens.append(match.group(0))
    return tokens


class SimpleBM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.corpus_tokens = corpus_tokens
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))
        self.term_freqs = [Counter(doc) for doc in corpus_tokens]
        df: Counter[str] = Counter()
        for doc in corpus_tokens:
            df.update(set(doc))
        total_docs = max(1, len(corpus_tokens))
        self.idf = {term: math.log(1 + (total_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.corpus_tokens), dtype=np.float32)
        for idx, freqs in enumerate(self.term_freqs):
            doc_len = self.doc_len[idx]
            score = 0.0
            for token in query_tokens:
                freq = freqs.get(token, 0)
                if freq == 0:
                    continue
                idf = self.idf.get(token, 0.0)
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-8))
                score += idf * freq * (self.k1 + 1) / denom
            scores[idx] = score
        return scores


class BM25Retriever:
    def __init__(self, chunks: list[TextChunk] | None = None) -> None:
        self.chunks: list[TextChunk] = chunks or []
        self.corpus_tokens: list[list[str]] = []
        self.index = None
        if chunks:
            self.build_index(chunks)

    def build_index(self, chunks: list[TextChunk]) -> None:
        self.chunks = chunks
        self.corpus_tokens = [tokenize(chunk.text) for chunk in chunks]
        index_cls = BM25Okapi or SimpleBM25
        self.index = index_cls(self.corpus_tokens) if self.corpus_tokens else None

    def save(self, path: str | Path) -> None:
        file_path = Path(path)
        ensure_dir(file_path.parent)
        payload = {
            "chunks": [model_dump(chunk) for chunk in self.chunks],
            "corpus_tokens": self.corpus_tokens,
            "index": self.index,
        }
        with file_path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        file_path = Path(path)
        with file_path.open("rb") as f:
            payload = pickle.load(f)
        retriever = cls()
        retriever.chunks = [TextChunk(**item) for item in payload["chunks"]]
        retriever.corpus_tokens = payload["corpus_tokens"]
        retriever.index = payload["index"]
        return retriever

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        if not self.index or not self.chunks:
            return []

        query_tokens = tokenize(query)
        raw_scores = self.index.get_scores(query_tokens)
        overlap_scores = self._overlap_scores(query_tokens)
        raw_scores = raw_scores + overlap_scores * 1e-6
        order = np.argsort(raw_scores)[::-1][:top_k]
        selected_scores = [float(raw_scores[i]) for i in order]
        normalized_scores = normalize_scores(selected_scores)

        results: list[RetrievalResult] = []
        for rank, (idx, score) in enumerate(zip(order, normalized_scores), start=1):
            chunk = self.chunks[int(idx)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    source_file=chunk.source_file,
                    page=chunk.page,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    score=float(score),
                    retrieval_type="bm25",
                    rank=rank,
                    metadata={
                        **chunk.metadata,
                        "section_title": chunk.section_title,
                        "raw_score": selected_scores[rank - 1],
                    },
                )
            )
        return results

    def _overlap_scores(self, query_tokens: list[str]) -> np.ndarray:
        query_set = set(query_tokens)
        scores = np.zeros(len(self.corpus_tokens), dtype=np.float32)
        if not query_set:
            return scores
        for idx, tokens in enumerate(self.corpus_tokens):
            scores[idx] = len(query_set & set(tokens)) / len(query_set)
        return scores
