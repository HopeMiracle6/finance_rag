from __future__ import annotations

import os
from pathlib import Path

try:
    import jieba
except Exception:
    jieba = None

from src.bm25_retriever import tokenize
from src.logging_utils import logger
from src.schema import RetrievalResult
from src.utils import normalize_scores


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        fallback_model_name: str = "BAAI/bge-reranker-base",
        device: str = "auto",
        enabled: bool = True,
        allow_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.device = None if device == "auto" else device
        self.enabled = enabled
        self.allow_fallback = allow_fallback
        self.model = None
        self.backend = "unloaded" if enabled else "disabled"
        if enabled:
            self._load_model()

    def _load_model(self) -> None:
        allow_download = os.getenv("FINANCE_RAG_ALLOW_MODEL_DOWNLOAD", "0") == "1"
        for name in [self.model_name, self.fallback_model_name]:
            if not allow_download and not Path(name).exists():
                message = f"未启用模型下载且本地不存在 reranker: {name}"
                if not self.allow_fallback:
                    raise FileNotFoundError(message)
                logger.warning(f"{message}，使用 keyword_overlap fallback")
                self.backend = "keyword_overlap"
                continue
            try:
                from FlagEmbedding import FlagReranker

                self.model = FlagReranker(name, use_fp16=True, device=self.device)
                self.backend = "FlagEmbedding"
                logger.info(f"已加载 reranker 模型: {name}")
                return
            except Exception as exc:
                if not self.allow_fallback:
                    raise RuntimeError(f"reranker 模型加载失败，且已禁用 fallback: {name}") from exc
                logger.warning(f"reranker 模型加载失败 {name}: {exc}")
                self.backend = "keyword_overlap"

    def rerank(self, query: str, results: list[RetrievalResult], top_n: int = 5) -> list[RetrievalResult]:
        if not results:
            return []
        if not self.enabled:
            return results[:top_n]

        if self.backend == "FlagEmbedding" and self.model is not None:
            try:
                pairs = [[query, result.text] for result in results]
                scores = self.model.compute_score(pairs, normalize=True)
                if isinstance(scores, float):
                    scores = [scores]
            except Exception as exc:
                if not self.allow_fallback:
                    raise RuntimeError("FlagEmbedding reranker 计算失败，且已禁用 fallback") from exc
                logger.warning(f"FlagEmbedding reranker 计算失败，使用 keyword_overlap fallback: {exc!r}")
                self.backend = "keyword_overlap"
                scores = [self._keyword_overlap_score(query, result.text) for result in results]
        else:
            scores = [self._keyword_overlap_score(query, result.text) for result in results]

        normalized = normalize_scores([float(score) for score in scores])
        enriched: list[RetrievalResult] = []
        for result, score, raw_score in zip(results, normalized, scores):
            metadata = result.metadata.copy()
            metadata["rerank_score"] = float(raw_score)
            metadata["reranker_backend"] = self.backend
            enriched.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    source_file=result.source_file,
                    page=result.page,
                    page_start=result.page_start,
                    page_end=result.page_end,
                    text=result.text,
                    score=float(score),
                    retrieval_type=f"{result.retrieval_type}+rerank",
                    rank=result.rank,
                    metadata=metadata,
                )
            )

        enriched.sort(key=lambda item: item.score, reverse=True)
        for idx, result in enumerate(enriched, start=1):
            result.rank = idx
        return enriched[:top_n]

    @staticmethod
    def _keyword_overlap_score(query: str, text: str) -> float:
        query_tokens = set(tokenize(query))
        text_tokens = set(tokenize(text))
        if not query_tokens or not text_tokens:
            return 0.0
        overlap = len(query_tokens & text_tokens)
        return overlap / max(1, len(query_tokens))
