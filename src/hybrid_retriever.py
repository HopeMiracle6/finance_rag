from __future__ import annotations

from src.schema import RetrievalResult
from src.utils import normalize_scores


class HybridRetriever:
    def __init__(
        self,
        bm25_retriever,
        dense_retriever,
        rrf_k: int = 60,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def search(
        self,
        query: str,
        top_k: int = 10,
        bm25_top_k: int = 30,
        dense_top_k: int = 30,
        fusion_method: str = "rrf",
    ) -> list[RetrievalResult]:
        bm25_results = self.bm25_retriever.search(query, top_k=bm25_top_k) if self.bm25_retriever else []
        dense_results = self.dense_retriever.search(query, top_k=dense_top_k) if self.dense_retriever else []
        if fusion_method == "weighted_score":
            return self._weighted_score_fusion(bm25_results, dense_results, top_k=top_k)
        return self._rrf_fusion(bm25_results, dense_results, top_k=top_k)

    def _rrf_fusion(
        self,
        bm25_results: list[RetrievalResult],
        dense_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}
        for result in bm25_results:
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + self.bm25_weight / (self.rrf_k + result.rank)
            result_map.setdefault(result.chunk_id, result)
        for result in dense_results:
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + self.dense_weight / (self.rrf_k + result.rank)
            result_map.setdefault(result.chunk_id, result)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        normalized = normalize_scores([score for _, score in ordered])
        return [self._as_hybrid_result(result_map[chunk_id], normalized[idx], rank=idx + 1) for idx, (chunk_id, _) in enumerate(ordered)]

    def _weighted_score_fusion(
        self,
        bm25_results: list[RetrievalResult],
        dense_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        result_map: dict[str, RetrievalResult] = {}
        for result in bm25_results:
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + self.bm25_weight * result.score
            result_map.setdefault(result.chunk_id, result)
        for result in dense_results:
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + self.dense_weight * result.score
            result_map.setdefault(result.chunk_id, result)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        normalized = normalize_scores([score for _, score in ordered])
        return [self._as_hybrid_result(result_map[chunk_id], normalized[idx], rank=idx + 1) for idx, (chunk_id, _) in enumerate(ordered)]

    @staticmethod
    def _as_hybrid_result(result: RetrievalResult, score: float, rank: int) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            source_file=result.source_file,
            page=result.page,
            page_start=result.page_start,
            page_end=result.page_end,
            text=result.text,
            score=float(score),
            retrieval_type="hybrid",
            rank=rank,
            metadata=result.metadata.copy(),
        )
