from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from src.bm25_retriever import BM25Retriever
from src.config import load_config, resolve_path
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.llm_client import LLMClient
from src.output_writer import save_qa_result, save_retrieved_contexts
from src.prompt import build_rag_prompt, is_investment_advice_question
from src.reranker import Reranker
from src.schema import RAGAnswer, RetrievalResult


class RAGPipeline:
    def __init__(
        self,
        config_path: str | Path = "configs/rag_config.yaml",
        load_bm25: bool = True,
        load_dense: bool = True,
        load_llm: bool = True,
    ) -> None:
        self.config = load_config(config_path)
        paths = self.config["paths"]
        self.bm25: BM25Retriever | None = None
        self.dense: DenseRetriever | None = None
        self.reranker: Reranker | None = None

        bm25_path = resolve_path(paths["bm25_index_path"])
        if load_bm25 and bm25_path.exists():
            self.bm25 = BM25Retriever.load(bm25_path)

        if load_dense:
            emb_cfg = self.config.get("embedding", {})
            self.dense = DenseRetriever(
                persist_dir=resolve_path(paths["chroma_persist_dir"]),
                embedding_model_name=emb_cfg.get("model_name", "BAAI/bge-m3"),
                device=emb_cfg.get("device", "auto"),
                batch_size=emb_cfg.get("batch_size", 16),
                chroma_batch_size=emb_cfg.get("chroma_batch_size", 512),
                allow_embedding_fallback=emb_cfg.get("allow_fallback", False),
            )

        llm_cfg = self.config.get("llm", {})
        self.llm = None
        if load_llm:
            self.llm = LLMClient(
                provider=llm_cfg.get("provider", "openai_compatible"),
                base_url=llm_cfg.get("base_url"),
                api_key=llm_cfg.get("api_key"),
                model=llm_cfg.get("model"),
                base_model=llm_cfg.get("base_model"),
                adapter_path=llm_cfg.get("adapter_path"),
                load_in_4bit=llm_cfg.get("load_in_4bit", True),
                temperature=llm_cfg.get("temperature", 0.2),
                max_tokens=llm_cfg.get("max_tokens", 1024),
                strict=llm_cfg.get("strict", False),
                reasoning_effort=llm_cfg.get("reasoning_effort"),
                thinking_enabled=llm_cfg.get("thinking_enabled", False),
            )

    def retrieve(self, question: str, retrieval_mode: str = "hybrid", top_k: int = 30) -> list[RetrievalResult]:
        retrieval_cfg = self.config.get("retrieval", {})
        mode = retrieval_mode.lower()
        if mode == "bm25":
            return self.bm25.search(question, top_k=top_k) if self.bm25 else []
        if mode == "dense":
            return self.dense.search(question, top_k=top_k) if self.dense else []
        if mode == "hybrid":
            hybrid = HybridRetriever(
                self.bm25,
                self.dense,
                rrf_k=retrieval_cfg.get("rrf_k", 60),
                bm25_weight=retrieval_cfg.get("bm25_weight", 0.5),
                dense_weight=retrieval_cfg.get("dense_weight", 0.5),
            )
            return hybrid.search(
                question,
                top_k=top_k,
                bm25_top_k=retrieval_cfg.get("bm25_top_k", 30),
                dense_top_k=retrieval_cfg.get("dense_top_k", 30),
                fusion_method="rrf",
            )
        raise ValueError(f"不支持的检索模式: {retrieval_mode}")

    def ask(
        self,
        question: str,
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
        top_k: int = 30,
        final_top_n: int = 5,
        persist_outputs: bool = True,
    ) -> RAGAnswer:
        started_at = time.perf_counter()
        created_at = datetime.now().isoformat(timespec="seconds")
        question_id = uuid4().hex[:12]
        initial_results = self.retrieve(question, retrieval_mode=retrieval_mode, top_k=top_k)
        used_reranker = False
        final_results = initial_results[:final_top_n]

        if use_reranker and initial_results:
            reranker_cfg = self.config.get("reranker", {})
            if self.reranker is None:
                self.reranker = Reranker(
                    model_name=reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3"),
                    fallback_model_name=reranker_cfg.get("fallback_model_name", "BAAI/bge-reranker-base"),
                    device=reranker_cfg.get("device", "auto"),
                    enabled=reranker_cfg.get("enabled", True),
                    allow_fallback=reranker_cfg.get("allow_fallback", False),
                )
            final_results = self.reranker.rerank(question, initial_results, top_n=final_top_n)
            used_reranker = self.reranker.enabled

        if is_investment_advice_question(question):
            final_results = []

        prompt = build_rag_prompt(question, final_results)
        if self.llm is None:
            self.llm = LLMClient(provider="openai_compatible", strict=True)
        generation = self.llm.generate(prompt, question=question, citations=final_results)
        answer = generation.text
        latency_seconds = time.perf_counter() - started_at
        emb_cfg = self.config.get("embedding", {})
        reranker_cfg = self.config.get("reranker", {})
        generation_payload = generation.to_dict()
        if persist_outputs:
            save_qa_result(
                question_id=question_id,
                question=question,
                answer=answer,
                citations=final_results,
                retrieved_chunks=initial_results,
                retrieval_top_k=top_k,
                embedding_model=emb_cfg.get("model_name", "BAAI/bge-m3"),
                reranker_model=reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3") if use_reranker else "disabled",
                generator_model=generation.actual_model,
                generation=generation_payload,
                latency_seconds=latency_seconds,
                created_at=created_at,
            )
            save_retrieved_contexts(
                question_id=question_id,
                question=question,
                retrieved_chunks=initial_results,
            )
        return RAGAnswer(
            question=question,
            answer=answer,
            citations=final_results,
            retrieval_mode=retrieval_mode,
            used_reranker=used_reranker,
            metadata={
                "question_id": question_id,
                "initial_result_count": len(initial_results),
                "latency_seconds": round(latency_seconds, 4),
                "created_at": created_at,
                "generation": generation_payload,
                "retrieved_chunks": [item.model_dump() for item in initial_results],
            },
        )
