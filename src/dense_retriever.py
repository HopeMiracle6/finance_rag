from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.logging_utils import logger
from src.schema import RetrievalResult, TextChunk
from src.utils import ensure_dir, model_dump, normalize_scores, read_json, write_json


def _as_simple_metadata(chunk: TextChunk) -> dict[str, str | int | float | bool]:
    metadata: dict[str, str | int | float | bool] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source_file": chunk.source_file,
        "section_title": chunk.section_title or "",
    }
    if chunk.page is not None:
        metadata["page"] = chunk.page
    if chunk.page_start is not None:
        metadata["page_start"] = chunk.page_start
    if chunk.page_end is not None:
        metadata["page_end"] = chunk.page_end
    for key, value in chunk.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            metadata[f"meta_{key}"] = value
    return metadata


class EmbeddingBackend:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto", batch_size: int = 16) -> None:
        self.model_name = model_name
        self.device = None if device == "auto" else device
        self.batch_size = batch_size
        self.backend = "simple_hash"
        self.model: Any = None
        self.dim = 384
        self._load_model()

    def _load_model(self) -> None:
        local_only = os.getenv("FINANCE_RAG_ALLOW_MODEL_DOWNLOAD", "0") != "1"
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        if local_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {"device": self.device}
            if local_only:
                kwargs["local_files_only"] = True
            self.model = SentenceTransformer(self.model_name, **kwargs)
            self.backend = "sentence_transformers"
            if hasattr(self.model, "get_embedding_dimension"):
                dim = self.model.get_embedding_dimension()
            else:
                dim = self.model.get_sentence_embedding_dimension()
            self.dim = int(dim) if dim else self.dim
            logger.info(f"已加载 embedding 模型: {self.model_name}")
        except Exception as exc:
            logger.warning(f"embedding 模型加载失败，使用 simple_hash fallback: {exc}")
        finally:
            if local_only and old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.backend == "sentence_transformers" and self.model is not None:
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.astype("float32").tolist()
        return [self._hash_embedding(text) for text in texts]

    def _hash_embedding(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float32)
        if not text:
            return vector.tolist()
        chars = [char for char in text if not char.isspace()]
        for idx, char in enumerate(chars):
            bucket = (ord(char) * 131 + idx * 17) % self.dim
            vector[bucket] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector.tolist()


class DenseRetriever:
    def __init__(
        self,
        persist_dir: str | Path,
        embedding_model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 16,
        chroma_batch_size: int = 512,
        collection_name: str = "finance_chunks",
    ) -> None:
        self.persist_dir = Path(persist_dir)
        ensure_dir(self.persist_dir)
        self.collection_name = collection_name
        self.chroma_batch_size = max(1, int(chroma_batch_size))
        self.embedder = EmbeddingBackend(embedding_model_name, device=device, batch_size=batch_size)
        self.client: Any = None
        self.collection: Any = None
        self.memory_chunks: list[TextChunk] = []
        self.memory_embeddings: np.ndarray | None = None
        self._init_chroma()
        self._load_memory_index()

    def _init_chroma(self) -> None:
        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.warning(f"Chroma 初始化失败，使用本地 JSON 向量索引: {exc}")
            self.client = None
            self.collection = None

    @property
    def memory_index_path(self) -> Path:
        return self.persist_dir / "dense_index.json"

    def _load_memory_index(self) -> None:
        payload = read_json(self.memory_index_path, default=None)
        if not isinstance(payload, dict):
            return
        self.memory_chunks = [TextChunk(**item) for item in payload.get("chunks", [])]
        embeddings = payload.get("embeddings", [])
        self.memory_embeddings = np.asarray(embeddings, dtype=np.float32) if embeddings else None

    def build_index(self, chunks: list[TextChunk]) -> None:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        self.memory_chunks = chunks
        self.memory_embeddings = np.asarray(embeddings, dtype=np.float32)
        write_json(
            self.memory_index_path,
            {"chunks": [model_dump(chunk) for chunk in chunks], "embeddings": embeddings},
        )

        if self.collection is None:
            return
        try:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            ids = [chunk.chunk_id for chunk in chunks]
            metadatas = [_as_simple_metadata(chunk) for chunk in chunks]
            for start in range(0, len(chunks), self.chroma_batch_size):
                end = start + self.chroma_batch_size
                self.collection.add(
                    ids=ids[start:end],
                    documents=texts[start:end],
                    metadatas=metadatas[start:end],
                    embeddings=embeddings[start:end],
                )
        except Exception as exc:
            logger.warning(f"写入 Chroma 失败，保留本地 JSON 向量索引: {exc}")
            self.collection = None

    def search(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        query_embedding = self.embedder.encode([query])[0]
        if self.collection is not None:
            try:
                raw = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )
                return self._parse_chroma_results(raw)
            except Exception as exc:
                logger.warning(f"Chroma 查询失败，改用本地向量索引: {exc}")
        return self._search_memory(query_embedding, top_k=top_k)

    def _parse_chroma_results(self, raw: dict[str, Any]) -> list[RetrievalResult]:
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        scores = [max(0.0, 1.0 - float(distance)) for distance in distances]
        normalized_scores = normalize_scores(scores)
        results: list[RetrievalResult] = []
        for rank, chunk_id in enumerate(ids, start=1):
            metadata = metadatas[rank - 1] or {}
            page = metadata.get("page")
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")
            parsed_metadata = {
                "section_title": metadata.get("section_title", ""),
                "raw_score": scores[rank - 1],
            }
            for key, value in metadata.items():
                if isinstance(key, str) and key.startswith("meta_"):
                    parsed_metadata[key[5:]] = value
            results.append(
                RetrievalResult(
                    chunk_id=str(chunk_id),
                    doc_id=str(metadata.get("doc_id", "")),
                    source_file=str(metadata.get("source_file", "")),
                    page=int(page) if page not in (None, "") else None,
                    page_start=int(page_start) if page_start not in (None, "") else None,
                    page_end=int(page_end) if page_end not in (None, "") else None,
                    text=documents[rank - 1],
                    score=float(normalized_scores[rank - 1]),
                    retrieval_type="dense",
                    rank=rank,
                    metadata=parsed_metadata,
                )
            )
        return results

    def _search_memory(self, query_embedding: list[float], top_k: int) -> list[RetrievalResult]:
        if self.memory_embeddings is None or not self.memory_chunks:
            return []
        query_vector = np.asarray(query_embedding, dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vector))
        matrix = self.memory_embeddings
        if query_norm == 0 or matrix.size == 0:
            return []
        if matrix.shape[1] != query_vector.shape[0]:
            logger.warning(
                f"向量维度不一致，跳过 Dense 检索: index_dim={matrix.shape[1]}, query_dim={query_vector.shape[0]}"
            )
            return []
        scores = matrix @ query_vector / max(query_norm, 1e-8)
        scores = np.nan_to_num(scores)
        order = np.argsort(scores)[::-1][:top_k]
        selected_scores = [float(scores[i]) for i in order]
        normalized_scores = normalize_scores(selected_scores)

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(order, start=1):
            chunk = self.memory_chunks[int(idx)]
            score = normalized_scores[rank - 1]
            if math.isnan(score):
                score = 0.0
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
                    retrieval_type="dense",
                    rank=rank,
                    metadata={
                        **chunk.metadata,
                        "section_title": chunk.section_title,
                        "raw_score": selected_scores[rank - 1],
                    },
                )
            )
        return results
