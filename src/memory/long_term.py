"""Long-term memory backed by ChromaDB for persistent research knowledge."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False


class LongTermMemory:
    """Persistent vector-backed memory for research findings.

    Stores research results so that similar future topics can benefit from
    past work. Each entry is a chunk of text with metadata.
    """

    def __init__(self, persist_dir: str = "./data/chroma", collection_name: str = "research_memory"):
        if not HAS_CHROMA:
            raise ImportError("chromadb is required for long-term memory. Install with: pip install chromadb")

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, topic: str, content: str, metadata: dict | None = None) -> str:
        """Store a research finding. Returns the entry ID."""
        entry_id = hashlib.md5(f"{topic}:{time.time()}".encode()).hexdigest()
        meta = metadata or {}
        meta["topic"] = topic
        meta["timestamp"] = time.time()

        # Split long content into chunks (max ~2000 chars per chunk)
        chunks = self._chunk_text(content, chunk_size=2000)
        ids = []
        documents = []
        metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{entry_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            chunk_meta = meta.copy()
            chunk_meta["chunk_index"] = i
            metadatas.append(chunk_meta)

        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)
        return entry_id

    def query(self, topic: str, n_results: int = 5) -> list[dict]:
        """Retrieve relevant past findings for a topic."""
        results = self._collection.query(query_texts=[topic], n_results=n_results)
        findings = []
        if results.get("ids") and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                findings.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "score": results["distances"][0][i] if results.get("distances") else 0,
                })
        return findings

    def delete_topic(self, topic: str) -> int:
        """Remove all entries for a topic. Returns count deleted."""
        results = self._collection.get(where={"topic": topic})
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 2000) -> list[str]:
        """Split text into roughly equal chunks at sentence boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        current = ""
        for sentence in text.replace('\n', ' ').split('. '):
            if len(current) + len(sentence) > chunk_size and current:
                chunks.append(current.strip())
                current = sentence + ". "
            else:
                current += sentence + ". "
        if current.strip():
            chunks.append(current.strip())
        return chunks
