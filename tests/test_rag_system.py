"""
Tests for RAGSystem using MockEmbedder and MockVectorStore.

Demonstrates DIP in action: RAGSystem is tested without loading any real
embedding model or ChromaDB instance. The mocks implement IEmbedder and
IVectorStore so they are fully substitutable (LSP).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from typing import List, Tuple, Dict
from interfaces import IEmbedder, IVectorStore
from rag_system import RAGSystem


# ---------------------------------------------------------------------------
# Mock implementations
# ---------------------------------------------------------------------------

class MockEmbedder(IEmbedder):
    """Fake embedder — returns a fixed-length zero vector instantly."""

    DIM = 8

    def embed(self, text: str) -> List[float]:
        return [0.1] * self.DIM

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * self.DIM for _ in texts]


class MockVectorStore(IVectorStore):
    """In-memory fake vector store for testing."""

    def __init__(self, preset_docs: List[Tuple[str, Dict, float]] | None = None):
        self._indexed: List[Tuple[str, Dict]] = []
        self._search_results: List[Tuple[str, Dict, float]] = preset_docs or []

    def index_documents(self, chunks: List[Tuple[str, Dict]]) -> None:
        self._indexed.extend(chunks)

    def search(self, query: str, top_k: int) -> List[Tuple[str, Dict, float]]:
        return self._search_results[:top_k]

    def count(self) -> int:
        return len(self._indexed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRAGSystemInit:
    def test_uses_injected_vector_store(self):
        store = MockVectorStore()
        system = RAGSystem(vector_store=store)
        assert system.vector_store is store

    def test_indexes_documents_when_store_empty(self, tmp_path):
        """If the store is empty, RAGSystem should attempt to index documents."""
        store = MockVectorStore()
        # Patch DATA_DIR to an empty directory so no real files are needed
        import config
        original = config.DATA_DIR
        config.DATA_DIR = tmp_path
        try:
            system = RAGSystem(vector_store=store)
        finally:
            config.DATA_DIR = original

    def test_skips_indexing_when_store_populated(self):
        """Pre-populated store: RAGSystem must NOT re-index."""
        preset = [("chunk text", {"source": "doc.txt", "chunk_id": 0}, 0.9)]
        store = MockVectorStore(preset_docs=preset)
        # Manually add a doc so count() > 0
        store._indexed.append(("doc", {}))

        system = RAGSystem(vector_store=store)
        # Still only one document indexed (the one we pre-loaded)
        assert store.count() == 1


class TestRAGSystemRetrieve:
    def setup_method(self):
        self.preset = [
            ("Wi-Fi setup instructions.", {"source": "setup.txt", "chunk_id": 0}, 0.92),
            ("Driver troubleshooting steps.", {"source": "drivers.txt", "chunk_id": 1}, 0.75),
        ]
        store = MockVectorStore(preset_docs=self.preset)
        store._indexed.append(("dummy", {}))  # so count() > 0 → skip re-index
        self.system = RAGSystem(vector_store=store)

    def test_retrieve_returns_results(self):
        results = self.system.retrieve("How do I connect to Wi-Fi?", top_k=2)
        assert len(results) == 2

    def test_retrieve_respects_top_k(self):
        results = self.system.retrieve("printer", top_k=1)
        assert len(results) <= 1

    def test_retrieve_result_structure(self):
        results = self.system.retrieve("printer", top_k=2)
        for text, metadata, score in results:
            assert isinstance(text, str)
            assert isinstance(metadata, dict)
            assert isinstance(score, float)


class TestFormatContext:
    def setup_method(self):
        store = MockVectorStore()
        store._indexed.append(("dummy", {}))
        self.system = RAGSystem(vector_store=store)

    def test_empty_retrieved_returns_empty(self):
        context, citations = self.system.format_context([])
        assert context == ""
        assert citations == []

    def test_context_includes_document_markers(self):
        retrieved = [("Some printer text.", {"source": "guide.txt", "chunk_id": 3}, 0.85)]
        context, citations = self.system.format_context(retrieved)
        assert "[Document 1]" in context
        assert "Some printer text." in context

    def test_citations_contain_source(self):
        retrieved = [("Text.", {"source": "guide.txt", "chunk_id": 2}, 0.80)]
        _, citations = self.system.format_context(retrieved)
        assert len(citations) == 1
        assert "guide.txt" in citations[0]

    def test_multiple_documents_formatted(self):
        retrieved = [
            ("First.", {"source": "a.txt", "chunk_id": 0}, 0.9),
            ("Second.", {"source": "b.txt", "chunk_id": 1}, 0.8),
        ]
        context, citations = self.system.format_context(retrieved)
        assert "[Document 1]" in context
        assert "[Document 2]" in context
        assert len(citations) == 2
