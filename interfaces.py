"""
Abstract interfaces for the RAG system.

Applying the Dependency Inversion Principle (DIP): high-level modules depend on
these abstractions rather than on concrete implementations. Together with OCP,
new embedders, vector stores, or LLMs can be plugged in without modifying
any existing class.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict


class IEmbedder(ABC):
    """Abstract interface for text embedding models."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text string into a vector."""

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in one efficient pass."""


class IVectorStore(ABC):
    """Abstract interface for vector storage and similarity search."""

    @abstractmethod
    def index_documents(self, chunks: List[Tuple[str, Dict]]) -> None:
        """Index (text, metadata) tuples into the store."""

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Tuple[str, Dict, float]]:
        """Return the top_k most similar chunks as (text, metadata, score)."""

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents currently indexed."""


class ILLMGenerator(ABC):
    """Abstract interface for language-model text generation."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate and return text given a prompt string."""
