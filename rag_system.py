"""
RAG System: Embedding and Vector-based Retrieval.

Refactored for Project 2:
- EmbeddingModel now implements IEmbedder  (OCP: swappable without changing VectorStore)
- VectorStore now implements IVectorStore and accepts IEmbedder  (DIP)
- RAGSystem accepts IVectorStore via constructor injection  (DIP)
"""

from typing import List, Tuple, Dict, Optional
from pathlib import Path

from interfaces import IEmbedder, IVectorStore
from config import (
    EMBEDDING_MODEL, CHROMADB_COLLECTION_NAME, CHROMADB_PERSIST_DIR,
    TOP_K, SIMILARITY_THRESHOLD, VERBOSE_LOGGING
)
from data_ingest import load_and_chunk_documents


class EmbeddingModel(IEmbedder):
    """
    HuggingFace sentence-transformers implementation of IEmbedder.
    Uses all-MiniLM-L6-v2 for fast, lightweight 384-dim embeddings.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"  -> Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]


class VectorStore(IVectorStore):
    """
    ChromaDB implementation of IVectorStore.

    DIP: depends on IEmbedder (abstraction), not EmbeddingModel (concrete class).
    OCP: any IEmbedder implementation can be injected; no code change needed here.
    """

    def __init__(
        self,
        collection_name: str = CHROMADB_COLLECTION_NAME,
        persist_dir: str = CHROMADB_PERSIST_DIR,
        embedder: Optional[IEmbedder] = None,
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.embedder: IEmbedder = embedder or EmbeddingModel()

        import chromadb  # lazy import — keeps tests fast
        print(f"Initializing ChromaDB at: {persist_dir}")
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"  -> Collection '{collection_name}' ready")

    def index_documents(self, chunks: List[Tuple[str, Dict]]) -> None:
        if not chunks:
            print("Warning: No chunks to index")
            return

        print(f"Indexing {len(chunks)} chunks into vector store...")
        texts = [c[0] for c in chunks]
        metadatas = [c[1] for c in chunks]

        ids = []
        for metadata in metadatas:
            source = metadata["source"].replace(".txt", "").replace(" ", "_")
            ids.append(f"{source}_chunk_{metadata['chunk_id']}")

        print("  -> Embedding texts...")
        embeddings = self.embedder.embed_batch(texts)

        print("  -> Storing in vector database...")
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        print(f"  -> Vector store now contains {self.collection.count()} chunks")

    def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, Dict, float]]:
        query_embedding = self.embedder.embed(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results["documents"] and results["documents"][0]:
            for doc, metadata, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                similarity = 1 - distance
                if similarity >= SIMILARITY_THRESHOLD:
                    retrieved.append((doc, metadata, similarity))
                elif VERBOSE_LOGGING:
                    print(f"  Skipping result below threshold (similarity={similarity:.3f})")

        if VERBOSE_LOGGING:
            print(f"Retrieved {len(retrieved)} relevant chunks (threshold={SIMILARITY_THRESHOLD})")
        return retrieved

    def count(self) -> int:
        return self.collection.count()


class RAGSystem:
    """
    Orchestrates document loading, embedding, and retrieval.

    DIP: accepts IVectorStore via constructor injection — does not import or
    instantiate any concrete vector-store or embedder class directly.
    OCP: swapping to a different vector store (e.g. Pinecone) requires only
    passing a new IVectorStore implementation; RAGSystem itself is unchanged.
    """

    def __init__(self, vector_store: Optional[IVectorStore] = None):
        print("=" * 60)
        print("Initializing RAG System")
        print("=" * 60)

        self.vector_store: IVectorStore = vector_store or VectorStore()
        self._initialize_index()

        print("=" * 60)
        print("RAG System ready for queries!")
        print("=" * 60)

    def _initialize_index(self) -> None:
        if self.vector_store.count() > 0:
            print("Vector store already contains data. Skipping re-indexing.")
            return

        chunks = load_and_chunk_documents()
        if chunks:
            self.vector_store.index_documents(chunks)
        else:
            print("Warning: No documents loaded. The RAG system is empty.")

    def retrieve(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, Dict, float]]:
        if VERBOSE_LOGGING:
            print(f"Retrieving top-{top_k} results for query: {query[:100]}...")
        return self.vector_store.search(query, top_k)

    def format_context(
        self, retrieved_chunks: List[Tuple[str, Dict, float]]
    ) -> Tuple[str, List[str]]:
        if not retrieved_chunks:
            return "", []

        context_parts = []
        citations = []

        for i, (text, metadata, score) in enumerate(retrieved_chunks, 1):
            context_parts.append(f"[Document {i}]\n{text}\n")
            source = metadata.get("source", "Unknown")
            chunk_id = metadata.get("chunk_id", 0)
            citations.append(f"Source: {source} (Chunk {chunk_id}, relevance: {score:.2f})")

        return "\n---\n".join(context_parts), citations


def create_rag_system(vector_store: Optional[IVectorStore] = None) -> RAGSystem:
    """Factory: create a RAGSystem, optionally injecting a custom IVectorStore."""
    return RAGSystem(vector_store=vector_store)
