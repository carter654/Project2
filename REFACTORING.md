# Refactoring Report — Project 2

## SOLID Principles Applied

### 1. Dependency Inversion Principle (DIP)

> *High-level modules should not depend on low-level modules. Both should depend on abstractions.*

#### Problem in the Original Code

In Project 1, every high-level class directly instantiated its concrete dependencies:

```python
# Project 1 — rag_system.py (BEFORE)
class RAGSystem:
    def __init__(self):
        self.embedding_model = EmbeddingModel()          # hard-coded concrete class
        self.vector_store = VectorStore(
            embedding_model=self.embedding_model         # still a concrete class
        )
```

```python
# Project 1 — llm_handler.py (BEFORE)
class RAGGenerator:
    def __init__(self):
        self.llm = LLMGenerator()                        # hard-coded concrete class
```

This meant:
- Testing `RAGSystem` required loading a real 80 MB embedding model.
- Testing `RAGGenerator` required downloading GPT-2.
- Swapping to a different vector store (e.g., Pinecone) required editing `RAGSystem` directly.

#### Solution

A new `interfaces.py` module defines three abstract base classes:

```python
# interfaces.py (NEW)
from abc import ABC, abstractmethod

class IEmbedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]: ...
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...

class IVectorStore(ABC):
    @abstractmethod
    def index_documents(self, chunks: List[Tuple[str, Dict]]) -> None: ...
    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Tuple[str, Dict, float]]: ...
    @abstractmethod
    def count(self) -> int: ...

class ILLMGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...
```

High-level classes now **accept** their dependencies instead of creating them:

```python
# Project 2 — rag_system.py (AFTER)
class VectorStore(IVectorStore):
    def __init__(self, ..., embedder: Optional[IEmbedder] = None):
        self.embedder: IEmbedder = embedder or EmbeddingModel()  # abstraction, not concrete

class RAGSystem:
    def __init__(self, vector_store: Optional[IVectorStore] = None):
        self.vector_store: IVectorStore = vector_store or VectorStore()  # abstraction
```

```python
# Project 2 — llm_handler.py (AFTER)
class RAGGenerator:
    def __init__(self, llm: ILLMGenerator | None = None):
        self.llm: ILLMGenerator = llm or LLMGenerator()          # abstraction
```

**Measurable result:** All 47 unit tests run in under 0.5 s without downloading any
model, because the tests inject mock implementations that satisfy the interfaces:

```python
class MockLLMGenerator(ILLMGenerator):
    def generate(self, prompt: str, **kwargs) -> str:
        return "The printer connects via Wi-Fi settings menu."

generator = RAGGenerator(llm=MockLLMGenerator())   # zero model loading
```

---

### 2. Open/Closed Principle (OCP)

> *Software entities should be open for extension but closed for modification.*

#### Problem in the Original Code

Every time a new embedding technology or vector backend was needed, `RAGSystem` and
`VectorStore` had to be edited:

```python
# Project 1 — any swap required editing existing code
class VectorStore:
    def __init__(self, ..., embedding_model: Optional[EmbeddingModel] = None):
        self.embedding_model = embedding_model or EmbeddingModel()
        # ^ typed to the concrete EmbeddingModel — adding a new model means
        #   editing this class and the type annotation
```

#### Solution

Because `EmbeddingModel`, `VectorStore`, and `LLMGenerator` now implement
interfaces, a **new** implementation can be added without touching any existing class:

```python
# Example extension — no existing file is modified
class OpenAIEmbedder(IEmbedder):
    """Calls the OpenAI embeddings API instead of running locally."""
    def embed(self, text: str) -> List[float]:
        import openai
        return openai.embeddings.create(model="text-embedding-3-small",
                                        input=text).data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        ...

# Plug in without changing RAGSystem or VectorStore:
rag = RAGSystem(vector_store=VectorStore(embedder=OpenAIEmbedder()))
```

Likewise, swapping the LLM to any API provider or local Llama model only requires
implementing `ILLMGenerator.generate()` — `RAGGenerator` is untouched.

---

## Before-and-After Summary

| Component      | Before (Project 1)                      | After (Project 2)                          |
|----------------|-----------------------------------------|--------------------------------------------|
| `VectorStore`  | typed to `EmbeddingModel`               | typed to `IEmbedder`                       |
| `RAGSystem`    | creates `EmbeddingModel` + `VectorStore`| accepts `IVectorStore` via constructor     |
| `RAGGenerator` | creates `LLMGenerator`                  | accepts `ILLMGenerator` via constructor    |
| Testing        | requires model download (~80 MB–500 MB) | runs offline in < 0.5 s with mock classes  |
| Extensibility  | edit existing class to swap component   | implement new interface, no edits needed   |

---

## Other Improvements

- **Lazy imports** for `sentence_transformers`, `transformers`, and `chromadb` so the
  module can be imported in tests without triggering model downloads.
- **47 unit tests** across four test files covering `TextChunker`, `RAGSystem`,
  `RAGGenerator` / `AnswerProcessor` / `PromptBuilder`, and `CLIInterface`.
- `MockLLMGenerator`, `MockEmbedder`, and `MockVectorStore` demonstrate how
  interface-based design makes every layer independently testable.
