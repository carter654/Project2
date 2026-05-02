# RAG-Based Printer IT Helpdesk Bot (Project 2)

A refactored Retrieval-Augmented Generation (RAG) system that answers questions about printer setup, troubleshooting, and maintenance. This version applies SOLID design principles, adds a comprehensive test suite, and packages the application with Docker and GitHub Actions CI.

## What the Project Does

The bot:
1. Loads printer documentation from the `data/` directory
2. Chunks documents into overlapping segments for precise retrieval
3. Embeds text using HuggingFace `all-MiniLM-L6-v2` (384-dim vectors)
4. Stores and searches vectors in ChromaDB
5. Generates answers grounded in retrieved context via HuggingFace GPT-2
6. Displays answers with citations and confidence indicators via an interactive CLI

All processing runs locally — no cloud API keys required.

## Architecture Overview

```
data/*.txt
    ↓
[DocumentLoader → TextChunker]   data_ingest.py
    ↓
[EmbeddingModel (IEmbedder)]     rag_system.py
    ↓
[VectorStore (IVectorStore)]     rag_system.py
    ↓  ← inject via DIP
[RAGSystem]                      rag_system.py
    ↓
[RAGGenerator ← ILLMGenerator]   llm_handler.py
    ↓
[CLIInterface]                   main.py
```

UML diagrams (class + sequence) are in [diagrams/](diagrams/).

## Environment Setup

**Requirements:** Python 3.11+, pip

```bash
# Clone the repo
git clone <repo-url>
cd Project2

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

Available commands inside the bot:

| Command    | Description                              |
|------------|------------------------------------------|
| `question` | Ask anything about printers              |
| `/help`    | Show help message                        |
| `/debug`   | Show retrieved chunks and scores         |
| `/cite`    | Show citations from the last answer      |
| `/history` | Show last 10 queries                     |
| `/quit`    | Exit                                     |

## Running the Test Suite

```bash
python -m pytest tests/ -v
```

The tests run without downloading any ML models because all external services
(embedder, vector store, LLM) are replaced with lightweight mock implementations.

## Docker

### Build the image

```bash
docker build -t rag-printer-bot:latest .
```

### Run the container

```bash
docker run --rm -it \
    -v rag_printer_chroma:/app/chroma_db \
    -v rag_printer_hf_cache:/app/.cache/huggingface \
    rag-printer-bot:latest
```

Named volumes persist the ChromaDB index and downloaded HuggingFace model
weights across container restarts so you don't re-download on every run.

### Pull from Docker Hub (for evaluation)

```bash
docker pull <dockerhub-username>/rag-printer-bot:latest
docker run --rm -it \
    -v rag_printer_chroma:/app/chroma_db \
    -v rag_printer_hf_cache:/app/.cache/huggingface \
    <dockerhub-username>/rag-printer-bot:latest
```

## Configuration

Edit `config.py` to tune:

```python
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # swap for any IEmbedder
LLM_MODEL       = "gpt2"               # swap for any ILLMGenerator
TOP_K           = 5                    # chunks retrieved per query
SIMILARITY_THRESHOLD = 0.3             # min relevance score
CHUNK_SIZE      = 500                  # tokens per chunk
CHUNK_OVERLAP   = 75                   # overlap between chunks
DEBUG_MODE      = False                # verbose output
```

## Project Structure

```
Project2/
├── interfaces.py            # IEmbedder, IVectorStore, ILLMGenerator (DIP/OCP)
├── config.py                # Centralised settings
├── data_ingest.py           # DocumentLoader, TextChunker
├── rag_system.py            # EmbeddingModel, VectorStore, RAGSystem
├── llm_handler.py           # LLMGenerator, PromptBuilder, AnswerProcessor, RAGGenerator
├── main.py                  # CLIInterface entry point
├── tests/
│   ├── test_text_chunker.py
│   ├── test_rag_system.py
│   ├── test_llm_handler.py  # MockLLMGenerator demonstrated here
│   └── test_cli.py
├── diagrams/
│   ├── class_diagram.png
│   ├── sequence_diagram.png
│   └── generate_diagrams.py
├── data/                    # Printer documentation (.txt)
├── requirements.txt
├── Dockerfile
├── .github/workflows/tests.yml
├── README.md
└── REFACTORING.md
```

## CI Pipeline

GitHub Actions runs the full test suite on every push and pull request.
See [.github/workflows/tests.yml](.github/workflows/tests.yml).
