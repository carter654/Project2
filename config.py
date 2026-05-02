"""
Configuration settings for the Printer IT Helpdesk RAG System.
Centralized configuration to make tuning and deployment easy.
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DB_DIR = PROJECT_ROOT / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DB_DIR.mkdir(exist_ok=True)

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # HuggingFace sentence-transformers model
EMBEDDING_DIM = 384  # Dimensionality of embeddings from all-MiniLM-L6-v2

# Vector Store Configuration
CHROMADB_COLLECTION_NAME = "printer_docs"
CHROMADB_PERSIST_DIR = str(CHROMA_DB_DIR)

# Text Chunking Configuration
CHUNK_SIZE = 500  # tokens (~300-450 words, ~1500-2250 characters)
CHUNK_OVERLAP = 75  # tokens overlap between consecutive chunks
# Why: Chunk size balances specificity (small enough to find relevant answers)
# with context retention (large enough for coherence). Overlap prevents losing
# information at chunk boundaries and maintains semantic continuity.

# Retrieval Configuration
TOP_K = 2  # Number of nearest neighbors to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score to consider a result relevant

# LLM Configuration
LLM_MODEL = "google/flan-t5-base"  # HuggingFace model for generation
LLM_MAX_LENGTH = 200  # Maximum tokens to generate per answer
LLM_TEMPERATURE = 0.7  # Creativity level (0.0 = deterministic, 1.0 = creative)
LLM_TOP_P = 0.95  # Nucleus sampling parameter

# Prompt Templates
SYSTEM_PROMPT = """You are an IT Helpdesk assistant specializing in printer troubleshooting. 
Answer questions ONLY using the provided context. If the context does not contain 
sufficient information to answer the question, respond with: 'I cannot find this information in the available documentation.'"""

RETRIEVAL_PROMPT_TEMPLATE = """Context from documentation:
{context}

User Question: {query}

Answer (based ONLY on the context above, do not use external knowledge):"""

# CLI Configuration
CLI_MAX_HISTORY = 20  # Number of queries to keep in memory
CITATION_FORMAT = "Source: {source} (Chunks {chunk_ids})"  # How to format citations

# Debug Settings
DEBUG_MODE = False  # Set to True to show retrieved chunks and scores
VERBOSE_LOGGING = False  # Set to True for detailed logs
