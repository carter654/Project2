"""
Module for data ingestion and text chunking.
Loads documents from the data/ folder and splits them into overlapping chunks.
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict
import re

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


class TextChunker:
    """
    Splits text into overlapping chunks while preserving sentence boundaries.
    
    Key design decisions:
    - Splits on sentence boundaries (periods, newlines) to avoid cutting mid-thought
    - Maintains overlap between chunks to preserve semantic continuity
    - Tracks source document and chunk index for citation purposes
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        """
        Initialize the chunker with configurable chunk size and overlap.
        
        Args:
            chunk_size: Target chunk size in tokens (~0.75 words per token)
            chunk_overlap: Number of overlapping tokens between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Rough estimate: 1 token ≈ 0.75 words (based on GPT tokenization)
        # So for 500 tokens, expect ~375 words ≈ 1875 characters
        self.words_per_chunk = int(chunk_size * 0.75)
        self.overlap_words = int(chunk_overlap * 0.75)

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count from character/word count.
        Simple heuristic: 1 token ≈ 4 characters + 1 per word
        More accurate for English prose.
        """
        words = len(text.split())
        chars = len(text)
        # Weighted estimate suitable for English text
        return int((chars / 4) + (words / 3)) // 2

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences, preserving sentence structure.
        Handles common abbreviations and edge cases.
        """
        # Replace common abbreviations to avoid false splits
        text = re.sub(r'(\b[A-Z])\. ', r'\1_DOT_ ', text)  # e.g., "U.S." -> "U_DOT_S."
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|etc|etc|vs|i\.e|e\.g)\. ', r'\1_DOT_ ', text, flags=re.IGNORECASE)
        
        # Split on periods, question marks, exclamation marks followed by space and capital letter
        # Also split on newlines with blank lines (paragraph breaks)
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*\n', text)
        
        # Restore abbreviations
        sentences = [s.replace('_DOT_', '.') for s in sentences]
        
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def chunk_text(self, text: str, metadata: Dict) -> List[Tuple[str, Dict]]:
        """
        Split text into overlapping chunks with metadata.
        
        Args:
            text: The input text to chunk
            metadata: Dictionary with at minimum 'source' (filename)
        
        Returns:
            List of (chunk_text, metadata) tuples
        """
        chunks = []
        
        # Split into sentences first
        sentences = self.split_into_sentences(text)
        
        # Group sentences into chunks
        current_chunk = []
        current_tokens = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = {
                    **metadata,
                    'chunk_id': chunk_id,
                    'token_count': current_tokens
                }
                chunks.append((chunk_text, chunk_metadata))
                
                # Start new chunk with overlap
                # Backtrack to add overlap from previous chunk
                overlap_words = int(self.chunk_size * 0.75)  # rough conversion
                words_to_keep = self.overlap_words
                all_words = chunk_text.split()
                
                # Keep last N words for overlap
                if len(all_words) > words_to_keep:
                    overlap_text = ' '.join(all_words[-words_to_keep:])
                    current_chunk = overlap_text.split()
                    current_tokens = self.estimate_tokens(overlap_text)
                else:
                    current_chunk = all_words
                    current_tokens = self.estimate_tokens(chunk_text)
                
                chunk_id += 1
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = {
                **metadata,
                'chunk_id': chunk_id,
                'token_count': current_tokens
            }
            chunks.append((chunk_text, chunk_metadata))
        
        return chunks


class DocumentLoader:
    """
    Loads text documents from the data/ directory and prepares them for chunking.
    Supports .txt files (PDF support can be added later with pdfplumber).
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        """Initialize loader with data directory."""
        self.data_dir = data_dir
        self.chunker = TextChunker()

    def load_documents(self) -> List[Tuple[str, Dict]]:
        """
        Load all documents from data/ folder and chunk them.
        
        Returns:
            List of (chunk_text, metadata) tuples ready for embedding
        """
        all_chunks = []
        
        # Find all .txt files in data directory
        if not self.data_dir.exists():
            print(f"Warning: Data directory {self.data_dir} does not exist.")
            return all_chunks
        
        txt_files = sorted(self.data_dir.glob('*.txt'))
        
        if not txt_files:
            print(f"No .txt files found in {self.data_dir}")
            return all_chunks
        
        print(f"Found {len(txt_files)} document(s) to load.")
        
        for file_path in txt_files:
            print(f"  Loading: {file_path.name}")
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Create metadata for this document
                metadata = {'source': file_path.name}
                
                # Chunk the document
                chunks = self.chunker.chunk_text(text, metadata)
                all_chunks.extend(chunks)
                
                print(f"    → Chunked into {len(chunks)} segments")
                
            except Exception as e:
                print(f"  Error reading {file_path.name}: {e}")
                continue
        
        print(f"\nTotal chunks created: {len(all_chunks)}")
        return all_chunks


def load_and_chunk_documents() -> List[Tuple[str, Dict]]:
    """
    Convenience function to load and chunk all documents.
    
    Returns:
        List of (chunk_text, metadata) tuples
    """
    loader = DocumentLoader()
    return loader.load_documents()


if __name__ == "__main__":
    # Test the chunking directly
    chunks = load_and_chunk_documents()
    if chunks:
        print("\n" + "="*60)
        print("Sample chunk preview:")
        print("="*60)
        sample_chunk, sample_metadata = chunks[0]
        print(f"Source: {sample_metadata['source']}")
        print(f"Chunk ID: {sample_metadata['chunk_id']}")
        print(f"Token Count: {sample_metadata.get('token_count', 'unknown')}")
        print(f"\nPreview (first 300 chars):\n{sample_chunk[:300]}...")
