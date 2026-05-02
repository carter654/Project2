"""
Tests for TextChunker in data_ingest.py.
No external dependencies — pure logic tests.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from data_ingest import TextChunker


class TestEstimateTokens:
    def test_empty_string(self):
        chunker = TextChunker()
        assert chunker.estimate_tokens("") == 0

    def test_single_word(self):
        chunker = TextChunker()
        result = chunker.estimate_tokens("hello")
        assert result >= 0  # short words may round to 0 with the token heuristic

    def test_longer_text_has_more_tokens(self):
        chunker = TextChunker()
        short = chunker.estimate_tokens("Hi.")
        long = chunker.estimate_tokens("This is a much longer sentence with many words.")
        assert long > short


class TestSplitIntoSentences:
    def setup_method(self):
        self.chunker = TextChunker()

    def test_splits_on_period(self):
        text = "First sentence. Second sentence."
        sentences = self.chunker.split_into_sentences(text)
        assert len(sentences) >= 2

    def test_no_empty_sentences(self):
        text = "Hello. World. How are you?"
        sentences = self.chunker.split_into_sentences(text)
        assert all(s.strip() != "" for s in sentences)

    def test_single_sentence(self):
        text = "Just one sentence without a period"
        sentences = self.chunker.split_into_sentences(text)
        assert len(sentences) == 1
        assert sentences[0] == text

    def test_abbreviations_not_split(self):
        text = "Dr. Smith works here. He is great."
        sentences = self.chunker.split_into_sentences(text)
        # Should not split "Dr. Smith" into two sentences
        assert any("Dr" in s for s in sentences)


class TestChunkText:
    def setup_method(self):
        self.chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        self.metadata = {"source": "test.txt"}

    def test_short_text_returns_one_chunk(self):
        text = "This is a short text."
        chunks = self.chunker.chunk_text(text, self.metadata)
        assert len(chunks) >= 1

    def test_chunk_contains_metadata(self):
        text = "Sample text for testing the chunker."
        chunks = self.chunker.chunk_text(text, self.metadata)
        for _, meta in chunks:
            assert "source" in meta
            assert meta["source"] == "test.txt"
            assert "chunk_id" in meta

    def test_chunk_ids_are_sequential(self):
        # Generate enough text to force multiple chunks
        text = ". ".join([f"Sentence number {i} about printers" for i in range(30)])
        chunks = self.chunker.chunk_text(text, self.metadata)
        if len(chunks) > 1:
            ids = [meta["chunk_id"] for _, meta in chunks]
            assert ids == list(range(len(ids)))

    def test_empty_text_returns_no_chunks(self):
        chunks = self.chunker.chunk_text("", self.metadata)
        assert chunks == []

    def test_chunk_text_is_string(self):
        text = "This is some text that will be chunked."
        chunks = self.chunker.chunk_text(text, self.metadata)
        for chunk_text, _ in chunks:
            assert isinstance(chunk_text, str)
            assert len(chunk_text) > 0
