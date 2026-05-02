"""
Tests for RAGGenerator, PromptBuilder, and AnswerProcessor.

Key demonstration: MockLLMGenerator implements ILLMGenerator and is injected
into RAGGenerator — no real model is loaded, no API call is made.
This shows how DIP makes the LLM layer fully testable in isolation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from typing import List
from interfaces import ILLMGenerator
from llm_handler import RAGGenerator, PromptBuilder, AnswerProcessor


# ---------------------------------------------------------------------------
# MockLLMGenerator — substitutes for the real LLMGenerator (demonstrates DIP)
# ---------------------------------------------------------------------------

class MockLLMGenerator(ILLMGenerator):
    """
    Fake LLM that returns a pre-configured response without loading any model.
    Implements ILLMGenerator so it is fully substitutable for LLMGenerator (LSP).
    """

    def __init__(self, response: str = "The printer connects via Wi-Fi settings menu."):
        self.response = response
        self.called_with: List[str] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.called_with.append(prompt)
        return self.response


class AlwaysUncertainMockLLM(ILLMGenerator):
    """Returns an uncertainty phrase to test low-confidence detection."""

    def generate(self, prompt: str, **kwargs) -> str:
        return "I don't know the answer to that question."


class EmptyResponseMockLLM(ILLMGenerator):
    """Returns an empty string to test short-response handling."""

    def generate(self, prompt: str, **kwargs) -> str:
        return ""


# ---------------------------------------------------------------------------
# PromptBuilder tests
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_contains_context(self):
        prompt = PromptBuilder.build_retrieval_prompt("Some context.", "What is this?")
        assert "Some context." in prompt

    def test_contains_query(self):
        prompt = PromptBuilder.build_retrieval_prompt("ctx", "What is the IP address?")
        assert "What is the IP address?" in prompt

    def test_prompt_is_string(self):
        prompt = PromptBuilder.build_retrieval_prompt("ctx", "query")
        assert isinstance(prompt, str)

    def test_empty_context_still_builds(self):
        prompt = PromptBuilder.build_retrieval_prompt("", "question")
        assert isinstance(prompt, str)


# ---------------------------------------------------------------------------
# AnswerProcessor tests
# ---------------------------------------------------------------------------

class TestAnswerProcessor:
    def test_clean_trailing_comma(self):
        result = AnswerProcessor.clean_answer("Answer ends here,")
        assert not result.endswith(",")

    def test_clean_trailing_semicolon(self):
        result = AnswerProcessor.clean_answer("Something;")
        assert not result.endswith(";")

    def test_removes_duplicate_lines(self):
        text = "Line one\nLine one\nLine two"
        result = AnswerProcessor.clean_answer(text)
        assert result.count("Line one") == 1

    def test_contains_uncertainty_true(self):
        assert AnswerProcessor.contains_uncertainty("I don't know the answer.")

    def test_contains_uncertainty_false(self):
        assert not AnswerProcessor.contains_uncertainty("The printer connects via Wi-Fi.")

    def test_process_confident_answer(self):
        long_answer = "The printer connects to Wi-Fi through the settings menu on the display."
        cleaned, is_confident = AnswerProcessor.process(long_answer)
        assert is_confident is True
        assert isinstance(cleaned, str)

    def test_process_uncertain_answer(self):
        _, is_confident = AnswerProcessor.process("I'm not sure about that.")
        assert is_confident is False

    def test_process_very_short_answer_not_confident(self):
        _, is_confident = AnswerProcessor.process("Yes.")
        assert is_confident is False


# ---------------------------------------------------------------------------
# RAGGenerator tests (all using MockLLMGenerator — no real model)
# ---------------------------------------------------------------------------

class TestRAGGenerator:
    def test_uses_injected_llm(self):
        mock = MockLLMGenerator()
        generator = RAGGenerator(llm=mock)
        assert generator.llm is mock

    def test_generate_answer_calls_llm(self):
        mock = MockLLMGenerator()
        generator = RAGGenerator(llm=mock)
        generator.generate_answer("Some context.", "What is the IP?", ["cite1"])
        assert len(mock.called_with) == 1

    def test_generate_answer_returns_tuple(self):
        mock = MockLLMGenerator()
        generator = RAGGenerator(llm=mock)
        result = generator.generate_answer("ctx", "query", ["cite"])
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_generate_answer_passes_citations_through(self):
        mock = MockLLMGenerator()
        generator = RAGGenerator(llm=mock)
        cites = ["Source: guide.txt (Chunk 1)"]
        _, returned_cites, _ = generator.generate_answer("context text here", "query?", cites)
        assert returned_cites == cites

    def test_empty_context_returns_fallback(self):
        mock = MockLLMGenerator()
        generator = RAGGenerator(llm=mock)
        answer, cites, confident = generator.generate_answer("", "What?", [])
        assert confident is False
        assert len(cites) == 0
        assert len(mock.called_with) == 0  # LLM never called for empty context

    def test_uncertain_llm_response_is_low_confidence(self):
        generator = RAGGenerator(llm=AlwaysUncertainMockLLM())
        _, _, confident = generator.generate_answer("ctx text here.", "query?", [])
        assert confident is False

    def test_empty_llm_response_uses_fallback(self):
        generator = RAGGenerator(llm=EmptyResponseMockLLM())
        answer, _, confident = generator.generate_answer("ctx text here.", "query?", [])
        assert confident is False
        assert len(answer) > 0
