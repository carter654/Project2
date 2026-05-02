"""
Tests for CLIInterface in main.py.

All heavy dependencies (RAGSystem, RAGGenerator) are replaced with lightweight
fakes so these tests run fast and without downloading any models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch
from main import CLIInterface


def make_cli() -> CLIInterface:
    """Return a CLIInterface with mocked rag_system and generator."""
    cli = CLIInterface()

    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = [
        ("Wi-Fi instructions here.", {"source": "setup.txt", "chunk_id": 0}, 0.9)
    ]
    mock_rag.format_context.return_value = (
        "[Document 1]\nWi-Fi instructions here.\n",
        ["Source: setup.txt (Chunk 0, relevance: 0.90)"],
    )

    mock_gen = MagicMock()
    mock_gen.generate_answer.return_value = (
        "Navigate to Settings > Network > Wi-Fi.",
        ["Source: setup.txt (Chunk 0, relevance: 0.90)"],
        True,
    )

    cli.rag_system = mock_rag
    cli.generator = mock_gen
    return cli


class TestHandleCommand:
    def test_quit_returns_false(self):
        cli = make_cli()
        assert cli.handle_command("/quit") is False

    def test_exit_returns_false(self):
        cli = make_cli()
        assert cli.handle_command("/exit") is False

    def test_help_returns_true(self, capsys):
        cli = make_cli()
        result = cli.handle_command("/help")
        assert result is True

    def test_unknown_command_returns_true(self, capsys):
        cli = make_cli()
        result = cli.handle_command("/unknown")
        assert result is True

    def test_cite_no_citations(self, capsys):
        cli = make_cli()
        cli.current_citations = []
        cli.handle_command("/cite")
        captured = capsys.readouterr()
        assert "No citations" in captured.out

    def test_cite_with_citations(self, capsys):
        cli = make_cli()
        cli.current_citations = ["Source: guide.txt (Chunk 1)"]
        cli.handle_command("/cite")
        captured = capsys.readouterr()
        assert "guide.txt" in captured.out

    def test_history_empty(self, capsys):
        cli = make_cli()
        cli.query_history = []
        cli.handle_command("/history")
        captured = capsys.readouterr()
        assert "No query history" in captured.out

    def test_history_with_entries(self, capsys):
        cli = make_cli()
        cli.query_history = [("How to connect?", "Go to settings.", ["cite"])]
        cli.handle_command("/history")
        captured = capsys.readouterr()
        assert "How to connect?" in captured.out


class TestFormatAnswer:
    def setup_method(self):
        self.cli = make_cli()

    def test_confident_answer_no_warning(self):
        result = self.cli.format_answer("Great answer.", ["cite"], True)
        assert "Low Confidence" not in result

    def test_low_confidence_shows_warning(self):
        result = self.cli.format_answer("Unsure answer.", ["cite"], False)
        assert "Low Confidence" in result

    def test_citations_included(self):
        result = self.cli.format_answer("Answer.", ["Source: a.txt"], True)
        assert "Source: a.txt" in result

    def test_no_citations_no_sources_section(self):
        result = self.cli.format_answer("Answer.", [], True)
        assert "SOURCES" not in result


class TestProcessQuery:
    def test_process_query_stores_history(self):
        cli = make_cli()
        cli.process_query("How do I connect to Wi-Fi?")
        assert len(cli.query_history) == 1

    def test_process_query_stores_last_query(self):
        cli = make_cli()
        cli.process_query("What is the IP address?")
        assert cli.last_query == "What is the IP address?"

    def test_process_query_no_results_prints_message(self, capsys):
        cli = make_cli()
        cli.rag_system.retrieve.return_value = []
        cli.process_query("Something completely irrelevant")
        captured = capsys.readouterr()
        assert "No relevant documents" in captured.out

    def test_process_query_updates_citations(self):
        cli = make_cli()
        cli.process_query("printer Wi-Fi")
        assert len(cli.current_citations) > 0
