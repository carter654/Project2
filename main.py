"""
Main CLI application for the Printer IT Helpdesk RAG Bot.
User-interactive command-line interface for querying the RAG system.
"""

from typing import List, Tuple
from rag_system import create_rag_system, RAGSystem
from llm_handler import create_generator, RAGGenerator
from config import DEBUG_MODE


class CLIInterface:
    """
    Interactive command-line interface for the RAG system.
    Manages user input, query history, and formatted output.
    """

    def __init__(self):
        """Initialize the CLI application."""
        self.rag_system: RAGSystem = None
        self.generator: RAGGenerator = None
        self.query_history: List[Tuple[str, str, List[str]]] = []
        self.current_citations: List[str] = []
        self.last_context = ""
        self.last_query = ""

    def initialize(self) -> None:
        """Initialize RAG system and LLM."""
        print("\n" + "="*70)
        print("  PRINTER IT HELPDESK - RAG-POWERED ASSISTANT")
        print("="*70)
        print("\nInitializing system components...")
        
        # Load RAG system (loads documents, builds index)
        self.rag_system = create_rag_system()
        
        # Initialize text generator
        print("\nInitializing text generation model...")
        self.generator = create_generator()
        
        print("\n" + "="*70)
        print("✓ System ready! Type '/help' for commands or ask your question.")
        print("="*70 + "\n")

    def print_welcome(self) -> None:
        """Print welcome message with usage instructions."""
        print("""
╔════════════════════════════════════════════════════════════════════╗
║                 PRINTER TROUBLESHOOTING BOT                        ║
║                                                                    ║
║  Ask any question about printer setup, drivers, connectivity,     ║
║  troubleshooting, or maintenance. Answers are based on the        ║
║  provided documentation.                                          ║
║                                                                    ║
║  Commands:                                                         ║
║    /help        - Show this help message                          ║
║    /debug       - Show retrieved chunks and relevance scores      ║
║    /cite        - Show citations from last query                  ║
║    /history     - Show query history                              ║
║    /quit        - Exit the program                                ║
╚════════════════════════════════════════════════════════════════════╝
        """)

    def handle_command(self, user_input: str) -> bool:
        """
        Handle special commands (starting with /).
        
        Returns:
            True if should continue, False if should exit
        """
        if user_input.lower() == "/quit" or user_input.lower() == "/exit":
            print("\nGoodbye! Thank you for using the Printer Helpdesk Bot.")
            return False
        
        elif user_input.lower() == "/help":
            self.print_welcome()
        
        elif user_input.lower() == "/debug":
            if DEBUG_MODE:
                print("\n=== DEBUG INFO ===")
                print(f"Last Query: {self.last_query}")
                print(f"Number of retrieved chunks: {len(self.current_citations)}")
                print(f"Citations:\n  " + "\n  ".join(self.current_citations))
                if len(self.last_context) > 300:
                    print(f"\nContext preview:\n{self.last_context[:300]}...")
                print("==================\n")
            else:
                print("Debug mode is disabled. See config.py to enable.")
        
        elif user_input.lower() == "/cite":
            if self.current_citations:
                print("\n=== CITATIONS ===")
                for i, citation in enumerate(self.current_citations, 1):
                    print(f"{i}. {citation}")
                print("==================\n")
            else:
                print("No citations available. Ask a question first.\n")
        
        elif user_input.lower() == "/history":
            if self.query_history:
                print("\n=== QUERY HISTORY ===")
                for i, (query, answer, cites) in enumerate(self.query_history[-10:], 1):
                    print(f"{i}. Q: {query[:50]}{'...' if len(query) > 50 else ''}")
                    print(f"   A: {answer[:50]}{'...' if len(answer) > 50 else ''}")
                print("====================\n")
            else:
                print("No query history yet.\n")
        
        else:
            print("Unknown command. Type '/help' for available commands.\n")
        
        return True

    def format_answer(self, answer: str, citations: List[str], is_confident: bool) -> str:
        """
        Format the answer for display with citations and confidence indicator.
        
        Args:
            answer: Generated answer text
            citations: List of source citations
            is_confident: Whether the answer is confident
        
        Returns:
            Formatted string for display
        """
        output = []
        
        # Add confidence indicator
        if not is_confident:
            output.append("[⚠ Low Confidence]")
        
        # Add the answer
        output.append(answer)
        
        # Add citations
        if citations:
            output.append("\n📚 SOURCES:")
            for citation in citations:
                output.append(f"   • {citation}")
        
        return "\n".join(output)

    def process_query(self, user_query: str) -> None:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            user_query: User's question
        """
        print("\n⏳ Processing query...")
        
        # Store for reference
        self.last_query = user_query
        
        # Retrieve relevant chunks
        retrieved = self.rag_system.retrieve(user_query, top_k=5)
        
        if not retrieved:
            print("\n❌ No relevant documents found for your query.\n")
            return
        
        # Format context and extract citations
        context, citations = self.rag_system.format_context(retrieved)
        self.last_context = context
        self.current_citations = citations
        
        # Generate answer
        answer, final_citations, is_confident = self.generator.generate_answer(
            context, user_query, citations
        )
        
        # Format and display
        formatted_output = self.format_answer(answer, final_citations, is_confident)
        print("\n" + "="*70)
        print("ANSWER:")
        print("="*70)
        print(formatted_output)
        print("="*70 + "\n")
        
        # Store in history
        self.query_history.append((user_query, answer, final_citations))

    def run(self) -> None:
        """Run the interactive CLI loop."""
        self.initialize()
        self.print_welcome()
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = input("📝 You: ").strip()
                    
                    if not user_input:
                        continue
                    
                    # Handle commands
                    if user_input.startswith("/"):
                        should_continue = self.handle_command(user_input)
                        if not should_continue:
                            break
                    else:
                        # Process as query
                        self.process_query(user_input)
                
                except KeyboardInterrupt:
                    print("\n\nInterrupted by user.")
                    break
                except EOFError:
                    raise
                except Exception as e:
                    print(f"\n❌ Error: {e}")
                    print("Please try again or type '/help' for commands.\n")
        
        except EOFError:
            # Graceful exit when stdin closes
            print("\nSession ended.")


def main():
    """Entry point for the application."""
    cli = CLIInterface()
    cli.run()


if __name__ == "__main__":
    main()
