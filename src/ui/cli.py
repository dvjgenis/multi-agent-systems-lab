"""
Command Line Interface
Interactive CLI for the multi-agent research system.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from typing import Dict, Any
import yaml
import logging
from dotenv import load_dotenv

# Try sequential orchestrator first, fall back to AutoGen
try:
    from src.orchestrator import Orchestrator
    ORCHESTRATOR_TYPE = "sequential"
except ImportError:
    try:
        from src.autogen_orchestrator import AutoGenOrchestrator
        ORCHESTRATOR_TYPE = "autogen"
    except ImportError:
        ORCHESTRATOR_TYPE = None

# Load environment variables
load_dotenv()

class CLI:
    """
    Command-line interface for the research assistant.
    
    Features:
    - Interactive prompt loop
    - Clear agent trace display
    - Citation and source display
    - Safety event indicators
    - User commands (help, quit, clear, stats, traces)
    - Formatted output
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize CLI.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Setup logging
        self._setup_logging()

        # Initialize orchestrator
        try:
            if ORCHESTRATOR_TYPE == "sequential":
                self.orchestrator = Orchestrator(self.config)
                self.logger = logging.getLogger("cli")
                self.logger.info("Sequential orchestrator initialized successfully")
            elif ORCHESTRATOR_TYPE == "autogen":
                self.orchestrator = AutoGenOrchestrator(self.config)
                self.logger = logging.getLogger("cli")
                self.logger.info("AutoGen orchestrator initialized successfully")
            else:
                raise ImportError("No orchestrator available")
        except Exception as e:
            self.logger = logging.getLogger("cli")
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            raise

        self.running = True
        self.query_count = 0
        self.show_traces = False
        self.show_safety = True

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_format = log_config.get(
            "format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )

    async def run(self):
        """
        Main CLI loop.
        """
        self._print_welcome()

        while self.running:
            try:
                # Get user input
                query = input("\n🔍 Enter your research query (or 'help' for commands): ").strip()

                if not query:
                    continue

                # Handle commands
                if query.lower() in ['quit', 'exit', 'q']:
                    self._print_goodbye()
                    break
                elif query.lower() == 'help':
                    self._print_help()
                    continue
                elif query.lower() == 'clear':
                    self._clear_screen()
                    continue
                elif query.lower() == 'stats':
                    self._print_stats()
                    continue
                elif query.lower() == 'traces':
                    self.show_traces = not self.show_traces
                    print(f"\n{'✓' if self.show_traces else '✗'} Agent traces: {'ON' if self.show_traces else 'OFF'}")
                    continue
                elif query.lower() == 'safety':
                    self.show_safety = not self.show_safety
                    print(f"\n{'✓' if self.show_safety else '✗'} Safety events: {'ON' if self.show_safety else 'OFF'}")
                    continue

                # Process query
                print("\n" + "=" * 70)
                print("⏳ Processing your query...")
                print("=" * 70)
                
                try:
                    # Process through orchestrator (synchronous call)
                    result = self.orchestrator.process_query(query)
                    self.query_count += 1
                    
                    # Display result
                    self._display_result(result)
                    
                except Exception as e:
                    print(f"\n❌ Error processing query: {e}")
                    logging.exception("Error processing query")

            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user.")
                self._print_goodbye()
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                logging.exception("Error in CLI loop")

    def _print_welcome(self):
        """Print welcome message."""
        print("=" * 70)
        print(f"  🤖 {self.config['system']['name']}")
        print(f"  📚 Topic: {self.config['system']['topic']}")
        print("=" * 70)
        print("\nWelcome! Ask me anything about your research topic.")
        print("Type 'help' for available commands, or 'quit' to exit.\n")

    def _print_help(self):
        """Print help message."""
        print("\n" + "=" * 70)
        print("📖 AVAILABLE COMMANDS")
        print("=" * 70)
        print("  help     - Show this help message")
        print("  clear    - Clear the screen")
        print("  stats    - Show system statistics")
        print("  traces   - Toggle agent trace display")
        print("  safety   - Toggle safety event display")
        print("  quit     - Exit the application")
        print("\n" + "=" * 70)
        print("💡 TIP: Enter a research query to get started!")
        print("=" * 70)

    def _print_goodbye(self):
        """Print goodbye message."""
        print("\n" + "=" * 70)
        print("👋 Thank you for using the Multi-Agent Research Assistant!")
        print(f"📊 Processed {self.query_count} query/queries")
        print("=" * 70 + "\n")

    def _clear_screen(self):
        """Clear the terminal screen."""
        import os
        os.system('clear' if os.name == 'posix' else 'cls')

    def _print_stats(self):
        """Print system statistics."""
        print("\n" + "=" * 70)
        print("📊 SYSTEM STATISTICS")
        print("=" * 70)
        print(f"  Queries processed: {self.query_count}")
        print(f"  System: {self.config.get('system', {}).get('name', 'Unknown')}")
        print(f"  Topic: {self.config.get('system', {}).get('topic', 'Unknown')}")
        print(f"  Model: {self.config.get('models', {}).get('default', {}).get('name', 'Unknown')}")
        print(f"  Provider: {self.config.get('models', {}).get('default', {}).get('provider', 'Unknown')}")
        print(f"  Orchestrator: {ORCHESTRATOR_TYPE}")
        print("=" * 70)

    def _display_result(self, result: Dict[str, Any]):
        """Display query result with formatting."""
        print("\n" + "=" * 70)
        print("📝 RESPONSE")
        print("=" * 70)

        # Check for errors
        if "error" in result:
            print(f"\n❌ Error: {result['error']}")
            return

        # Check for safety violations
        metadata = result.get("metadata", {})
        if metadata.get("safety_violation"):
            print("\n⚠️  SAFETY ALERT")
            print("-" * 70)
            violations = metadata.get("violations", [])
            for violation in violations:
                reason = violation.get("reason", "Unknown violation")
                severity = violation.get("severity", "unknown")
                print(f"  [{severity.upper()}] {reason}")
            print("-" * 70 + "\n")

        # Display response
        response = result.get("response", "")
        print(f"\n{response}\n")

        # Display citations and sources
        citations = metadata.get("citations", [])
        sources = metadata.get("sources", [])
        
        if citations or sources:
            print("\n" + "-" * 70)
            print("📚 SOURCES & CITATIONS")
            print("-" * 70)
            
            # Display citations
            if citations:
                print("\nCitations:")
                for i, citation in enumerate(citations[:10], 1):
                    print(f"  [{i}] {citation}")
            
            # Display sources
            if sources:
                print("\nSources:")
                for i, source in enumerate(sources[:10], 1):
                    title = source.get("title", "Unknown")
                    url = source.get("url", "")
                    if url:
                        print(f"  [{i}] {title}")
                        print(f"      🔗 {url}")
                    else:
                        print(f"  [{i}] {title}")

        # Display metadata
        if metadata:
            print("\n" + "-" * 70)
            print("📊 METADATA")
            print("-" * 70)
            
            # Check orchestrator type for metadata structure
            if ORCHESTRATOR_TYPE == "sequential":
                print(f"  • Iterations: {metadata.get('iterations', 0)}")
                print(f"  • Sources gathered: {metadata.get('num_sources', 0)}")
                print(f"  • Citations: {len(citations)}")
                print(f"  • Elapsed time: {metadata.get('elapsed_time', 0):.2f}s")
                print(f"  • Status: {metadata.get('status', 'unknown')}")
            else:
                print(f"  • Messages exchanged: {metadata.get('num_messages', 0)}")
                print(f"  • Sources gathered: {metadata.get('num_sources', 0)}")
                print(f"  • Agents involved: {', '.join(metadata.get('agents_involved', []))}")

        # Display workflow trace if enabled
        if self.show_traces:
            self._display_workflow_trace(result)

        print("=" * 70 + "\n")
    
    def _display_workflow_trace(self, result: Dict[str, Any]):
        """Display workflow trace."""
        print("\n" + "-" * 70)
        print("🔍 WORKFLOW TRACE")
        print("-" * 70)
        
        # Check orchestrator type for trace structure
        if ORCHESTRATOR_TYPE == "sequential":
            workflow_trace = result.get("workflow_trace", [])
            if workflow_trace:
                for i, entry in enumerate(workflow_trace, 1):
                    phase = entry.get("phase", "unknown").upper()
                    message = entry.get("message", "")
                    data = entry.get("data", {})
                    
                    print(f"\n{i}. [{phase}] {message}")
                    if data:
                        for key, value in data.items():
                            if isinstance(value, str) and len(value) > 100:
                                value = value[:100] + "..."
                            print(f"   • {key}: {value}")
            else:
                print("\n  No workflow trace available")
        else:
            # AutoGen orchestrator - use conversation history
            conversation_history = result.get("conversation_history", [])
            if conversation_history:
                print("\nAgent Conversation:")
                for i, msg in enumerate(conversation_history, 1):
                    agent = msg.get("source", "Unknown")
                    content = msg.get("content", "")
                    preview = content[:200] + "..." if len(content) > 200 else content
                    preview = preview.replace("\n", " ")
                    print(f"\n{i}. [{agent}]")
                    print(f"   {preview}")
            else:
                print("\n  No conversation history available")

    def _extract_citations(self, result: Dict[str, Any]) -> list:
        """Extract citations/URLs from result."""
        citations = []
        metadata = result.get("metadata", {})
        
        # Get citations from metadata
        citations.extend(metadata.get("citations", []))
        
        # Extract URLs from response and conversation history
        import re
        response = result.get("response", "")
        urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', response)
        citations.extend(urls)
        
        # Extract from conversation history if available
        for msg in result.get("conversation_history", []):
            content = msg.get("content", "")
            urls = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', content)
            citations.extend(urls)
        
        # Deduplicate
        return list(dict.fromkeys(citations))[:10]  # Limit to top 10


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant CLI"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Run CLI
    cli = CLI(config_path=args.config)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
