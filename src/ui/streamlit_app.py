"""
Streamlit Web Interface
Web UI for the multi-agent research system.

Run with: streamlit run src/ui/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import time
import asyncio
import yaml
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

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


def load_config():
    """Load configuration file."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def initialize_session_state():
    """Initialize Streamlit session state."""
    if 'history' not in st.session_state:
        st.session_state.history = []

    if 'orchestrator' not in st.session_state:
        config = load_config()
        # Initialize orchestrator
        try:
            if ORCHESTRATOR_TYPE == "sequential":
                st.session_state.orchestrator = Orchestrator(config)
            elif ORCHESTRATOR_TYPE == "autogen":
                st.session_state.orchestrator = AutoGenOrchestrator(config)
            else:
                st.session_state.orchestrator = None
                st.error("No orchestrator available. Please check your imports.")
        except Exception as e:
            st.error(f"Failed to initialize orchestrator: {e}")
            st.session_state.orchestrator = None

    if 'show_traces' not in st.session_state:
        st.session_state.show_traces = False

    if 'show_safety_log' not in st.session_state:
        st.session_state.show_safety_log = False
    
    if 'safety_manager' not in st.session_state:
        try:
            from src.guardrails.safety_manager import SafetyManager
            config = load_config()
            safety_config = config.get("safety", {})
            if safety_config.get("enabled", True):
                safety_config_full = {
                    **safety_config,
                    "topic": config.get("system", {}).get("topic", "HCI Research"),
                    "logging": config.get("logging", {})
                }
                st.session_state.safety_manager = SafetyManager(safety_config_full)
            else:
                st.session_state.safety_manager = None
        except Exception as e:
            st.session_state.safety_manager = None


def process_query(query: str, orchestrator=None) -> Dict[str, Any]:
    """
    Process a query through the orchestrator.
    
    Args:
        query: Research query to process
        orchestrator: Orchestrator instance (if None, gets from session state)
        
    Returns:
        Result dictionary with response, citations, and metadata
    """
    # Get orchestrator from parameter or session state
    if orchestrator is None:
        orchestrator = st.session_state.get('orchestrator')
    
    if orchestrator is None:
        return {
            "query": query,
            "error": "Orchestrator not initialized",
            "response": "Error: System not properly initialized. Please check your configuration.",
            "citations": [],
            "metadata": {}
        }
    
    try:
        # Process query through orchestrator (synchronous)
        result = orchestrator.process_query(query)
        
        # Check for errors
        if "error" in result:
            return result
        
        # Extract citations and sources
        metadata = result.get("metadata", {})
        citations = metadata.get("citations", [])
        sources = metadata.get("sources", [])
        
        # Extract workflow trace
        workflow_trace = result.get("workflow_trace", [])
        
        # Extract safety information
        safety_violation = metadata.get("safety_violation", False)
        safety_violations = metadata.get("violations", [])
        
        # Format metadata
        metadata["citations"] = citations
        metadata["sources"] = sources
        metadata["workflow_trace"] = workflow_trace
        metadata["safety_violation"] = safety_violation
        metadata["safety_violations"] = safety_violations
        
        return {
            "query": query,
            "response": result.get("response", ""),
            "citations": citations,
            "sources": sources,
            "metadata": metadata,
            "workflow_trace": workflow_trace
        }
        
    except Exception as e:
        return {
            "query": query,
            "error": str(e),
            "response": f"An error occurred: {str(e)}",
            "citations": [],
            "metadata": {"error": True}
        }


def display_response(result: Dict[str, Any]):
    """
    Display query response with all details.
    """
    # Check for errors
    if "error" in result:
        st.error(f"❌ Error: {result['error']}")
        return

    # Safety violation alert
    metadata = result.get("metadata", {})
    if metadata.get("safety_violation"):
        violations = metadata.get("safety_violations", [])
        with st.expander("⚠️ SAFETY ALERT - Content Blocked/Sanitized", expanded=True):
            st.warning("This query or response triggered safety checks.")
            for violation in violations:
                reason = violation.get("reason", "Unknown violation")
                severity = violation.get("severity", "unknown")
                st.text(f"[{severity.upper()}] {reason}")

    # Display response
    st.markdown("### 📝 Response")
    response = result.get("response", "")
    st.markdown(response)

    # Display metadata metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sources", metadata.get("num_sources", 0))
    with col2:
        st.metric("Citations", len(result.get("citations", [])))
    with col3:
        if ORCHESTRATOR_TYPE == "sequential":
            st.metric("Iterations", metadata.get("iterations", 0))
        else:
            st.metric("Messages", metadata.get("num_messages", 0))
    with col4:
        elapsed_time = metadata.get("elapsed_time", 0)
        st.metric("Time", f"{elapsed_time:.1f}s")

    # Display citations
    citations = result.get("citations", [])
    if citations:
        with st.expander("📚 Citations & References", expanded=False):
            for i, citation in enumerate(citations, 1):
                st.markdown(f"**[{i}]** {citation}")

    # Display sources
    sources = result.get("sources", [])
    if sources:
        with st.expander("🔗 Sources", expanded=False):
            for i, source in enumerate(sources[:10], 1):
                title = source.get("title", "Unknown")
                url = source.get("url", "")
                authors = source.get("authors", [])
                
                st.markdown(f"**{i}. {title}**")
                if authors:
                    author_names = ", ".join([a.get("name", "") for a in authors[:3]])
                    st.caption(f"Authors: {author_names}")
                if url:
                    st.markdown(f"🔗 [{url}]({url})")
                st.divider()

    # Display workflow trace
    if st.session_state.show_traces:
        display_workflow_trace(result)

    # Display safety events if enabled
    if st.session_state.show_safety_log and st.session_state.safety_manager:
        display_safety_log()


def display_workflow_trace(result: Dict[str, Any]):
    """
    Display agent execution traces and workflow.
    """
    workflow_trace = result.get("workflow_trace", [])
    conversation_history = result.get("conversation_history", [])
    
    with st.expander("🔍 Workflow Trace", expanded=False):
        if workflow_trace:
            # Sequential orchestrator trace
            st.markdown("### Agent Workflow")
            for i, entry in enumerate(workflow_trace, 1):
                phase = entry.get("phase", "unknown").upper()
                message = entry.get("message", "")
                data = entry.get("data", {})
                timestamp = entry.get("timestamp", "")
                
                # Phase badge
                phase_colors = {
                    "PLANNING": "🔵",
                    "RESEARCH": "🟢",
                    "WRITING": "🟡",
                    "CRITIQUE": "🟠",
                    "SAFETY": "🔴",
                    "COMPLETE": "✅"
                }
                emoji = phase_colors.get(phase, "⚪")
                
                st.markdown(f"**{emoji} {i}. [{phase}] {message}**")
                if data:
                    with st.container():
                        for key, value in data.items():
                            if isinstance(value, (list, dict)):
                                st.json({key: value})
                            elif isinstance(value, str) and len(value) > 200:
                                st.text(f"{key}: {value[:200]}...")
                            else:
                                st.caption(f"{key}: {value}")
                st.divider()
        elif conversation_history:
            # AutoGen orchestrator - conversation history
            st.markdown("### Agent Conversation")
            for i, msg in enumerate(conversation_history, 1):
                agent = msg.get("source", "Unknown")
                content = msg.get("content", "")
                
                with st.container():
                    st.markdown(f"**{i}. [{agent}]**")
                    st.text_area("", value=content[:500] + "..." if len(content) > 500 else content, 
                                height=100, key=f"msg_{i}", disabled=True)
                st.divider()
        else:
            st.info("No workflow trace available for this query.")


def display_safety_log():
    """Display safety event log."""
    if not st.session_state.safety_manager:
        return
    
    safety_manager = st.session_state.safety_manager
    events = safety_manager.get_safety_events()
    stats = safety_manager.get_safety_stats()
    
    with st.expander("🛡️ Safety Event Log", expanded=False):
        if events:
            st.metric("Total Events", stats.get("total_events", 0))
            st.metric("Violations", stats.get("violations", 0))
            st.metric("Violation Rate", f"{stats.get('violation_rate', 0):.1%}")
            
            st.divider()
            st.markdown("### Recent Events")
            
            # Show last 10 events
            for event in events[-10:]:
                event_type = event.get("type", "unknown")
                is_safe = event.get("safe", True)
                violations = event.get("violations", [])
                timestamp = event.get("timestamp", "")
                
                if not is_safe:
                    st.warning(f"**{event_type.upper()}** - {timestamp}")
                    for violation in violations:
                        reason = violation.get("reason", "Unknown")
                        severity = violation.get("severity", "unknown")
                        st.caption(f"  [{severity}] {reason}")
                else:
                    st.success(f"**{event_type.upper()}** - {timestamp} - Safe")
                st.divider()
        else:
            st.info("No safety events recorded yet.")


def display_sidebar():
    """Display sidebar with settings and statistics."""
    with st.sidebar:
        st.title("⚙️ Settings")

        # Show traces toggle
        st.session_state.show_traces = st.checkbox(
            "Show Agent Traces",
            value=st.session_state.show_traces,
            help="Display detailed workflow trace showing each agent's actions"
        )

        # Show safety log toggle
        st.session_state.show_safety_log = st.checkbox(
            "Show Safety Log",
            value=st.session_state.show_safety_log,
            help="Display safety event log and violation statistics"
        )

        st.divider()

        st.title("📊 Statistics")

        # Query statistics
        st.metric("Total Queries", len(st.session_state.history))
        
        # Safety statistics
        if st.session_state.safety_manager:
            safety_stats = st.session_state.safety_manager.get_safety_stats()
            st.metric("Safety Events", safety_stats.get("total_events", 0))
            st.metric("Violations", safety_stats.get("violations", 0))
        else:
            st.metric("Safety Events", 0)

        st.divider()

        # Clear history button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        st.divider()

        # About section
        st.markdown("### ℹ️ About")
        config = load_config()
        system_name = config.get("system", {}).get("name", "Research Assistant")
        topic = config.get("system", {}).get("topic", "General")
        model = config.get("models", {}).get("default", {}).get("name", "Unknown")
        
        st.markdown(f"**System:** {system_name}")
        st.markdown(f"**Topic:** {topic}")
        st.markdown(f"**Model:** {model}")
        st.markdown(f"**Orchestrator:** {ORCHESTRATOR_TYPE}")


def display_history():
    """Display query history."""
    if not st.session_state.history:
        return

    with st.expander("📜 Query History", expanded=False):
        for i, item in enumerate(reversed(st.session_state.history), 1):
            timestamp = item.get("timestamp", "")
            query = item.get("query", "")
            result = item.get("result", {})
            metadata = result.get("metadata", {})
            score = metadata.get("num_sources", 0)
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i}.** {query}")
                st.caption(timestamp)
            with col2:
                st.caption(f"{score} sources")
            
            # Click to show result
            if st.button(f"View Result {i}", key=f"view_{i}"):
                st.session_state.selected_result = result
                st.rerun()


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Multi-Agent Research Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    initialize_session_state()

    # Header
    config = load_config()
    system_name = config.get("system", {}).get("name", "Multi-Agent Research Assistant")
    topic = config.get("system", {}).get("topic", "HCI Research")
    
    st.title(f"🤖 {system_name}")
    st.markdown(f"**Research Topic:** {topic}")
    st.markdown("Ask me anything about your research topic!")

    # Sidebar
    display_sidebar()

    # Main area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input - check for example query in session state
        # Get example query if one was selected
        default_query = st.session_state.get('example_query', '')
        
        query = st.text_area(
            "Enter your research query:",
            value=default_query,
            height=100,
            placeholder="e.g., What are the latest developments in explainable AI for novice users?",
            key="query_input"
        )
        
        # Clear example query after user submits or modifies the query
        if 'example_query' in st.session_state and query != default_query:
            del st.session_state.example_query
        # Submit button
        if st.button("🔍 Search", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("⏳ Processing your query (max 2 minutes)..."):
                    # Clear example query after submission
                    if 'example_query' in st.session_state:
                        del st.session_state.example_query
                    
                    # Get orchestrator before starting thread (session state not accessible in thread)
                    orchestrator = st.session_state.get('orchestrator')
                    if orchestrator is None:
                        result = {
                                "response": "Error: Orchestrator not initialized. Please refresh the page.",
                                "citations": [],
                                "metadata": {"error": "orchestrator_not_initialized"},
                                "traces": []
                        }
                    else:
                        # Use timeout wrapper
                        try:
                            result = process_query_with_timeout(
                                lambda q: process_query(q, orchestrator=orchestrator), 
                                timeout_seconds=120
                            )(query)
                        except FutureTimeoutError:
                            result = {
                                "response": "⚠️ Query processing timed out after 2 minutes. This may be due to network issues. Please try:\n1. Check your internet connection\n2. Try a simpler query\n3. Wait a moment and try again",
                                "citations": [],
                                "metadata": {"error": "timeout"},
                                "traces": []
                            }
                        except Exception as e:
                            result = {
                                "response": f"Error: {str(e)}",
                                "citations": [],
                                "metadata": {"error": str(e)},
                                "traces": []
                            }
                    
                    # Add to history
                    st.session_state.history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "query": query,
                        "result": result
                    })

                    # Display result
                    st.divider()
                    display_response(result)
            else:
                st.warning("⚠️ Please enter a query.")
            
            # Display selected result if any
        if 'selected_result' in st.session_state:
            st.divider()
            st.markdown("### Selected Result")
            display_response(st.session_state.selected_result)
            del st.session_state.selected_result

        # History
        display_history()

    with col2:
        st.markdown("### 💡 Example Queries")
        examples = [
            "What are the key principles of explainable AI for novice users?",
            "How has AR usability evolved in the past 5 years?",
            "What are ethical considerations in using AI for education?",
            "Compare different approaches to measuring user experience",
        ]

        for example in examples:
            if st.button(example, use_container_width=True, key=f"example_{example[:20]}"):
                # Set query in text area - will be picked up on rerun
                st.session_state.example_query = example
                st.rerun()

        st.divider()

        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        1. **Planner** breaks down your query
        2. **Researcher** gathers evidence from web and papers
        3. **Writer** synthesizes findings
        4. **Critic** verifies quality
        5. **Safety** checks ensure appropriate content
        """)

        st.divider()

        st.markdown("### 🔧 System Info")
        st.markdown(f"**Orchestrator:** {ORCHESTRATOR_TYPE}")
        st.markdown(f"**Model:** {config.get('models', {}).get('default', {}).get('name', 'Unknown')}")
        st.markdown(f"**Provider:** {config.get('models', {}).get('default', {}).get('provider', 'Unknown')}")



from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from functools import wraps

def process_query_with_timeout(query_func, timeout_seconds=120):
    """Wrapper to add timeout to process_query using threading."""
    def wrapper(query):
        # Use ThreadPoolExecutor for timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(query_func, query)
            try:
                result = future.result(timeout=timeout_seconds)
                return result
            except FutureTimeoutError:
                return {
                    "response": f"⚠️ Query processing timed out after {timeout_seconds} seconds. This may be due to network issues. Please try:\n1. Check your internet connection\n2. Try a simpler query\n3. Wait a moment and try again",
                    "citations": [],
                    "metadata": {"error": "timeout"},
                    "traces": []
                    }
            except Exception as e:
                return {
                    "response": f"Error: {str(e)}",
                    "citations": [],
                    "metadata": {"error": str(e)},
                    "traces": []
                    }
    return wrapper


if __name__ == "__main__":
    main()
