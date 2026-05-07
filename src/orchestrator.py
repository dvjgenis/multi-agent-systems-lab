"""
Sequential Orchestrator

This orchestrator coordinates agents in a sequential workflow:
plan → research → write → critique → revise (if needed)

It uses the agent classes directly (not AutoGen) for more control over the workflow.
"""

import logging
import time
import signal
from contextlib import contextmanager
from threading import Timer
from typing import Dict, Any, List, Optional
from src.agents.planner_agent import PlannerAgent
from src.agents.researcher_agent import ResearcherAgent
from src.agents.writer_agent import WriterAgent
from src.agents.critic_agent import CriticAgent
from src.guardrails.safety_manager import SafetyManager


@contextmanager
def timeout_context(seconds):
    """Context manager for timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    # Set up signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class Orchestrator:
    """
    Sequential orchestrator for multi-agent research system.
    Workflow:
    1. Planner: Creates research plan
    2. Researcher: Gathers evidence from web and papers
    3. Writer: Synthesizes findings into response
    4. Critic: Evaluates quality
    5. If approved → return response
    6. If needs revision → loop back to Writer (max iterations)
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the orchestrator.
        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("orchestrator")
        # System configuration
        system_config = config.get("system", {})
        self.max_iterations = system_config.get("max_iterations", 10)
        self.timeout_seconds = system_config.get("timeout_seconds", 300)
        # Initialize safety manager
        safety_config = config.get("safety", {})
        if safety_config.get("enabled", True):
            # Merge safety config with system config
            safety_config_full = {
                **safety_config,
                "topic": config.get("system", {}).get("topic", "HCI Research"),
                "logging": config.get("logging", {})
            }
            self.safety_manager = SafetyManager(safety_config_full)
            self.logger.info("Safety manager initialized")
        else:
            self.safety_manager = None
            self.logger.info("Safety manager disabled")
        # Initialize agents
        self.logger.info("Initializing agents...")
        try:
            self.planner = PlannerAgent(config)
            self.researcher = ResearcherAgent(config)
            self.writer = WriterAgent(config)
            self.critic = CriticAgent(config)
            self.logger.info("All agents initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing agents: {e}", exc_info=True)
            raise
        # Workflow trace for debugging and UI display
        self.workflow_trace: List[Dict[str, Any]] = []
        # State tracking
        self.current_state = {
            "query": None,
            "plan": None,
            "research_findings": None,
            "draft": None,
            "critique": None,
            "iteration": 0,
            "status": "idle"
        }

    def _check_timeout(self) -> bool:
        """
        Check if timeout has been exceeded.
        Returns:
            True if timeout exceeded, False otherwise
        """
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            self.logger.error(f"Timeout exceeded: {elapsed:.1f}s > {self.timeout_seconds}s")
            return True
        return False
    def _get_timeout_response(self) -> Dict[str, Any]:
        """Get a timeout error response."""
        elapsed = time.time() - self.start_time
        return {
            "response": f"Error: Query processing timed out after {self.timeout_seconds} seconds.",
            "citations": [],
            "metadata": {"error": "timeout", "elapsed": elapsed, "iteration": self.current_state.get("iteration", 0)},
            "traces": self.workflow_trace,
            "status": "error"
        }
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a research query through the multi-agent workflow.
        Args:
            query: The research question to answer
        Returns:
            Dictionary containing:
                - query: Original query
                - response: Final synthesized response
                - workflow_trace: Step-by-step trace of the workflow
                - metadata: Additional information (sources, citations, etc.)
        """
        self.start_time = time.time()
        self.logger.info(f"Processing query: {query}")
        # Safety check: Validate input
        if self.safety_manager:
            input_safety = self.safety_manager.check_input_safety(query)
            if not input_safety["safe"]:
                self.logger.warning(f"Input safety check failed: {input_safety['violations']}")
                self._add_trace("safety", "Input safety check failed", {
                    "violations": input_safety["violations"],
                    "action": input_safety["action"]
                })
                # Handle based on action
                if input_safety["action"] == "refuse":
                    return {
                        "query": query,
                        "response": self.safety_manager.violation_message,
                        "workflow_trace": self.workflow_trace,
                        "metadata": {
                            "elapsed_time": time.time() - self.start_time,
                            "safety_violation": True,
                            "violations": input_safety["violations"],
                            "status": "blocked"
                        }
                    }
                elif input_safety["action"] == "sanitize":
                    query = input_safety["sanitized_query"]
                    self.logger.info(f"Query sanitized: {query[:100]}...")
        # Reset state
        self.current_state = {
            "query": query,
            "plan": None,
            "research_findings": None,
            "draft": None,
            "critique": None,
            "iteration": 0,
            "status": "processing"
        }
        self.workflow_trace = []
        try:
            # Step 1: Planning
            plan_result = self._planning_phase(query)
            if plan_result["status"] == "error":
                return self._create_error_response(query, "Planning phase failed", plan_result)
            # Step 2: Research
            research_result = self._research_phase(query, plan_result)
            if research_result["status"] == "error":
                return self._create_error_response(query, "Research phase failed", research_result)
            # Step 3: Writing and Critiquing (with revision loop)
            final_result = self._writing_critique_loop(query, plan_result, research_result)
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time
            # Get final response
            final_response = final_result.get("response", "")
            # Validate response is not empty
            if not final_response or not final_response.strip():
                self.logger.warning("Final response is empty, using error message")
                final_response = "Error: The system was unable to generate a response. This may be due to API rate limits or processing errors."
            # Safety check: Validate output
            safety_violation = False
            safety_metadata = {}
            if self.safety_manager:
                sources = final_result.get("metadata", {}).get("sources", [])
                output_safety = self.safety_manager.check_output_safety(final_response, sources)
                if not output_safety["safe"]:
                    self.logger.warning(f"Output safety check failed: {output_safety['violations']}")
                    self._add_trace("safety", "Output safety check failed", {
                        "violations": output_safety["violations"],
                        "action": output_safety["action"]
                    })
                    safety_violation = True
                    safety_metadata = {
                        "violations": output_safety["violations"],
                        "action": output_safety["action"]
                    }
                    # Use sanitized/refused response
                    if output_safety["action"] == "refuse":
                        final_response = output_safety["response"]
                    elif output_safety["action"] == "sanitize":
                        final_response = output_safety["sanitized_response"]
            # Build final response
            result = {
                "query": query,
                "response": final_response,
                "workflow_trace": self.workflow_trace,
                "metadata": {
                    "elapsed_time": elapsed_time,
                    "iterations": self.current_state["iteration"],
                    "plan": self.current_state["plan"],
                    "num_sources": final_result.get("metadata", {}).get("num_citations", 0),
                    "sources": final_result.get("metadata", {}).get("sources", []),
                    "citations": final_result.get("metadata", {}).get("citations", []),
                    "critique": self.current_state["critique"],
                    "status": final_result.get("status", "complete"),
                    "safety_violation": safety_violation,
                    **safety_metadata
                }
            }
            self.logger.info(f"Query processing complete in {elapsed_time:.2f}s")
            return result
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return self._create_error_response(query, f"Orchestration error: {str(e)}", {})
    def _planning_phase(self, query: str) -> Dict[str, Any]:
        """
        Execute the planning phase.
        Args:
            query: Research query
        Returns:
            Planning result dictionary
        """
        # Check timeout
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            self.logger.error(f"Timeout exceeded: {elapsed:.1f}s > {self.timeout_seconds}s")
            return {
                "response": f"Error: Query processing timed out after {self.timeout_seconds} seconds.",
                "citations": [],
                "metadata": {"error": "timeout", "elapsed": elapsed},
                "traces": self.workflow_trace
            }
        self.logger.info("=== PLANNING PHASE ===")
        self._add_trace("planning", "Starting planning phase", {"query": query})
        # Check timeout before planning
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout_seconds:
            return self._get_timeout_response()
        try:
            plan_result = self.planner.process({"query": query})
            if plan_result["status"] == "error":
                self.logger.error("Planning failed")
                return plan_result
            plan_text = plan_result["output"]
            self.current_state["plan"] = plan_text
            self._add_trace("planning", "Plan created", {
                "plan": plan_text[:200] + "..." if len(plan_text) > 200 else plan_text,
                "status": plan_result["status"]
            })
            self.logger.info("Planning phase complete")
            return plan_result
        except Exception as e:
            self.logger.error(f"Planning phase error: {e}", exc_info=True)
            return {"status": "error", "output": f"Planning error: {str(e)}"}
    def _research_phase(self, query: str, plan_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the research phase.
        Args:
            query: Research query
            plan_result: Result from planning phase
        Returns:
            Research result dictionary
        """
        self.logger.info("=== RESEARCH PHASE ===")
        self._add_trace("research", "Starting research phase", {"query": query})
        try:
            # Extract search queries from plan
            plan_metadata = plan_result.get("metadata", {})
            search_queries = plan_metadata.get("search_queries", [])
            research_input = {
                "query": query,
                "plan": plan_result["output"],
                "search_queries": search_queries
            }
            # Check timeout before research
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                return self._get_timeout_response()
            research_result = self.researcher.process(research_input)
            if research_result["status"] == "error":
                self.logger.error("Research failed")
                return research_result
            research_text = research_result["output"]
            self.current_state["research_findings"] = research_text
            self._add_trace("research", "Research complete", {
                "findings_length": len(research_text),
                "sources": research_result.get("metadata", {}).get("sources", []),
                "status": research_result["status"]
            })
            self.logger.info("Research phase complete")
            return research_result
        except Exception as e:
            self.logger.error(f"Research phase error: {e}", exc_info=True)
            return {"status": "error", "output": f"Research error: {str(e)}"}
    def _writing_critique_loop(
        self,
        query: str,
        plan_result: Dict[str, Any],
        research_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute writing and critique phases with revision loop.
        Args:
            query: Research query
            plan_result: Result from planning phase
            research_result: Result from research phase
        Returns:
            Final result dictionary
        """
        self.logger.info("=== WRITING & CRITIQUE LOOP ===")
        max_revisions = self.max_iterations - 2  # Reserve iterations for plan and research
        draft = None
        for iteration in range(max_revisions):
            self.current_state["iteration"] = iteration + 1
            # Check timeout before each iteration
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                self.logger.error(f"Timeout exceeded during iteration {iteration + 1}: {elapsed:.1f}s > {self.timeout_seconds}s")
                return {
                    "response": f"Error: Query processing timed out after {self.timeout_seconds} seconds.",
                    "citations": [],
                    "metadata": {"error": "timeout", "elapsed": elapsed, "iteration": iteration + 1},
                    "traces": self.workflow_trace
                }
            # Writing phase
            self.logger.info(f"Writing iteration {iteration + 1}")
            self._add_trace("writing", f"Writing iteration {iteration + 1}", {})
            try:
                # Prepare sources from research metadata
                research_metadata = research_result.get("metadata", {})
                sources = research_metadata.get("sources", [])
                # Extract sources from research text if not in metadata
                if not sources:
                    sources = self._extract_sources_from_research(research_result["output"])
                writing_input = {
                    "query": query,
                    "plan": plan_result["output"],
                    "research_findings": research_result["output"],
                    "sources": sources
                }
                # Add critique feedback if this is a revision
                if iteration > 0 and self.current_state["critique"]:
                    writing_input["critique_feedback"] = self.current_state["critique"]
                # Check timeout before writing
                elapsed = time.time() - self.start_time
                if elapsed > self.timeout_seconds:
                    return self._get_timeout_response()
                writing_result = self.writer.process(writing_input)
                if writing_result["status"] == "error":
                    self.logger.error(f"Writing failed on iteration {iteration + 1}")
                    return writing_result
                draft = writing_result["output"]
                self.current_state["draft"] = draft
                self._add_trace("writing", f"Draft {iteration + 1} complete", {
                    "draft_length": len(draft),
                    "citations": writing_result.get("metadata", {}).get("num_citations", 0)
                })
            except Exception as e:
                self.logger.error(f"Writing error on iteration {iteration + 1}: {e}", exc_info=True)
                return {
                    "status": "error",
                    "response": f"Writing error: {str(e)}",
                    "metadata": {}
                }
            # Critique phase
            self.logger.info(f"Critique iteration {iteration + 1}")
            self._add_trace("critique", f"Critique iteration {iteration + 1}", {})
            try:
                critique_input = {
                    "query": query,
                    "response": draft,
                    "plan": plan_result["output"],
                    "research_findings": research_result["output"]
                }
                # Check timeout before critique
                elapsed = time.time() - self.start_time
                if elapsed > self.timeout_seconds:
                    return self._get_timeout_response()
                critique_result = self.critic.process(critique_input)
                if critique_result["status"] == "error":
                    self.logger.error(f"Critique failed on iteration {iteration + 1}")
                    # Continue with current draft if critique fails
                    break
                critique_text = critique_result["output"]
                critique_status = critique_result["status"]
                self.current_state["critique"] = critique_text
                self._add_trace("critique", f"Critique {iteration + 1} complete", {
                    "status": critique_status,
                    "scores": critique_result.get("metadata", {}).get("scores", {})
                })
                # Check if approved
                if critique_status == "approved":
                    self.logger.info(f"Response approved after {iteration + 1} iteration(s)")
                    self._add_trace("complete", "Response approved", {
                        "iterations": iteration + 1
                    })
                    return {
                        "status": "complete",
                        "response": draft,
                        "metadata": writing_result.get("metadata", {})
                    }
                # If needs revision and we have more iterations, continue loop
                if critique_status == "needs_revision" and iteration < max_revisions - 1:
                    self.logger.info(f"Revision needed, continuing to iteration {iteration + 2}")
                    continue
                else:
                    # Max iterations reached or other status
                    self.logger.warning(f"Stopping after {iteration + 1} iteration(s), status: {critique_status}")
                    break
            except Exception as e:
                self.logger.error(f"Critique error on iteration {iteration + 1}: {e}", exc_info=True)
                # Continue with current draft if critique fails
                break
        # Return final draft (may not be approved if max iterations reached)
        self.logger.info("Writing-critique loop complete")
        return {
            "status": "complete" if draft else "error",
            "response": draft or "No draft generated",
            "metadata": writing_result.get("metadata", {}) if draft else {}
        }
    def _extract_sources_from_research(self, research_text: str) -> List[Dict[str, Any]]:
        """
        Extract source information from research text.
        Args:
            research_text: Research findings text
        Returns:
            List of source dictionaries
        """
        sources = []
        # Look for URLs
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', research_text)
        for url in set(urls):  # Deduplicate
            sources.append({
                "type": "webpage",
                "url": url,
                "title": url.split("/")[-1] if "/" in url else url
            })
        # Look for paper citations (lines with titles and authors)
        lines = research_text.split("\n")
        for i, line in enumerate(lines):
            # Pattern: Numbered list items that look like papers
            if re.match(r'^\d+\.', line.strip()):
                # Try to extract title and URL from this line and next lines
                title = line.replace(re.match(r'^\d+\.', line.strip()).group(), "").strip()
                url = None
                # Look for URL in next few lines
                for j in range(i + 1, min(i + 5, len(lines))):
                    if "http" in lines[j]:
                        url_match = re.search(r'http[s]?://[^\s]+', lines[j])
                        if url_match:
                            url = url_match.group()
                            break
                if title:
                    sources.append({
                        "type": "paper",
                        "title": title,
                        "url": url or ""
                    })
        return sources
    def _add_trace(self, phase: str, message: str, data: Dict[str, Any]):
        """
        Add an entry to the workflow trace.
        Args:
            phase: Workflow phase (planning, research, writing, critique)
            message: Trace message
            data: Additional trace data
        """
        trace_entry = {
            "phase": phase,
            "message": message,
            "timestamp": time.time(),
            "data": data
        }
        self.workflow_trace.append(trace_entry)
    def _create_error_response(
        self,
        query: str,
        error_message: str,
        error_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create an error response dictionary.
        Args:
            query: Original query
            error_message: Error message
            error_data: Additional error data
        Returns:
            Error response dictionary
        """
        return {
            "query": query,
            "response": f"Error: {error_message}",
            "workflow_trace": self.workflow_trace,
            "metadata": {
                "error": True,
                "error_message": error_message,
                "error_data": error_data,
                "status": "error"
            }
        }
    def get_agent_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all agents.
        Returns:
            Dictionary mapping agent names to their descriptions
        """
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }
    def visualize_workflow(self) -> str:
        """
        Generate a text visualization of the workflow.
        Returns:
            String representation of the workflow
        """
        workflow = """
Sequential Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics and search queries
   ↓
3. Researcher
   - Uses web_search() tool (Tavily/Brave)
   - Uses paper_search() tool (Semantic Scholar)
   - Gathers evidence from multiple sources
   - Synthesizes findings
   ↓
4. Writer
   - Synthesizes research findings
   - Creates structured response
   - Adds inline citations
   - Generates references section
   ↓
5. Critic
   - Evaluates quality (relevance, evidence, completeness, accuracy, clarity)
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Return Final Response
   - If NEEDS REVISION → Loop back to Writer (max iterations)
        """
        return workflow
    def get_workflow_trace_summary(self) -> str:
        """
        Get a summary of the workflow trace.
        Returns:
            Formatted trace summary string
        """
        if not self.workflow_trace:
            return "No workflow trace available."
        summary = "Workflow Trace:\n"
        summary += "=" * 70 + "\n"
        for i, entry in enumerate(self.workflow_trace, 1):
            summary += f"\n{i}. [{entry['phase'].upper()}] {entry['message']}\n"
            if entry.get('data'):
                for key, value in entry['data'].items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    summary += f"   - {key}: {value}\n"
        return summary