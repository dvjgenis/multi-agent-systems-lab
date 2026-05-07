"""
Researcher Agent

This agent gathers evidence from web sources and academic papers.
It uses search tools (Tavily, Semantic Scholar) to find relevant information.
"""

import os
import logging
import warnings
import urllib3
from typing import Dict, Any, List
from openai import OpenAI
from src.agents.base_agent import BaseAgent
from src.tools.web_search import web_search
from src.tools.paper_search import paper_search

# Suppress SSL warnings for vLLM servers
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ResearcherAgent(BaseAgent):
    """
    Researcher Agent - Gathers evidence from web and academic sources.
    
    The researcher uses search tools to find relevant information:
    - Web search via Tavily or Brave API
    - Academic paper search via Semantic Scholar API
    
    It collects evidence, extracts key findings, and tracks sources
    for citation purposes.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Researcher Agent.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        super().__init__("researcher", config)
        
        # Initialize LLM client with fallback support
        try:
            self.client = self._create_llm_client()
        except Exception as e:
            # If primary provider fails, try fallback providers
            self.logger.warning(f"Primary provider {self.model_provider} failed: {e}")
            self.logger.info("Attempting fallback to other available providers...")
            self.client = self._try_providers_with_fallback()
        
        # Tool configuration
        tool_config = config.get("tools", {})
        self.web_search_provider = tool_config.get("web_search", {}).get("provider", "tavily")
        self.max_web_results = tool_config.get("web_search", {}).get("max_results", 5)
        self.max_paper_results = tool_config.get("paper_search", {}).get("max_results", 10)
        
        # Default system prompt
        self.default_system_prompt = """You are a Research Assistant. Your job is to gather high-quality information from academic papers and web sources.

You have access to tools for web search and paper search. When conducting research:
1. Use both web search and paper search for comprehensive coverage
2. Look for recent, high-quality sources
3. Extract key findings, quotes, and data
4. Note all source URLs and citations
5. Gather evidence that directly addresses the research query

After gathering sufficient evidence (8-10 sources), summarize your findings and end with "RESEARCH COMPLETE"."""
        
        # Use custom prompt if provided
        if self.system_prompt and self.system_prompt.strip():
            self.system_message = self.system_prompt
        else:
            self.system_message = self.default_system_prompt
    
    def _create_llm_client(self, provider: str = None) -> OpenAI:
        """
        Create an OpenAI-compatible client for LLM calls with fallback support.
        
        Args:
            provider: Provider to use (defaults to self.model_provider)
        
        Returns:
            OpenAI client instance
        """
        provider = provider or self.model_provider
        
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            
            return OpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
        
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            
            return OpenAI(api_key=api_key)
        
        elif provider == "vllm":
            api_key = os.getenv("OPENAI_API_KEY", "dummy")
            base_url = os.getenv("OPENAI_BASE_URL")
            if not base_url:
                raise ValueError("OPENAI_BASE_URL not found in environment")
            
            # For vLLM servers, disable SSL verification (internal servers)
            import httpx
            http_client = httpx.Client(verify=False, timeout=60.0)
            
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _try_providers_with_fallback(self) -> OpenAI:
        """
        Try to create LLM client with fallback to other providers.
        
        Returns:
            OpenAI client instance from first working provider
        """
        # Get preferred provider and available providers
        preferred = self.model_provider
        available = self._get_available_providers()
        
        if not available:
            raise ValueError("No LLM providers available. Check your .env file for API keys.")
        
        # Try preferred provider first
        providers_to_try = [preferred] if preferred in available else []
        # Add other available providers as fallbacks
        providers_to_try.extend([p for p in available if p != preferred])
        
        last_error = None
        for provider in providers_to_try:
            try:
                self.logger.info(f"Trying provider: {provider}")
                client = self._create_llm_client(provider)
                self.logger.info(f"Successfully connected to {provider}")
                # Update the model_provider to the working one
                self.model_provider = provider
                return client
            except Exception as e:
                self.logger.warning(f"Provider {provider} failed: {e}")
                last_error = e
                continue
        
        # All providers failed
        raise ValueError(
            f"All available providers failed. Tried: {providers_to_try}. "
            f"Last error: {last_error}"
        )
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a research plan and gather evidence.
        
        Args:
            input_data: Dictionary containing:
                - query: The research query
                - plan: The research plan from Planner
                - search_queries: Optional list of specific search queries
                
        Returns:
            Dictionary containing:
                - output: Summary of research findings
                - metadata: Sources found, citations, etc.
                - status: "complete" if research is done
        """
        if not self.validate_input(input_data):
            return {
                "output": "Invalid input provided.",
                "metadata": {},
                "status": "error"
            }
        
        query = input_data.get("query", input_data.get("content", ""))
        plan = input_data.get("plan", "")
        search_queries = input_data.get("search_queries", [])
        
        self.log_action("Starting research", {"query": query})
        
        try:
            # Extract search queries from plan if not provided
            if not search_queries and plan:
                search_queries = self._extract_search_queries(plan, query)
            
            # If still no queries, use the main query
            if not search_queries:
                search_queries = [query]
            
            # Gather evidence from web and papers
            web_results = []
            paper_results = []
            
            for search_query in search_queries[:3]:  # Limit to 3 queries to avoid too many API calls
                self.logger.info(f"Searching: {search_query}")
                
                # Web search
                try:
                    web_output = web_search(
                        query=search_query,
                        provider=self.web_search_provider,
                        max_results=self.max_web_results
                    )
                    if web_output and "No search results" not in web_output:
                        web_results.append({
                            "query": search_query,
                            "results": web_output
                        })
                except Exception as e:
                    self.logger.warning(f"Web search failed for '{search_query}': {e}")
                
                # Paper search
                try:
                    paper_output = paper_search(
                        query=search_query,
                        max_results=self.max_paper_results,
                        year_from=2019  # Focus on recent papers
                    )
                    if paper_output and "No academic papers" not in paper_output:
                        paper_results.append({
                            "query": search_query,
                            "results": paper_output
                        })
                except Exception as e:
                    self.logger.warning(f"Paper search failed for '{search_query}': {e}")
            
            # Combine results
            all_results = self._combine_results(web_results, paper_results)
            
            # Use LLM to synthesize findings
            research_summary = self._synthesize_findings(query, plan, all_results)
            
            # Extract metadata (sources, citations)
            metadata = self._extract_research_metadata(all_results, research_summary)
            
            # Check if research is complete
            status = "complete" if "RESEARCH COMPLETE" in research_summary.upper() else "in_progress"
            
            self.log_action("Research complete", {
                "status": status,
                "web_sources": len(web_results),
                "paper_sources": len(paper_results)
            })
            
            return {
                "output": research_summary,
                "metadata": metadata,
                "status": status
            }
            
        except Exception as e:
            self.logger.error(f"Error during research: {e}", exc_info=True)
            return {
                "output": f"Error gathering research: {str(e)}",
                "metadata": {},
                "status": "error"
            }
    
    def _extract_search_queries(self, plan: str, query: str) -> List[str]:
        """
        Extract search queries from the research plan.
        
        Args:
            plan: The research plan text
            query: Original query
            
        Returns:
            List of search query strings
        """
        queries = []
        
        # Look for quoted text or lines with "search"
        lines = plan.split("\n")
        for line in lines:
            line_lower = line.lower()
            if "search" in line_lower or "query" in line_lower:
                # Extract quoted text
                import re
                quotes = re.findall(r'"([^"]*)"', line)
                queries.extend(quotes)
                
                # Or extract text after colon
                if ":" in line and not quotes:
                    query_part = line.split(":", 1)[1].strip()
                    if query_part and len(query_part) > 5:
                        queries.append(query_part)
        
        # If no queries found, use key phrases from the plan
        if not queries:
            # Extract key phrases (simple heuristic)
            words = query.split()
            if len(words) > 3:
                queries.append(query)
            else:
                queries.append(query)
        
        return queries[:5]  # Limit to 5 queries
    
    def _combine_results(self, web_results: List[Dict], paper_results: List[Dict]) -> str:
        """
        Combine web and paper search results into a single text.
        
        Args:
            web_results: List of web search result dictionaries
            paper_results: List of paper search result dictionaries
            
        Returns:
            Combined results as formatted text
        """
        combined = ""
        
        if web_results:
            combined += "=== WEB SEARCH RESULTS ===\n\n"
            for result_set in web_results:
                combined += f"Query: {result_set['query']}\n"
                combined += result_set['results'] + "\n\n"
        
        if paper_results:
            combined += "=== ACADEMIC PAPER RESULTS ===\n\n"
            for result_set in paper_results:
                combined += f"Query: {result_set['query']}\n"
                combined += result_set['results'] + "\n\n"
        
        return combined
    
    def _synthesize_findings(self, query: str, plan: str, results: str) -> str:
        """
        Use LLM to synthesize research findings.
        
        Args:
            query: Original research query
            plan: Research plan
            results: Combined search results
            
        Returns:
            Synthesized research summary
        """
        user_message = f"""Research Query: {query}

Research Plan:
{plan}

Search Results:
{results}

Please synthesize these findings into a coherent summary. Focus on:
1. Key findings relevant to the query
2. Important sources and citations
3. Evidence that directly addresses the research question

End with "RESEARCH COMPLETE" when done."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            # Check if it's a connection error that might benefit from fallback
            is_connection_error = any(keyword in error_msg.lower() for keyword in [
                "ssl", "tls", "connection error", "connect", "timeout"
            ])
            
            if is_connection_error:
                self.logger.warning(f"Connection error with {self.model_provider}: {error_msg}")
                self.logger.info("Attempting fallback to other providers...")
                
                try:
                    self.client = self._try_providers_with_fallback()
                    # Retry the request with the new client
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.system_message},
                            {"role": "user", "content": user_message}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    self.logger.info(f"Successfully used fallback provider: {self.model_provider}")
                    return response.choices[0].message.content
                except Exception as fallback_error:
                    self.logger.error(f"All providers failed. Fallback error: {fallback_error}")
                    # Return basic findings without LLM synthesis
                    return f"Research findings gathered:\n\n{results}\n\nRESEARCH COMPLETE"
            else:
                # Non-connection error, return basic findings
                self.logger.error(f"Error synthesizing findings: {e}")
                return f"Research findings gathered:\n\n{results}\n\nRESEARCH COMPLETE"
    
    def _extract_research_metadata(self, results: str, summary: str) -> Dict[str, Any]:
        """
        Extract metadata from research results.
        
        Args:
            results: Combined search results text
            summary: Research summary
            
        Returns:
            Dictionary with metadata (sources, URLs, etc.)
        """
        metadata = {
            "sources": [],
            "urls": [],
            "papers": []
        }
        
        # Extract URLs
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', results)
        metadata["urls"] = list(set(urls))  # Deduplicate
        
        # Extract paper titles (lines starting with numbers)
        lines = results.split("\n")
        for i, line in enumerate(lines):
            if line.strip() and (line[0].isdigit() or line.startswith("   ")):
                # Might be a paper title
                if "http" in line or "URL:" in line:
                    continue
                if len(line.strip()) > 10:
                    metadata["papers"].append(line.strip())
        
        return metadata