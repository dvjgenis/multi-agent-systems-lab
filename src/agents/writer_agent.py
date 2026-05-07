"""
Writer Agent

This agent synthesizes research findings into coherent, well-cited responses.
It formats the output with proper citations and references.
"""

import os
import logging
import warnings
import urllib3
from typing import Dict, Any, List
from openai import OpenAI
from src.agents.base_agent import BaseAgent
from src.tools.citation_tool import CitationTool

# Suppress SSL warnings for vLLM servers
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class WriterAgent(BaseAgent):
    """
    Writer Agent - Synthesizes research findings into coherent responses.
    
    The writer takes research findings and creates a well-structured,
    properly cited response. It:
    - Synthesizes information from multiple sources
    - Adds inline citations
    - Creates a references section
    - Ensures clarity and organization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Writer Agent.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        super().__init__("writer", config)
        
        # Initialize LLM client with fallback support
        try:
            self.client = self._create_llm_client()
        except Exception as e:
            # If primary provider fails, try fallback providers
            self.logger.warning(f"Primary provider {self.model_provider} failed: {e}")
            self.logger.info("Attempting fallback to other available providers...")
            self.client = self._try_providers_with_fallback()
        
        # Initialize citation tool
        self.citation_tool = CitationTool(style="apa")
        
        # Default system prompt
        self.default_system_prompt = """You are a Research Writer. Your job is to synthesize research findings into clear, well-organized responses.

When writing:
1. Start with an overview/introduction
2. Present findings in a logical structure
3. Cite sources inline using [Source: Title/Author] format
4. Synthesize information from multiple sources
5. Avoid copying text directly - paraphrase and synthesize
6. Include a references section at the end
7. Ensure the response directly answers the original query

Format your response professionally with clear headings, paragraphs, in-text citations, and a References section at the end.

After completing the draft, end with "DRAFT COMPLETE"."""
        
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
        Process research findings and create a written response.
        
        Args:
            input_data: Dictionary containing:
                - query: The original research query
                - plan: The research plan
                - research_findings: Findings from Researcher
                - sources: List of sources with metadata
                
        Returns:
            Dictionary containing:
                - output: The written response with citations
                - metadata: Citations, references, etc.
                - status: "complete" if draft is ready
        """
        if not self.validate_input(input_data):
            return {
                "output": "Invalid input provided.",
                "metadata": {},
                "status": "error"
            }
        
        query = input_data.get("query", input_data.get("content", ""))
        plan = input_data.get("plan", "")
        research_findings = input_data.get("research_findings", "")
        sources = input_data.get("sources", [])
        
        self.log_action("Writing response", {"query": query})
        
        try:
            # Process sources for citation tool
            self.citation_tool.clear_citations()
            processed_sources = self._process_sources(sources, research_findings)
            
            # Build the writing prompt
            user_message = self._build_writing_prompt(query, plan, research_findings, processed_sources)
            
            # Generate the written response with fallback support
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
                
                draft = response.choices[0].message.content
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
                        draft = response.choices[0].message.content
                    except Exception as fallback_error:
                        self.logger.error(f"All providers failed. Fallback error: {fallback_error}")
                        draft = f"Research Response:\n\n{research_summary}\n\nDRAFT COMPLETE"
                else:
                    # Non-connection error
                    self.logger.error(f"Error generating draft: {e}")
                    draft = f"Research Response:\n\n{research_summary}\n\nDRAFT COMPLETE"
            
            # Add formatted references section
            bibliography = self.citation_tool.generate_bibliography()
            if bibliography:
                draft = self._add_references_section(draft, bibliography)
            
            # Check if draft is complete
            status = "complete" if "DRAFT COMPLETE" in draft.upper() else "in_progress"
            
            self.log_action("Draft created", {
                "status": status,
                "num_sources": len(processed_sources),
                "num_citations": len(bibliography)
            })
            
            return {
                "output": draft,
                "metadata": {
                    "sources": processed_sources,
                    "citations": bibliography,
                    "num_citations": len(bibliography)
                },
                "status": status
            }
            
        except Exception as e:
            self.logger.error(f"Error writing response: {e}", exc_info=True)
            return {
                "output": f"Error creating written response: {str(e)}",
                "metadata": {},
                "status": "error"
            }
    
    def _process_sources(self, sources: List[Dict], research_findings: str) -> List[Dict[str, Any]]:
        """
        Process sources and add them to the citation tool.
        
        Args:
            sources: List of source dictionaries
            research_findings: Research findings text (may contain source info)
            
        Returns:
            List of processed source dictionaries
        """
        processed = []
        
        # Process provided sources
        for source in sources:
            source_dict = self._normalize_source(source)
            if source_dict:
                self.citation_tool.add_citation(source_dict)
                processed.append(source_dict)
        
        # Extract sources from research findings text
        extracted_sources = self._extract_sources_from_text(research_findings)
        for source in extracted_sources:
            if source not in processed:
                source_dict = self._normalize_source(source)
                if source_dict:
                    self.citation_tool.add_citation(source_dict)
                    processed.append(source_dict)
        
        return processed
    
    def _normalize_source(self, source: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a source dictionary to citation tool format.
        
        Args:
            source: Source dictionary (may be in various formats)
            
        Returns:
            Normalized source dictionary
        """
        # Handle different source formats
        if isinstance(source, str):
            # Try to parse string source
            return {
                "type": "webpage",
                "title": source,
                "url": source if source.startswith("http") else "",
                "year": None
            }
        
        # Normalize dictionary source
        source_type = source.get("type", "article")
        if "url" in source and not source_type:
            source_type = "webpage"
        elif "venue" in source or "authors" in source:
            source_type = "paper"
        
        normalized = {
            "type": source_type,
            "title": source.get("title", "Untitled"),
            "authors": source.get("authors", []),
            "year": source.get("year"),
            "url": source.get("url", ""),
            "venue": source.get("venue", ""),
            "doi": source.get("doi"),
            "site_name": source.get("site_name", "")
        }
        
        return normalized
    
    def _extract_sources_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract source information from research findings text.
        
        Args:
            text: Research findings text
            
        Returns:
            List of extracted source dictionaries
        """
        sources = []
        
        # Look for URLs
        import re
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        
        for url in urls:
            sources.append({
                "type": "webpage",
                "url": url,
                "title": url.split("/")[-1] if "/" in url else url
            })
        
        # Look for paper citations (lines with authors and years)
        lines = text.split("\n")
        for line in lines:
            # Pattern: Author et al. (Year) or Author, A. & Author, B. (Year)
            match = re.search(r'([A-Z][a-z]+(?:\s+et\s+al\.)?(?:\s*,\s*[A-Z]\.\s*&?\s*[A-Z][a-z]+)*)\s*\((\d{4})\)', line)
            if match:
                authors_str = match.group(1)
                year = match.group(2)
                
                # Try to find title in the same line or next line
                title = line.split("(")[0].strip() if "(" in line else ""
                
                sources.append({
                    "type": "paper",
                    "authors": [{"name": a.strip()} for a in authors_str.split("&")],
                    "year": int(year),
                    "title": title
                })
        
        return sources
    
    def _build_writing_prompt(
        self,
        query: str,
        plan: str,
        research_findings: str,
        sources: List[Dict[str, Any]]
    ) -> str:
        """
        Build the prompt for the writer agent.
        
        Args:
            query: Original research query
            plan: Research plan
            research_findings: Research findings
            sources: List of sources
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""Research Query: {query}

Research Plan:
{plan}

Research Findings:
{research_findings}

Available Sources ({len(sources)}):
"""
        
        for i, source in enumerate(sources[:10], 1):  # Limit to 10 sources
            prompt += f"{i}. {source.get('title', 'Unknown')}\n"
            if source.get('authors'):
                authors = ", ".join([a.get('name', '') for a in source['authors'][:3]])
                prompt += f"   Authors: {authors}\n"
            if source.get('year'):
                prompt += f"   Year: {source['year']}\n"
            if source.get('url'):
                prompt += f"   URL: {source['url']}\n"
            prompt += "\n"
        
        prompt += """
Please write a comprehensive response that:
1. Directly answers the research query
2. Synthesizes information from the findings
3. Uses inline citations like [Source: Title] when referencing sources
4. Is well-organized with clear sections
5. Includes a References section at the end

End with "DRAFT COMPLETE" when finished."""
        
        return prompt
    
    def _add_references_section(self, draft: str, bibliography: List[str]) -> str:
        """
        Add a formatted references section to the draft.
        
        Args:
            draft: The draft text
            bibliography: List of formatted citations
            
        Returns:
            Draft with references section added
        """
        # Remove "DRAFT COMPLETE" if present
        draft = draft.replace("DRAFT COMPLETE", "").strip()
        
        # Check if references section already exists
        if "References" in draft or "REFERENCES" in draft:
            return draft
        
        # Add references section
        references_section = "\n\n## References\n\n"
        for i, citation in enumerate(bibliography, 1):
            references_section += f"{i}. {citation}\n"
        
        draft += references_section + "\n\nDRAFT COMPLETE"
        
        return draft