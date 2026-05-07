"""
Planner Agent

This agent breaks down research queries into actionable research steps.
It analyzes the query, identifies key concepts, and creates a structured plan.
"""

import os
import logging
import warnings
import urllib3
from typing import Dict, Any, List
from openai import OpenAI
from openai import APIConnectionError
from src.agents.base_agent import BaseAgent

# Suppress SSL warnings for vLLM servers
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class PlannerAgent(BaseAgent):
    """
    Planner Agent - Breaks down research queries into actionable steps.
    
    The planner analyzes research queries and creates structured plans
    that guide the research process. It identifies:
    - Key concepts and topics to investigate
    - Types of sources needed (academic papers, web articles, etc.)
    - Specific search queries for the Researcher
    - How findings should be synthesized
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Planner Agent.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        super().__init__("planner", config)
        
        # Initialize LLM client with fallback support
        try:
            self.client = self._create_llm_client()
        except Exception as e:
            # If primary provider fails, try fallback providers
            self.logger.warning(f"Primary provider {self.model_provider} failed: {e}")
            self.logger.info("Attempting fallback to other available providers...")
            self.client = self._try_providers_with_fallback()
        
        # Default system prompt
        self.default_system_prompt = """You are a Research Planner. Your job is to break down research queries into clear, actionable steps.

When given a research query, you should:
1. Identify the key concepts and topics to investigate
2. Determine what types of sources would be most valuable (academic papers, web articles, etc.)
3. Suggest specific search queries for the Researcher
4. Outline how the findings should be synthesized

Provide your plan in a structured format with numbered steps.
Be specific about what information to gather and why it's relevant.

After creating the plan, end your response with "PLAN COMPLETE"."""
        
        # Use custom prompt if provided, otherwise use default
        if self.system_prompt and self.system_prompt.strip():
            self.system_message = self.system_prompt
        else:
            self.system_message = self.default_system_prompt
    
    def _create_llm_client(self, provider: str = None) -> OpenAI:
        """
        Create an OpenAI-compatible client for LLM calls with fallback support.
        
        Supports Groq, OpenAI, and vLLM providers. Will try fallback providers
        if the primary provider fails.
        
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
            
            http_client = httpx.Client(
                verify=False,
                timeout=60.0,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                follow_redirects=True
            )
            
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=http_client
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _get_available_providers(self) -> list:
        """
        Get list of available providers based on environment variables.
        
        Returns:
            List of provider names in order of preference
        """
        providers = []
        
        # Check what's available
        if os.getenv("GROQ_API_KEY"):
            providers.append("groq")
        if os.getenv("OPENAI_API_KEY"):
            providers.append("openai")
        if os.getenv("OPENAI_BASE_URL"):
            providers.append("vllm")
        
        return providers
    
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
        
        # Try preferred provider first, then others (prioritize groq for reliability)
        providers_to_try = [preferred] if preferred in available else []
        # Add other available providers as fallbacks
        other_providers = [p for p in available if p != preferred]
        # Put groq first if available (most reliable)
        if "groq" in other_providers:
            other_providers.remove("groq")
            other_providers.insert(0, "groq")
        providers_to_try.extend(other_providers)
        
        last_error = None
        for provider in providers_to_try:
            try:
                self.logger.info(f"Trying provider: {provider}")
                client = self._create_llm_client(provider)
                self.logger.info(f"Successfully created client for {provider}")
                # Update the model_provider to the working one
                self.model_provider = provider
                return client
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"Provider {provider} client creation failed: {error_msg}")
                last_error = e
                continue
        
        # All providers failed
        raise ValueError(
            f"All available providers failed. Tried: {providers_to_try}. "
            f"Last error: {last_error}"
        )
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a research query and create a research plan.
        
        Args:
            input_data: Dictionary containing:
                - query: The research query to plan for
                - context: Optional additional context
                
        Returns:
            Dictionary containing:
                - output: The research plan
                - metadata: Plan details (steps, topics, etc.)
                - status: "complete" if plan is ready
        """
        if not self.validate_input(input_data):
            return {
                "output": "Invalid input provided.",
                "metadata": {},
                "status": "error"
            }
        
        query = input_data.get("query", input_data.get("content", ""))
        context = input_data.get("context", "")
        
        self.log_action("Creating research plan", {"query": query})
        
        try:
            # Build the prompt
            user_message = f"Research Query: {query}\n"
            if context:
                user_message += f"\nAdditional Context: {context}\n"
            user_message += "\nPlease create a detailed research plan for answering this query."
            
            # Call LLM with fallback support
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
                
                plan_text = response.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                error_repr = repr(e)
                
                # Check if it's a connection error that might benefit from fallback
                # Check exception type, error message, and error representation
                is_connection_error = (
                    isinstance(e, APIConnectionError) or
                    "APIConnectionError" in error_type or
                    "ConnectError" in error_type or
                    "ConnectionError" in error_type or
                    any(keyword in error_msg.lower() for keyword in [
                        "ssl", "tls", "connection error", "connect", "timeout", 
                        "internal error", "tlsv1", "alert internal error", "connection"
                    ]) or
                    any(keyword in error_repr.lower() for keyword in [
                        "ssl", "tls", "connection", "connect"
                    ])
                )
                
                self.logger.warning(
                    f"LLM call failed with {self.model_provider} ({error_type}): {error_msg[:200]}"
                )
                
                if is_connection_error:
                    self.logger.info(
                        f"Detected connection error. Attempting fallback to other providers..."
                    )
                    
                    # Try fallback providers
                    try:
                        self.client = self._try_providers_with_fallback()
                        self.logger.info(f"Fallback successful, retrying with {self.model_provider}")
                        
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
                        plan_text = response.choices[0].message.content
                        self.logger.info(f"Successfully completed request using fallback provider: {self.model_provider}")
                    except Exception as fallback_error:
                        fallback_error_msg = str(fallback_error)
                        fallback_error_type = type(fallback_error).__name__
                        available = self._get_available_providers()
                        self.logger.error(
                            f"All providers failed.\n"
                            f"Original error ({error_type}): {error_msg[:200]}\n"
                            f"Fallback error ({fallback_error_type}): {fallback_error_msg[:200]}\n"
                            f"Available providers: {available}"
                        )
                        # Re-raise with more context
                        raise RuntimeError(
                            f"All LLM providers failed. Tried: {available}. "
                            f"Last error: {fallback_error_msg[:200]}"
                        ) from fallback_error
                else:
                    # Non-connection error, just raise it
                    self.logger.error(f"Non-connection error ({error_type}): {error_msg[:200]}")
                    raise
            
            # Extract plan components
            metadata = self._extract_plan_metadata(plan_text, query)
            
            # Check if plan is complete
            status = "complete" if "PLAN COMPLETE" in plan_text.upper() else "in_progress"
            
            self.log_action("Plan created", {"status": status, "steps": len(metadata.get("steps", []))})
            
            return {
                "output": plan_text,
                "metadata": metadata,
                "status": status
            }
            
        except Exception as e:
            self.logger.error(f"Error creating plan: {e}", exc_info=True)
            return {
                "output": f"Error creating research plan: {str(e)}",
                "metadata": {},
                "status": "error"
            }
    
    def _extract_plan_metadata(self, plan_text: str, query: str) -> Dict[str, Any]:
        """
        Extract structured metadata from the plan text.
        
        Args:
            plan_text: The plan text from the LLM
            query: Original query
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "query": query,
            "plan_text": plan_text,
            "steps": [],
            "topics": [],
            "search_queries": []
        }
        
        # Extract numbered steps
        lines = plan_text.split("\n")
        for line in lines:
            line = line.strip()
            # Look for numbered steps (1., 2., etc.)
            if line and (line[0].isdigit() or line.startswith("-")):
                metadata["steps"].append(line)
        
        # Extract potential search queries (lines with "search" or "query")
        for line in lines:
            line_lower = line.lower()
            if "search" in line_lower or "query" in line_lower:
                # Try to extract quoted text or text after colon
                if '"' in line:
                    import re
                    quotes = re.findall(r'"([^"]*)"', line)
                    metadata["search_queries"].extend(quotes)
                elif ":" in line:
                    query_part = line.split(":", 1)[1].strip()
                    if query_part:
                        metadata["search_queries"].append(query_part)
        
        return metadata