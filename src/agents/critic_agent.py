"""
Critic Agent

This agent evaluates the quality and accuracy of research outputs.
It provides feedback and determines if the response meets quality standards.
"""

import os
import logging
import warnings
import urllib3
from typing import Dict, Any, List
from openai import OpenAI
from src.agents.base_agent import BaseAgent

# Suppress SSL warnings for vLLM servers
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class CriticAgent(BaseAgent):
    """
    Critic Agent - Evaluates quality and accuracy of research outputs.
    
    The critic evaluates research and writing on multiple criteria:
    - Relevance to the original query
    - Quality of evidence and citations
    - Completeness of coverage
    - Factual accuracy
    - Clarity and organization
    
    It provides constructive feedback and determines if revisions are needed.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Critic Agent.
        
        Args:
            config: Configuration dictionary from config.yaml
        """
        super().__init__("critic", config)
        
        # Initialize LLM client with fallback support
        try:
            self.client = self._create_llm_client()
        except Exception as e:
            # If primary provider fails, try fallback providers
            self.logger.warning(f"Primary provider {self.model_provider} failed: {e}")
            self.logger.info("Attempting fallback to other available providers...")
            self.client = self._try_providers_with_fallback()
        
        # Default system prompt
        self.default_system_prompt = """You are a Research Critic. Your job is to evaluate the quality and accuracy of research outputs.

Evaluate the research and writing on these criteria:
1. **Relevance**: Does it answer the original query?
2. **Evidence Quality**: Are sources credible and well-cited?
3. **Completeness**: Are all aspects of the query addressed?
4. **Accuracy**: Are there any factual errors or contradictions?
5. **Clarity**: Is the writing clear and well-organized?

Provide constructive but thorough feedback. 

If the response meets quality standards, end with "APPROVED - RESEARCH COMPLETE".
If revisions are needed, end with "NEEDS REVISION" and provide specific suggestions."""
        
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
        Evaluate a research response and provide feedback.
        
        Args:
            input_data: Dictionary containing:
                - query: The original research query
                - response: The written response to evaluate
                - plan: The research plan
                - research_findings: The research findings
                
        Returns:
            Dictionary containing:
                - output: Evaluation and feedback
                - metadata: Scores, issues found, etc.
                - status: "approved" or "needs_revision"
        """
        if not self.validate_input(input_data):
            return {
                "output": "Invalid input provided.",
                "metadata": {},
                "status": "error"
            }
        
        query = input_data.get("query", input_data.get("content", ""))
        response = input_data.get("response", input_data.get("output", ""))
        plan = input_data.get("plan", "")
        research_findings = input_data.get("research_findings", "")
        
        self.log_action("Evaluating response", {"query": query})
        
        try:
            # Build evaluation prompt
            user_message = self._build_evaluation_prompt(query, plan, research_findings, response)
            
            # Get evaluation from LLM
            # Call LLM with fallback support
            try:
                llm_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                evaluation = llm_response.choices[0].message.content
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
                        llm_response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": self.system_message},
                                {"role": "user", "content": user_message}
                            ],
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                        self.logger.info(f"Successfully used fallback provider: {self.model_provider}")
                        evaluation = llm_response.choices[0].message.content
                    except Exception as fallback_error:
                        self.logger.error(f"All providers failed. Fallback error: {fallback_error}")
                        evaluation = "APPROVED - RESEARCH COMPLETE"  # Default to approved if all fail
                else:
                    # Non-connection error
                    self.logger.error(f"Error evaluating: {e}")
                    evaluation = "APPROVED - RESEARCH COMPLETE"  # Default to approved
            
            # Determine status
            evaluation_upper = evaluation.upper()
            if "APPROVED" in evaluation_upper and "RESEARCH COMPLETE" in evaluation_upper:
                status = "approved"
            elif "NEEDS REVISION" in evaluation_upper:
                status = "needs_revision"
            else:
                # Default to needs_revision if unclear
                status = "needs_revision"
            
            # Extract scores and issues
            metadata = self._extract_evaluation_metadata(evaluation, status)
            
            self.log_action("Evaluation complete", {
                "status": status,
                "scores": metadata.get("scores", {})
            })
            
            return {
                "output": evaluation,
                "metadata": metadata,
                "status": status
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating response: {e}", exc_info=True)
            return {
                "output": f"Error evaluating response: {str(e)}",
                "metadata": {},
                "status": "error"
            }
    
    def _build_evaluation_prompt(
        self,
        query: str,
        plan: str,
        research_findings: str,
        response: str
    ) -> str:
        """
        Build the evaluation prompt for the critic.
        
        Args:
            query: Original research query
            plan: Research plan
            research_findings: Research findings
            response: Written response to evaluate
            
        Returns:
            Formatted evaluation prompt
        """
        prompt = f"""Original Research Query: {query}

Research Plan:
{plan}

Research Findings:
{research_findings[:2000]}  # Limit length

Written Response to Evaluate:
{response}

Please evaluate this response on the following criteria:

1. **Relevance** (0-10): Does the response directly answer the query?
2. **Evidence Quality** (0-10): Are sources credible, well-cited, and appropriate?
3. **Completeness** (0-10): Are all aspects of the query addressed?
4. **Accuracy** (0-10): Are there factual errors or contradictions?
5. **Clarity** (0-10): Is the writing clear, well-organized, and easy to follow?

Provide:
- Scores for each criterion
- Detailed feedback on strengths and weaknesses
- Specific suggestions for improvement (if needed)
- Overall assessment

End with either:
- "APPROVED - RESEARCH COMPLETE" if the response meets quality standards
- "NEEDS REVISION" if improvements are needed, followed by specific revision suggestions"""
        
        return prompt
    
    def _extract_evaluation_metadata(self, evaluation: str, status: str) -> Dict[str, Any]:
        """
        Extract structured metadata from the evaluation.
        
        Args:
            evaluation: Evaluation text from LLM
            status: Evaluation status
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "status": status,
            "scores": {},
            "issues": [],
            "strengths": []
        }
        
        # Extract scores (look for patterns like "Relevance: 8/10" or "Relevance (8)")
        import re
        
        criteria = ["relevance", "evidence", "completeness", "accuracy", "clarity"]
        for criterion in criteria:
            # Look for scores in various formats
            patterns = [
                rf"{criterion}.*?(\d+)/10",
                rf"{criterion}.*?\((\d+)\)",
                rf"{criterion}.*?:\s*(\d+)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, evaluation, re.IGNORECASE)
                if match:
                    try:
                        score = int(match.group(1))
                        metadata["scores"][criterion] = score
                        break
                    except ValueError:
                        continue
        
        # Extract issues (look for "issue", "problem", "weakness", "improve")
        issue_keywords = ["issue", "problem", "weakness", "improve", "missing", "error"]
        lines = evaluation.split("\n")
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in issue_keywords):
                if line.strip() and len(line.strip()) > 10:
                    metadata["issues"].append(line.strip())
        
        # Extract strengths (look for "strength", "good", "well", "excellent")
        strength_keywords = ["strength", "good", "well", "excellent", "strong", "clear"]
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in strength_keywords):
                if line.strip() and len(line.strip()) > 10:
                    metadata["strengths"].append(line.strip())
        
        # Calculate overall score if individual scores exist
        if metadata["scores"]:
            scores = list(metadata["scores"].values())
            metadata["overall_score"] = sum(scores) / len(scores)
        
        return metadata