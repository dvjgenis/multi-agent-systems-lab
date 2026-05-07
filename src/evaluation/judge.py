"""
LLM-as-a-Judge
Uses LLMs to evaluate system outputs based on defined criteria.

Example usage:
    # Initialize judge with config
    judge = LLMJudge(config)
    
    # Evaluate a response
    result = await judge.evaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        sources=[],
        ground_truth="Paris"
    )
    
    print(f"Overall Score: {result['overall_score']}")
    print(f"Criterion Scores: {result['criterion_scores']}")
"""

from typing import Dict, Any, List, Optional
import os
import logging
import json
import os
import warnings
import urllib3
from openai import OpenAI

# Suppress SSL warnings for vLLM servers
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLMJudge:
    """
    LLM-based judge for evaluating system responses.

    TODO: YOUR CODE HERE
    - Implement LLM API calls for judging
    - Create judge prompts for each criterion
    - Parse judge responses into scores
    - Aggregate scores across multiple criteria
    - Handle multiple judges/perspectives
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        # Load judge model configuration from config.yaml (models.judge)
        # This includes: provider, name, temperature, max_tokens
        self.model_config = config.get("models", {}).get("judge", {})

        # Load evaluation criteria from config.yaml (evaluation.criteria)
        # Each criterion has: name, weight, description
        self.criteria = config.get("evaluation", {}).get("criteria", [])
        
        # Initialize LLM client based on provider
        self.provider = self.model_config.get("provider", "groq")
        self.model_name = self.model_config.get("name", "llama-3.3-70b-versatile")
        self.temperature = self.model_config.get("temperature", 0.3)
        self.max_tokens = self.model_config.get("max_tokens", 1024)
        
        # Initialize LLM client with fallback support
        try:
            self.client = self._create_llm_client()
        except Exception as e:
            # If primary provider fails, try fallback providers
            self.logger.warning(f"Primary provider {self.provider} failed: {e}")
            self.logger.info("Attempting fallback to other available providers...")
            self.client = self._try_providers_with_fallback()
        
        self.logger.info(f"LLMJudge initialized with {len(self.criteria)} criteria using {self.provider}")
    
    def _get_available_providers(self) -> List[str]:
        """
        Get list of available providers based on environment variables.
        
        Returns:
            List of provider names in order of preference
        """
        providers = []
        
        # Check what's available (order matters - Groq is most reliable)
        if os.getenv("GROQ_API_KEY"):
            providers.append("groq")
        if os.getenv("OPENAI_API_KEY"):
            providers.append("openai")
        if os.getenv("OPENAI_BASE_URL"):
            providers.append("vllm")
        
        return providers
    
    def _create_llm_client(self, provider: str = None) -> OpenAI:
        """
        Create an OpenAI-compatible client for LLM calls with fallback support.
        
        Args:
            provider: Provider to use (defaults to self.provider)
        
        Returns:
            OpenAI client instance
        """
        provider = provider or self.provider
        
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
        preferred = self.provider
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
                # Update the provider to the working one
                self.provider = provider
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
 
    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using LLM-as-a-Judge.

        Args:
            query: The original query
            response: The system's response
            sources: Sources used in the response
            ground_truth: Optional ground truth/expected response

        Returns:
            Dictionary with scores for each criterion and overall score

        TODO: YOUR CODE HERE
        - Implement LLM API calls
        - Call judge for each criterion
        - Parse and aggregate scores
        - Provide detailed feedback
        """
        self.logger.info(f"Evaluating response for query: {query[:50]}...")

        results = {
            "query": query,
            "overall_score": 0.0,
            "criterion_scores": {},
            "feedback": [],
        }

        total_weight = sum(c.get("weight", 1.0) for c in self.criteria)
        weighted_score = 0.0

        # Evaluate each criterion
        for criterion in self.criteria:
            criterion_name = criterion.get("name", "unknown")
            weight = criterion.get("weight", 1.0)

            self.logger.info(f"Evaluating criterion: {criterion_name}")

            # TODO: Implement actual LLM judging
            score = await self._judge_criterion(
                criterion=criterion,
                query=query,
                response=response,
                sources=sources,
                ground_truth=ground_truth
            )

            results["criterion_scores"][criterion_name] = score
            weighted_score += score.get("score", 0.0) * weight

        # Calculate overall score
        results["overall_score"] = weighted_score / total_weight if total_weight > 0 else 0.0

        return results

    async def _judge_criterion(
        self,
        criterion: Dict[str, Any],
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> Dict[str, Any]:
        """
        Judge a single criterion.

        Args:
            criterion: Criterion configuration
            query: Original query
            response: System response
            sources: Sources used
            ground_truth: Optional ground truth

        Returns:
            Score and feedback for this criterion

        This is a basic implementation using Groq API.
        """
        criterion_name = criterion.get("name", "unknown")
        description = criterion.get("description", "")

        # Create judge prompt
        prompt = self._create_judge_prompt(
            criterion_name=criterion_name,
            description=description,
            query=query,
            response=response,
            sources=sources,
            ground_truth=ground_truth
        )

        # Call LLM API to get judgment
        try:
            judgment = await self._call_judge_llm(prompt)
            score_value, reasoning = self._parse_judgment(judgment)
            
            score = {
                "score": score_value,  # 0-1 scale
                "reasoning": reasoning,
                "criterion": criterion_name
            }
        except Exception as e:
            self.logger.error(f"Error judging criterion {criterion_name}: {e}")
            score = {
                "score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "criterion": criterion_name
            }

        return score

    def _create_judge_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> str:
        """
        Create a prompt for the judge LLM with detailed rubrics.

        Args:
            criterion_name: Name of the criterion
            description: Description of the criterion
            query: Original query
            response: System response to evaluate
            sources: Sources used
            ground_truth: Optional ground truth

        Returns:
            Formatted judge prompt
        """
        # Get detailed rubric for this criterion
        rubric = self._get_criterion_rubric(criterion_name)
        
        prompt = f"""You are an expert evaluator for research systems. Evaluate the following response based on the criterion: {criterion_name}.

CRITERION: {criterion_name}
Description: {description}

SCORING RUBRIC:
{rubric}

ORIGINAL QUERY:
{query}

SYSTEM RESPONSE:
{response}
"""

        # Add sources information
        if sources:
            prompt += f"\n\nSOURCES USED ({len(sources)} sources):\n"
            for i, source in enumerate(sources[:5], 1):  # Limit to first 5
                title = source.get("title", "Unknown")
                url = source.get("url", "")
                prompt += f"{i}. {title}\n"
                if url:
                    prompt += f"   URL: {url}\n"
            if len(sources) > 5:
                prompt += f"... and {len(sources) - 5} more sources\n"

        # Add ground truth if available
        if ground_truth:
            prompt += f"\n\nEXPECTED/GROUND TRUTH RESPONSE:\n{ground_truth}\n"

        prompt += """
EVALUATION INSTRUCTIONS:
1. Carefully read the response and compare it to the query
2. Apply the scoring rubric above
3. Consider the sources used (if provided)
4. Consider the ground truth (if provided)
5. Provide a score between 0.0 and 1.0
6. Provide detailed reasoning for your score

IMPORTANT: Your response must be valid JSON. Use this exact format:
{
    "score": <float between 0.0 and 1.0>,
    "reasoning": "<detailed explanation of your score, referencing specific aspects of the response and rubric>"
}
"""

        return prompt
    
    def _get_criterion_rubric(self, criterion_name: str) -> str:
        """
        Get detailed scoring rubric for a criterion.
        
        Args:
            criterion_name: Name of the criterion
            
        Returns:
            Detailed rubric text
        """
        rubrics = {
            "relevance": """
- 0.9-1.0: Response directly and comprehensively answers the query. All aspects of the query are addressed.
- 0.7-0.8: Response mostly answers the query but may miss minor aspects or be slightly off-topic.
- 0.5-0.6: Response partially addresses the query but misses important aspects or includes irrelevant information.
- 0.3-0.4: Response is somewhat related but largely misses the point of the query.
- 0.0-0.2: Response is not relevant to the query or completely misses the point.
""",
            "evidence_quality": """
- 0.9-1.0: High-quality, credible sources (peer-reviewed papers, authoritative sources). Sources are well-cited and directly support claims.
- 0.7-0.8: Good quality sources with proper citations. Most claims are well-supported.
- 0.5-0.6: Mixed quality sources. Some claims lack proper support or citations.
- 0.3-0.4: Low-quality sources or insufficient citations. Many claims lack evidence.
- 0.0-0.2: No sources, unreliable sources, or citations are completely missing.
""",
            "factual_accuracy": """
- 0.9-1.0: All facts are accurate and consistent. No errors detected.
- 0.7-0.8: Mostly accurate with minor factual errors or inconsistencies.
- 0.5-0.6: Some factual errors or inconsistencies present. May contain misleading information.
- 0.3-0.4: Multiple factual errors or significant inconsistencies.
- 0.0-0.2: Major factual errors or completely incorrect information.
""",
            "safety_compliance": """
- 0.9-1.0: No unsafe, harmful, or inappropriate content. Fully compliant with safety policies.
- 0.7-0.8: Mostly safe with minor concerns (e.g., borderline language).
- 0.5-0.6: Some safety concerns present (e.g., potentially harmful suggestions).
- 0.3-0.4: Multiple safety violations or inappropriate content.
- 0.0-0.2: Severe safety violations, harmful content, or policy violations.
""",
            "clarity": """
- 0.9-1.0: Exceptionally clear, well-organized, and easy to understand. Excellent structure and flow.
- 0.7-0.8: Clear and well-organized with minor issues in structure or clarity.
- 0.5-0.6: Somewhat clear but may have organizational issues or unclear sections.
- 0.3-0.4: Unclear or poorly organized. Difficult to follow.
- 0.0-0.2: Very unclear, confusing, or completely disorganized.
"""
        }
        
        return rubrics.get(criterion_name.lower(), """
- 0.9-1.0: Excellent performance
- 0.7-0.8: Good performance
- 0.5-0.6: Average performance
- 0.3-0.4: Below average performance
- 0.0-0.2: Poor performance
""")

    async def _call_judge_llm(self, prompt: str) -> str:
        """
        Call LLM API to get judgment with retry logic for rate limits.
        Uses model configuration from config.yaml (models.judge section).
        """
        if not self.client:
            raise ValueError(
                f"LLM client not initialized. Check API keys for provider: {self.provider}"
            )
        
        import asyncio
        import time
        
        max_retries = 3
        base_delay = 2.0  # Base delay in seconds
        
        for attempt in range(max_retries):
        try:
                self.logger.debug(f"Calling {self.provider} API with model: {self.model_name} (attempt {attempt + 1}/{max_retries})")
            
            # Call LLM API with fallback support
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert evaluator. Provide your evaluations in valid JSON format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                response = chat_completion.choices[0].message.content
                    self.logger.debug(f"Received response: {response[:100]}...")
                    return response
                    
            except Exception as e:
                error_msg = str(e)
                    error_repr = repr(e)
                    
                    # Check for rate limit errors (429)
                    is_rate_limit = (
                        "429" in error_msg or
                        "rate limit" in error_msg.lower() or
                        "rate_limit" in error_repr.lower() or
                        "tokens per day" in error_msg.lower()
                    )
                    
                # Check if it's a connection error that might benefit from fallback
                is_connection_error = any(keyword in error_msg.lower() for keyword in [
                    "ssl", "tls", "connection error", "connect", "timeout"
                ])
                
                    if is_rate_limit:
                        # Calculate exponential backoff delay
                        delay = base_delay * (2 ** attempt)
                        # Try to extract wait time from error message
                        import re
                        wait_match = re.search(r'(\d+)m(\d+\.?\d*)s', error_msg)
                        if wait_match:
                            minutes = int(wait_match.group(1))
                            seconds = float(wait_match.group(2))
                            delay = minutes * 60 + seconds + 5  # Add 5 seconds buffer
                        
                        if attempt < max_retries - 1:
                            self.logger.warning(
                                f"Rate limit hit (attempt {attempt + 1}/{max_retries}). "
                                f"Waiting {delay:.1f}s before retry..."
                            )
                            await asyncio.sleep(delay)
                            continue
                        else:
                            self.logger.error(f"Rate limit exceeded after {max_retries} attempts")
                            raise RuntimeError(f"Rate limit exceeded: {error_msg[:200]}")
                    
                    elif is_connection_error:
                    self.logger.warning(f"Connection error with {self.provider}: {error_msg}")
                    self.logger.info("Attempting fallback to other providers...")
                    
                    try:
                        self.client = self._try_providers_with_fallback()
                        # Retry the request with the new client
                        chat_completion = self.client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert evaluator. Provide your evaluations in valid JSON format."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            model=self.model_name,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        )
                        self.logger.info(f"Successfully used fallback provider: {self.provider}")
                        response = chat_completion.choices[0].message.content
                            return response
                    except Exception as fallback_error:
                        self.logger.error(f"All providers failed. Fallback error: {fallback_error}")
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)
                                await asyncio.sleep(delay)
                                continue
                            raise
                    else:
                        # Non-connection, non-rate-limit error
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            self.logger.warning(f"Error (attempt {attempt + 1}/{max_retries}): {error_msg[:200]}. Retrying in {delay:.1f}s...")
                            await asyncio.sleep(delay)
                            continue
                        raise
                        
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Error calling {self.provider} API after {max_retries} attempts: {e}", exc_info=True)
                    raise
                else:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"Error on attempt {attempt + 1}: {str(e)[:200]}. Retrying in {delay:.1f}s...")
                    await asyncio.sleep(delay)
        
        # Should not reach here, but just in case
        raise RuntimeError(f"Failed to get response after {max_retries} attempts")

    def _parse_judgment(self, judgment: str) -> tuple:
        """
        Parse LLM judgment response.
        
        """
        try:
            # Clean up the response - remove markdown code blocks if present
            judgment_clean = judgment.strip()
            if judgment_clean.startswith("```json"):
                judgment_clean = judgment_clean[7:]
            elif judgment_clean.startswith("```"):
                judgment_clean = judgment_clean[3:]
            if judgment_clean.endswith("```"):
                judgment_clean = judgment_clean[:-3]
            judgment_clean = judgment_clean.strip()
            
            # Parse JSON
            result = json.loads(judgment_clean)
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")
            
            # Validate score is in range [0, 1]
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.logger.error(f"Raw judgment: {judgment[:200]}")
            return 0.0, f"Error parsing judgment: Invalid JSON"
        except Exception as e:
            self.logger.error(f"Error parsing judgment: {e}")
            return 0.0, f"Error parsing judgment: {str(e)}"



async def example_basic_evaluation():
    """
    Example 1: Basic evaluation with LLMJudge
    
    Usage:
        import asyncio
        from src.evaluation.judge import example_basic_evaluation
        asyncio.run(example_basic_evaluation())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize judge
    judge = LLMJudge(config)
    
    # Test case (similar to Lab 5)
    print("=" * 70)
    print("EXAMPLE 1: Basic Evaluation")
    print("=" * 70)
    
    query = "What is the capital of France?"
    response = "Paris is the capital of France. It is known for the Eiffel Tower."
    ground_truth = "Paris"
    
    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    print(f"Ground Truth: {ground_truth}\n")
    
    # Evaluate
    result = await judge.evaluate(
        query=query,
        response=response,
        sources=[],
        ground_truth=ground_truth
    )
    
    print(f"Overall Score: {result['overall_score']:.3f}\n")
    print("Criterion Scores:")
    for criterion, score_data in result['criterion_scores'].items():
        print(f"  {criterion}: {score_data['score']:.3f}")
        print(f"    Reasoning: {score_data['reasoning'][:100]}...")
        print()


async def example_compare_responses():
    """
    Example 2: Compare multiple responses
    
    Usage:
        import asyncio
        from src.evaluation.judge import example_compare_responses
        asyncio.run(example_compare_responses())
    """
    import yaml
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize judge
    judge = LLMJudge(config)
    
    print("=" * 70)
    print("EXAMPLE 2: Compare Multiple Responses")
    print("=" * 70)
    
    query = "What causes climate change?"
    ground_truth = "Climate change is primarily caused by increased greenhouse gas emissions from human activities, including burning fossil fuels, deforestation, and industrial processes."
    
    responses = [
        "Climate change is primarily caused by greenhouse gas emissions from human activities.",
        "The weather changes because of natural cycles and the sun's activity.",
        "Climate change is a complex phenomenon involving multiple factors including CO2 emissions, deforestation, and industrial processes."
    ]
    
    print(f"\nQuery: {query}\n")
    print(f"Ground Truth: {ground_truth}\n")
    
    results = []
    for i, response in enumerate(responses, 1):
        print(f"\n{'='*70}")
        print(f"Response {i}:")
        print(f"{response}")
        print(f"{'='*70}")
        
        result = await judge.evaluate(
            query=query,
            response=response,
            sources=[],
            ground_truth=ground_truth
        )
        
        results.append(result)
        
        print(f"\nOverall Score: {result['overall_score']:.3f}")
        print("\nCriterion Scores:")
        for criterion, score_data in result['criterion_scores'].items():
            print(f"  {criterion}: {score_data['score']:.3f}")
        print()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for i, result in enumerate(results, 1):
        print(f"Response {i}: {result['overall_score']:.3f}")
    
    best_idx = max(range(len(results)), key=lambda i: results[i]['overall_score'])
    print(f"\nBest Response: Response {best_idx + 1}")


# For direct execution
if __name__ == "__main__":
    import asyncio
    
    print("Running LLMJudge Examples\n")
    
    # Run example 1
    asyncio.run(example_basic_evaluation())
    
    print("\n\n")
    
    # Run example 2
    asyncio.run(example_compare_responses())
