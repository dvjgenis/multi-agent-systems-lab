"""
Base Agent Class

This module provides a base class for all research agents in the system.
It defines common interfaces and functionality that all agents share.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import os


class BaseAgent(ABC):
    """
    Base class for all research agents.
    
    This class provides common functionality and defines the interface
    that all specialized agents must implement.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name/identifier
            config: Configuration dictionary from config.yaml
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"agents.{name.lower()}")
        
        # Agent-specific configuration
        agent_config = config.get("agents", {}).get(name.lower(), {})
        self.enabled = agent_config.get("enabled", True)
        self.system_prompt = agent_config.get("system_prompt", "")
        
        # Model configuration
        model_config = config.get("models", {}).get("default", {})
        self.model_provider = model_config.get("provider", "groq")
        self.model_name = model_config.get("name", "llama-3.3-70b-versatile")
        self.temperature = model_config.get("temperature", 0.7)
        self.max_tokens = model_config.get("max_tokens", 2048)
    
    @abstractmethod
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        This method must be implemented by each specialized agent.
        
        Args:
            input_data: Dictionary containing input data (query, context, etc.)
            
        Returns:
            Dictionary containing:
            - output: The agent's output/response
            - metadata: Additional information (sources, citations, etc.)
            - status: Processing status ("complete", "in_progress", "error")
        """
        pass
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data before processing.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        if not isinstance(input_data, dict):
            self.logger.error("Input data must be a dictionary")
            return False
        
        if "query" not in input_data and "content" not in input_data:
            self.logger.warning("Input data missing 'query' or 'content' field")
            return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "model_provider": self.model_provider,
            "model_name": self.model_name,
        }
    
    def log_action(self, action: str, details: Optional[Dict[str, Any]] = None):
        """
        Log an agent action for debugging and traceability.
        
        Args:
            action: Description of the action
            details: Additional details about the action
        """
        if details:
            self.logger.info(f"{action}: {details}")
        else:
            self.logger.info(action)
    
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