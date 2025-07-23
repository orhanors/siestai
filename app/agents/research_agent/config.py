"""Configuration for the Research Agent."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ResearchAgentConfig:
    """Configuration class for the Research Agent."""
    
    # Model settings
    model: str = "gpt-4"
    temperature: float = 0.1
    
    # Retrieval settings
    max_documents: int = 5
    max_web_results: int = 3
    vector_search_threshold: float = 0.7
    
    # Feature toggles
    enable_kg: bool = True
    enable_web_search: bool = True
    enable_hybrid_search: bool = False
    
    # Search weights (for hybrid search)
    text_weight: float = 0.3
    vector_weight: float = 0.7
    
    # Tavily settings
    tavily_search_depth: str = "advanced"  # "basic" or "advanced"
    
    # Environment variables
    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load environment variables if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            
        if self.tavily_api_key is None:
            self.tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    def validate(self) -> bool:
        """Validate the configuration."""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required")
            
        if self.enable_web_search and not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required when web search is enabled")
            
        if self.max_documents <= 0:
            raise ValueError("max_documents must be positive")
            
        if self.max_web_results <= 0:
            raise ValueError("max_web_results must be positive")
            
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
            
        return True


# Default configuration
DEFAULT_CONFIG = ResearchAgentConfig()