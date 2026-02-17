"""LLM provider module - supports OpenAI, Anthropic, and Ollama."""

from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


class LLMProvider:
    """Abstract LLM provider interface."""
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 500):
        """
        Initialize LLM provider.
        
        Args:
            model_name: Model identifier
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate(self, prompt: str) -> str:
        """
        Generate response from prompt.
        
        Args:
            prompt: Input prompt
        
        Returns:
            Generated text response
        """
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", **kwargs):
        super().__init__(model_name, **kwargs)
        
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI provider with model {model_name}")
    
    def generate(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", **kwargs):
        super().__init__(model_name, **kwargs)
        
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = Anthropic(api_key=api_key)
        logger.info(f"Initialized Anthropic provider with model {model_name}")
    
    def generate(self, prompt: str) -> str:
        """Generate response using Anthropic API."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text


class OllamaProvider(LLMProvider):
    """Ollama local model provider."""
    
    def __init__(self, model_name: str = "llama2", **kwargs):
        super().__init__(model_name, **kwargs)
        
        try:
            import requests
        except ImportError:
            raise ImportError("requests package required. Install with: pip install requests")
        
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.requests = requests
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            response.raise_for_status()
            logger.info(f"Initialized Ollama provider with model {model_name}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama at {self.base_url}: {e}")
    
    def generate(self, prompt: str) -> str:
        """Generate response using Ollama local API."""
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]


def create_llm_provider(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Factory function to create LLM provider from environment config.
    
    Args:
        provider_type: "openai", "anthropic", or "ollama" (defaults to env var LLM_PROVIDER)
        model_name: Model name (defaults to env var LLM_MODEL)
        **kwargs: Additional provider arguments
    
    Returns:
        Configured LLM provider instance
    
    Example .env:
        LLM_PROVIDER=anthropic
        LLM_MODEL=claude-3-5-sonnet-20241022
        ANTHROPIC_API_KEY=sk-ant-...
    """
    provider_type = provider_type or os.getenv("LLM_PROVIDER", "openai")
    
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "ollama": OllamaProvider
    }
    
    if provider_type not in providers:
        raise ValueError(
            f"Unknown provider: {provider_type}. "
            f"Supported: {list(providers.keys())}"
        )
    
    provider_class = providers[provider_type]
    
    # Use model_name if provided, otherwise use env var or provider default
    if model_name:
        return provider_class(model_name=model_name, **kwargs)
    else:
        model_from_env = os.getenv("LLM_MODEL")
        if model_from_env:
            return provider_class(model_name=model_from_env, **kwargs)
        else:
            return provider_class(**kwargs)
