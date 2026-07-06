"""Pluggable local LLM clients for Interview Helper.

This package defines a small provider abstraction so the RAG pipeline can work
with any local LLM server (Ollama, mlx_lm, or a custom implementation) without
changing the core code.

    from interview_helper.llm import create_llm_client

    client = create_llm_client()            # uses LLM_PROVIDER env (default: mlx)
    client = create_llm_client("ollama")    # explicit provider
    result = client.generate("Hello")
    print(result["response"])
"""

from .base import LLMClient
from .ollama_client import OllamaClient
from .mlx_client import MLXClient
from .factory import create_llm_client, register_provider, available_providers

__all__ = [
    "LLMClient",
    "OllamaClient",
    "MLXClient",
    "create_llm_client",
    "register_provider",
    "available_providers",
]
