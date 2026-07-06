"""Interview Helper - Personal Interview RAG System with ChromaDB and a
pluggable local LLM backend (Ollama, MLX, or custom)."""

__version__ = "0.1.0"
__author__ = "Ajaiswal1818"

from .core import InterviewHelper, QueryResult
from .vector_store import VectorStore
from .data_loader import DataLoader
from .llm import (
    LLMClient,
    OllamaClient,
    MLXClient,
    create_llm_client,
    register_provider,
    available_providers,
)
from .cli import interview_helper

__all__ = [
    "InterviewHelper",
    "QueryResult",
    "VectorStore",
    "DataLoader",
    "LLMClient",
    "OllamaClient",
    "MLXClient",
    "create_llm_client",
    "register_provider",
    "available_providers",
    "interview_helper",
]

