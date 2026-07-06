"""Backwards-compatibility shim.

The Ollama client now lives in :mod:`interview_helper.llm.ollama_client` as one
of several pluggable providers. This module re-exports it so existing imports
(``from interview_helper.ollama_client import OllamaClient``) keep working.
"""

from .llm.ollama_client import OllamaClient

__all__ = ["OllamaClient"]
