"""Abstract base class for local LLM clients.

Any provider (Ollama, mlx_lm, llama.cpp, LM Studio, ...) can be supported by
subclassing :class:`LLMClient` and implementing :meth:`generate`. Providers are
generation-only; embeddings are handled separately by the RAG core using
sentence-transformers, so they stay provider-independent.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Common interface for local LLM generation backends.

    Subclasses talk to a specific server but expose the same
    :meth:`generate` contract so the rest of the app never depends on a
    particular provider.
    """

    #: Human-readable provider name, overridden by subclasses.
    provider_name: str = "base"

    def __init__(self, base_url: str, default_model: str):
        """Initialize the client.

        Args:
            base_url: Base URL of the local LLM server.
            default_model: Model name used when ``generate`` is called without
                an explicit ``model`` argument.
        """
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        logger.info(
            f"{self.provider_name} client initialized at {self.base_url} "
            f"(default model: {self.default_model})"
        )

    def _resolve_model(self, model: Optional[str]) -> str:
        """Return the explicit model or fall back to the configured default."""
        return model or self.default_model

    @abstractmethod
    def generate(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Generate a completion for ``prompt``.

        Args:
            prompt: The prompt/user message to send to the model.
            model: Optional model override; defaults to ``self.default_model``.

        Returns:
            A dict with at least a ``"response"`` string key. Implementations
            should also populate ``"usage"`` and ``"duration"`` when available.
        """
        raise NotImplementedError

    def list_models(self) -> List[Dict]:
        """List models available on the server.

        Optional; providers that cannot enumerate models may leave this
        unimplemented.
        """
        raise NotImplementedError(
            f"{self.provider_name} client does not support listing models"
        )
