"""Factory and registry for local LLM providers.

Selection precedence for the provider:
    explicit ``provider`` argument > ``LLM_PROVIDER`` env var > ``DEFAULT_PROVIDER``.

Environment variables:
    LLM_PROVIDER   Provider name ("mlx", "ollama", or a registered custom one).
    LLM_BASE_URL   Override the server URL for the selected provider.
    LLM_MODEL      Override the default model (falls back to OLLAMA_MODEL).
"""

import logging
import os
from typing import Dict, Optional, Type

from .base import LLMClient
from .ollama_client import OllamaClient
from .mlx_client import MLXClient


logger = logging.getLogger(__name__)

# The user is on Apple Silicon, so mlx is the out-of-the-box default.
DEFAULT_PROVIDER = "mlx"

# Registry of provider name -> client class. Extend via register_provider().
_PROVIDERS: Dict[str, Type[LLMClient]] = {
    "ollama": OllamaClient,
    "mlx": MLXClient,
}


def register_provider(name: str, client_cls: Type[LLMClient]) -> None:
    """Register a custom LLM client so it can be selected by name.

    Args:
        name: Provider identifier used by LLM_PROVIDER / ``provider`` arg.
        client_cls: An :class:`LLMClient` subclass.
    """
    if not issubclass(client_cls, LLMClient):
        raise TypeError("client_cls must be a subclass of LLMClient")
    _PROVIDERS[name.lower()] = client_cls
    logger.info(f"Registered LLM provider: {name}")


def available_providers() -> list:
    """Return the list of registered provider names."""
    return sorted(_PROVIDERS)


def create_llm_client(
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> LLMClient:
    """Create an LLM client for the selected provider.

    Args:
        provider: Provider name; if None, uses ``LLM_PROVIDER`` env, then the
            built-in default.
        base_url: Optional server URL override (else ``LLM_BASE_URL`` env, else
            the provider's own default).
        model: Optional default model override (else ``LLM_MODEL`` /
            ``OLLAMA_MODEL`` env, else the provider's own default).

    Returns:
        A ready-to-use :class:`LLMClient` instance.
    """
    provider = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Available: {', '.join(available_providers())}"
        )

    client_cls = _PROVIDERS[provider]

    base_url = base_url or os.getenv("LLM_BASE_URL")
    model = model or os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL")

    # Only pass overrides that were actually provided so each client keeps its
    # own sensible defaults for anything left unset.
    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if model:
        kwargs["default_model"] = model

    logger.info(f"Creating LLM client for provider: {provider}")
    return client_cls(**kwargs)
