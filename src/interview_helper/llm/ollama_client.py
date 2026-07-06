"""LLM client for a local Ollama server.

Talks to Ollama's native API (``/api/generate`` and ``/api/tags``) which
listens on ``http://localhost:11434`` by default.

Start it with, for example::

    ollama serve
    ollama pull llama3
"""

import logging
from typing import Any, Dict, List, Optional

from .base import LLMClient


logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3"


class OllamaClient(LLMClient):
    """Client for interacting with a local Ollama server."""

    provider_name = "ollama"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        # Backwards-compatible alias for the previous positional argument name.
        ollama_url: Optional[str] = None,
    ):
        super().__init__(base_url=ollama_url or base_url, default_model=default_model)

    # Preserve the old attribute name used elsewhere in the codebase.
    @property
    def ollama_url(self) -> str:
        return self.base_url

    def generate(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Generate text using Ollama's completion API."""
        model = self._resolve_model(model)
        logger.info(f"Generating with ollama model: {model}")

        payload = {"prompt": prompt, "model": model, "stream": False}

        try:
            import requests

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=1200,
            )
            response.raise_for_status()
            data = response.json()

            return {
                "response": data.get("response", ""),
                "usage": {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                },
                "duration": data.get("total_duration", 0),
            }

        except Exception as e:
            logger.error(f"Failed to generate with Ollama: {e}")
            raise

    def list_models(self) -> List[Dict]:
        """List all available models on the Ollama server."""
        logger.info("Listing available Ollama models")
        try:
            import requests

            response = requests.get(
                f"{self.base_url}/api/tags",
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise
