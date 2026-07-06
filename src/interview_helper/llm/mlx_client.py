"""LLM client for Apple's ``mlx_lm`` server (Apple Silicon).

``mlx_lm.server`` exposes an OpenAI-compatible API (``/v1/chat/completions``
and ``/v1/models``) and listens on ``http://localhost:8080`` by default.

Start it with, for example::

    mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit
"""

import logging
from typing import Any, Dict, List, Optional

from .base import LLMClient


logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


class MLXClient(LLMClient):
    """Client for the mlx_lm OpenAI-compatible server."""

    provider_name = "mlx"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
    ):
        super().__init__(base_url=base_url, default_model=default_model)
        self.temperature = temperature

    def generate(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Generate text via the OpenAI-compatible chat completions endpoint."""
        model = self._resolve_model(model)
        logger.info(f"Generating with mlx model: {model}")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False,
        }

        try:
            import requests

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=1200,
            )
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            text = ""
            if choices:
                message = choices[0].get("message") or {}
                text = message.get("content", "") or choices[0].get("text", "")

            usage = data.get("usage", {}) or {}
            return {
                "response": text,
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
                "duration": 0,
            }

        except Exception as e:
            logger.error(f"Failed to generate with mlx_lm: {e}")
            raise

    def list_models(self) -> List[Dict]:
        """List models via the OpenAI-compatible ``/v1/models`` endpoint."""
        logger.info("Listing available mlx models")
        try:
            import requests

            response = requests.get(
                f"{self.base_url}/v1/models",
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            logger.error(f"Failed to list mlx models: {e}")
            raise
