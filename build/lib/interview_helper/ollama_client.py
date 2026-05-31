
"""Ollama API client for interview helper."""

import logging
from typing import List, Dict, Any


logger = logging.getLogger(__name__)


class OllamaClient:
     """Client for interacting with Ollama API."""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
         """Initialize Ollama client.

        Args:
            ollama_url: Ollama API server URL (default: localhost:11434)
         """
        self.ollama_url = ollama_url.rstrip("/")
        logger.info(f"Ollama client initialized at {self.ollama_url}")

    def generate(self, prompt: str, model: str) -> Dict[str, Any]:
           """Generate text using Ollama completion API.

        Args:
            prompt: Prompt to send to the model
            model: Name of the model to use

        Returns:
            Dictionary with response and metadata
         """
        logger.info(f"Generating with model: {model}")

        payload = {
             "prompt": prompt,
              "model": model,
               "stream": False
          }

        try:
            import requests
            from httpx import RequestError

             # Make request to Ollama generate endpoint
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                 json=payload,
                  headers={"Content-Type": "application/json"},
                   timeout=120
               )

            response.raise_for_status()
             data = response.json()

            return {
                "response": data.get("response", ""),
                "usage": {
                    "prompt_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("eval_count", 0)
                 },
                "duration": data.get("duration", 0)
             }

         except RequestError as e:
            logger.error(f"Failed to generate with Ollama: {e}")
            raise

    def create_embedding(self, text: str, model: str = None) -> Dict[str, Any]:
        """Create vector embeddings for text using Ollama.

        Args:
            text: Text to embed
            model: Name of the embedding model (default: uses same as generate)

        Returns:
            Dictionary with embeddings and metadata
         """
        logger.info(f"Creating embedding with model: {model}")

        payload = {
             "prompt": text,
              "model": model or self.ollama_url.split("/")[-1]
               "keep_alive": -1   # Don't keep image around after generation
          }

        try:
            import requests

            response = requests.post(
                f"{self.ollama_url}/api/embed",
                 json=payload,
                  headers={"Content-Type": "application/json"},
                   timeout=120
               )

            response.raise_for_status()
             data = response.json()

             return {
                 "embedding": data.get("embedding", []),
                "model_used": model or self.ollama_url.split("/")[-1]
             }

        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            raise

    def list_models(self) -> List[Dict]:
           """List all available models in Ollama.

        Returns:
            List of available models with metadata
         """
        logger.info("Listing available Ollama models")

        try:
            import requests

            response = requests.post(
                f"{self.ollama_url}/api/tags",
                 headers={"Content-Type": "application/json"},
                  timeout=30
             )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            raise


if __name__ == "__main__":
     client = OllamaClient()