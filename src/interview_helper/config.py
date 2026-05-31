"""Configuration management for Interview Helper."""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field


class AppConfig:
    """Application configuration manager."""

    def __init__(self):
        self.data_dir = Path.home() / ".interview_helper" if os.getenv("HOME") else Path("/tmp/interview_helper")
        self.chroma_db_path = self.data_dir / "chroma_db"
        self.ollama_base_url: Optional[str] = None
        self.default_model = os.getenv("OLLAMA_MODEL", "llama3").strip()
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-minist:v6").strip()

    def set_ollama_url(self, url: str) -> None:
        """Set the Ollama API URL."""
        self.ollama_base_url = url.strip("/")


class VectorStoreConfig:
    """Vector database configuration for ChromaDB."""

    persist_path: str = Field(default_factory=lambda: str(AppConfig().chroma_db_path))
    collection_name: str = "interviews"

    @classmethod
    def create_collection_name(cls, collection_suffix: str = None) -> str:
        """Create a unique collection name with optional suffix."""
        return collection_suffix or cls.collection_name
