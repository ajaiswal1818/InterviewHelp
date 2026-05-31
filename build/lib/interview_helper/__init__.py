"""Interview Helper - Personal Interview RAG System with ChromaDB and Ollama Integration"""

__version__ = "0.1.0"
__author__ = "Ajaiswal1818"

from .core import InterviewHelper, QueryResult
from .vector_store import VectorStore
from .data_loader import DataLoader
from .cli import interview_helper

