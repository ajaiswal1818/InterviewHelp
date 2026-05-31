
"""Data loading and processing utilities for Interview Helper."""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class InterviewInfo:
    """Structured interview information."""
    title: str
    date: str
    company: Optional[str] = None
    role: Optional[str] = None
    location: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass  
class DocumentChunk:
    """A chunked segment of document text for vectorization."""
    content: str
    chunk_index: int
    parent_metadata: Dict


class DataLoader:
    """Handles loading and processing of interview data into chunks for RAG."""

    def __init__(self, text: str, document_type: str, metadata: dict = None, chunk_size: int = 500, chunk_overlap: int = 30):
        self.text = text.strip()
        self.document_type = document_type
        self.metadata = metadata or {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self) -> List[Dict]:
        """Load and process the data into structured chunks."""
        logger.info(f"Loading {self.document_type} document")
        
        # Clean and structure the content
        processed_text = self._clean_text()
        
        # Chunk the text for better retrieval
        chunks = self._chunk_text(processed_text)
        
        # Build metadata dictionary
        full_metadata = {"type": self.document_type, **self.metadata}
        
        # Create chunks with metadata
        documents = []
        for idx, chunk in enumerate(chunks):
            document = {
                "text": chunk["content"],
                "metadata": {**full_metadata},
            }
            documents.append(document)

        return documents

    def _clean_text(self) -> str:
        """Clean and normalize the input text."""
        cleaned = (self.text.replace("\x00", " ").expandtabs()).strip()
        return cleaned

    def _chunk_text(self, text: str) -> List[Dict[str, str]]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            
            # If we reached the end, include this chunk even if it's smaller
            if end >= len(text) or chunk_text:
                chunks.append({"content": chunk_text})
            
            # Move start with overlap for next chunk
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def extract_tags(self) -> List[str]:
        """Extract relevant tags from the content."""
        if not self.metadata.get("tags"):
            return ["interview"]
        
        return self.metadata["tags"]
