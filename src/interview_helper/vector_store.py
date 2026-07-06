"""Vector store implementation using ChromaDB."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
import os


logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class VectorStore:
    """Vector storage and retrieval using ChromaDB.

    Provides persistent vector storage for interview content with metadata filtering.
    """

    def __init__(self, collection_name: str = "interviews"):
        """Initialize the vector store.
            Args:
                collection_name: Name of the ChromaDB collection
        """
        import chromadb

        # Load ChromaDB path from environment variable with fallback to default
        chroma_db_path = Path(os.getenv("CHROMA_DB_PATH", "./chroma_db")).resolve()

        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(chroma_db_path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Interview content and metadata"}
         )
        logger.info(f"VectorStore initialized with collection: {self.collection_name} at {chroma_db_path}")

    @staticmethod
    def _sanitize_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce metadata into ChromaDB-compatible scalar values.

        ChromaDB only accepts str/int/float/bool metadata values, so list
        values (e.g. ``tags``) are joined into a comma-separated string and
        ``None`` values are dropped.
        """
        clean: Dict[str, Any] = {}
        for key, value in (meta or {}).items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                clean[key] = ", ".join(str(v) for v in value)
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            else:
                clean[key] = str(value)
        return clean

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        metadatas: List[Dict] = None,
     ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of document texts
            embeddings: Pre-computed embeddings for each document
            metadatas: Metadata dicts for each document

        Returns:
            List of added document IDs
        """
        logger.info(f"Adding {len(documents)} documents to vector store")

        if metadatas is None:
            metadatas = [doc.get("metadata", {}) for doc in documents]

        try:
            ids = []
            texts = []
            metadata_dicts = []

             # Prepare document text, embeddings and metadata. Use unique IDs
             # so that adding a new interview does not overwrite existing ones.
            import uuid

            for i, (doc, embedding, meta) in enumerate(zip(documents, embeddings, metadatas)):
                ids.append(uuid.uuid4().hex)
                texts.append(doc.get("text", "") if isinstance(doc, dict) else str(doc))
                metadata_dicts.append(self._sanitize_metadata(meta))

            logger.info(f"Adding {len(ids)} documents with embeddings")

             # Add to collection (documents must be passed so text is retrievable)
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadata_dicts,
             )

            logger.info(f"Successfully added {len(ids)} documents")

            return ids

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise

    def query(
        self,
        query_vector: List[float],
        where: Dict[str, Any] = None,
        top_k: int = 5,
     ) -> List[Dict]:
        """Query the vector store for similar documents.

        Args:
            query_vector: Embedding vector to search against
            where: Metadata filter (e.g., {"company": {"$eq": "Google"}})
            top_k: Number of results to return

        Returns:
            List of matching documents with metadata
        """
        logger.info(f"Querying vector store (top_k={top_k})")

        try:
            where_clause = where if where else None
            results = self.collection.query(
                query_embeddings=[query_vector],
                where=where_clause,
                include=["documents", "metadatas"],
                n_results=top_k,
             )

             # Structure the results
            matches = []
            if results and len(results) > 0:
                documents = results["documents"][0] if "documents" in results else []
                metadatas = results["metadatas"][0] if "metadatas" in results else []

                for doc, meta in zip(documents, metadatas):
                    match = {"text": doc if isinstance(doc, str) else ""}
                    match["metadata"] = meta or {}
                    matches.append(match)

            logger.info(f"Found {len(matches)} matches for query")

            return matches

        except Exception as e:
            logger.error(f"Query error: {e}")
            return []

    def get_all(self, limit: int = 100) -> List[Dict]:
        """Get all documents from the collection.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of all documents
        """
        logger.info(f"Getting all documents (limit={limit})")

        try:
            results = self.collection.get(
                limit=limit,
                include=["documents", "metadatas"],
             )

             # Structure the results
            matches = []
            if results and len(results) > 0:
                documents = results.get("documents", [])
                metadatas = results.get("metadatas", [])

                for doc, meta in zip(documents, metadatas):
                    match = {"text": doc if isinstance(doc, str) else "", "metadata": meta or {}}
                    matches.append(match)

            logger.info(f"Retrieved {len(matches)} documents")

            return matches

        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []
