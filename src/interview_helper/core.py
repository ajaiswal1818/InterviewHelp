"""Core InterviewHelper class - orchestrates the RAG pipeline."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from .llm import LLMClient, create_llm_client


logger = logging.getLogger(__name__)

# Local embedding model shipped with the project, resolved relative to this
# file so it works regardless of the current working directory.
LOCAL_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "local_all-MiniLM-L6-v2"


@dataclass
class QueryResult:
    """Result from a search or query."""
    documents: List[Dict[str, Any]]
    relevance_scores: Optional[List[float]] = None
    query: str = ""


class InterviewHelper:
    """Main orchestrator for the interview helper RAG system.

    Provides methods to store interviews and answer questions about them.

    Usage:
         >>> helper = InterviewHelper()
         >>> helper.add_interview(content, metadata={})
         >>> answer = helper.ask("What were my technical strengths?")
    """

    def __init__(self, llm_client: Optional[LLMClient] = None, ollama_client: Any = None):
        """Initialize InterviewHelper with an optional custom LLM client.

        Args:
            llm_client: Any :class:`LLMClient` (Ollama, MLX, or custom). If
                None, one is built from configuration via ``create_llm_client``
                (default provider: mlx).
            ollama_client: Deprecated alias for ``llm_client``, kept for
                backwards compatibility.
        """
        client = llm_client or ollama_client
        self.llm_client = client if client is not None else create_llm_client()
        logger.info(
            f"InterviewHelper initialized with provider: "
            f"{getattr(self.llm_client, 'provider_name', 'custom')}"
        )

    @property
    def ollama_client(self) -> LLMClient:
        """Deprecated alias for :attr:`llm_client`."""
        return self.llm_client

    def add_interview(self, content: str, metadata: Dict[str, Any], chunk_size: int = 500) -> List[str]:
        """Add a new interview to the vector database.

        Args:
            content: Interview text content
            metadata: Interview metadata (date, company, role, tags, etc.)
            chunk_size: Size of chunks in characters

        Returns:
            List of added document IDs
        """
        from .data_loader import DataLoader

        logger.info(f"Adding interview with metadata: {metadata}")

        # Initialize data loader
        data_loader = DataLoader(
            text=content,
            document_type="interview",
            metadata=metadata,
            chunk_size=chunk_size,
            chunk_overlap=30
        )

        # Load and process text into chunks
        documents = data_loader.load()
        logger.info(f"Created {len(documents)} document chunks")

        # Generate embeddings, keeping documents and embeddings aligned by
        # dropping any chunk whose embedding could not be generated.
        kept_documents = []
        embeddings = []

        for doc in documents:
            embedding = self._generate_embedding(doc["text"])
            if embedding:
                kept_documents.append(doc)
                embeddings.append(embedding)

        logger.info(f"Generated {len(embeddings)} embeddings")

        if not embeddings:
            logger.error("No embeddings were generated; nothing to store")
            return []

        # Add documents to vector store
        from .vector_store import VectorStore

        collection = VectorStore()
        added_docs = collection.add_documents(
            documents=kept_documents,
            embeddings=embeddings,
            metadatas=[doc["metadata"] for doc in kept_documents]
        )

        logger.info(f"Added {len(added_docs)} documents to vector store")

        return added_docs

    _embedding_model = None

    @classmethod
    def _get_embedding_model(cls):
        """Load (and cache) the sentence-transformers embedding model.

        Uses the local model bundled with the project when available, and
        falls back to downloading ``all-MiniLM-L6-v2`` from Hugging Face.
        """
        if cls._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            model_ref = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else "all-MiniLM-L6-v2"
            logger.info(f"Loading embedding model from: {model_ref}")
            cls._embedding_model = SentenceTransformer(model_ref)
        return cls._embedding_model

    def _generate_embedding(self, text: str) -> Optional[Any]:
        """Generate embedding for a text using sentence-transformers.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding or None on error
        """
        try:
            model = self._get_embedding_model()
            embedding = model.encode([text])
            return embedding[0].tolist() if hasattr(embedding[0], 'tolist') else list(embedding[0])
        except Exception as e:
            logger.error(f"Failed to generate embedding for text (length {len(text)}): {e}")
            return None

    def search(self, query: str, where_clause: Dict[str, Any] = None, top_k: int = 5) -> QueryResult:
        """Search for relevant content using the query.

        Args:
            query: Search query string
            where_clause: Optional metadata filters (e.g., company="Google")
            top_k: Number of results to return

        Returns:
            QueryResult with matched documents and relevance scores
        """
        logger.info(f"Searching for: {query}")

        # Generate embedding for query
        query_embedding = self._generate_embedding(query)

        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return QueryResult(documents=[], query=query)

        # Search vector store with optional filters
        from .vector_store import VectorStore

        collection = VectorStore()

        try:
            results = collection.query(
                query_vector=query_embedding,
                where=where_clause or {},
                top_k=top_k
            )

            return QueryResult(documents=results, query=query)
        except Exception as e:
            logger.error(f"Search error: {e}")
            return QueryResult(documents=[], query=query)

    def ask(self, question: str, company_name: str = None) -> str:
        """Ask a question about interviews using RAG.

        Args:
            question: The question to ask
            company_name: Optional company filter

        Returns:
            Answer generated by the LLM
        """
        logger.info(f"Asking question: {question}")

        # Perform search for context
        where_filter = {"company": company_name} if company_name else None

        results = self.search(" ".join([question]), where_clause=where_filter)

        if not results.documents:
            return f"No relevant context found for: {question}"

        # Prepare prompt with retrieved context
        system_prompt = (
             "You are an interview helper assistant. "
             "Answer questions based on the context provided from the user's interviews. "
             "Be concise, helpful, and cite specific details from the context."
         )

        context = results.documents[0].get('text', '') if results.documents else 'No context available'
        prompt = f"""{system_prompt}

Context retrieved from interview database:
{context[:1000]}

Question: {question}"""

        logger.info("Generating response with LLM")
        logger.info(f"Prompt for LLM:\n{prompt}")
        try:
            # Generate using the configured provider's default model
            result = self.llm_client.generate(prompt=prompt)
            return result.get("response", "No answer available")
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"Error generating response: {str(e)}"

    def get_all_interviews(self) -> List[Dict]:
        """Get all stored interviews.

        Returns:
            List of all interview documents with metadata
        """
        from .vector_store import VectorStore

        logger.info("Retrieving all interviews from vector store")

        collection = VectorStore()

        try:
            results = collection.get_all(limit=100)

            return sorted(results, key=lambda x: x.get("metadata", {}).get("date", ""))
        except Exception as e:
            logger.error(f"Failed to retrieve all interviews: {e}")
            return []
