"""Core InterviewHelper class - orchestrates the RAG pipeline."""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from .ollama_client import OllamaClient


logger = logging.getLogger(__name__)


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

    def __init__(self, ollama_client: Any = None):
        """Initialize InterviewHelper with optional custom Ollama client.

        Args:
            ollama_client: Custom Ollama client instance or None for default
        """

        if ollama_client is None:
            self.ollama_client = OllamaClient()
        else:
            self.ollama_client = ollama_client
        logger.info("InterviewHelper initialized")

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

        # Generate embeddings for all chunks
        embeddings = []
        ids = []

        for i, doc in enumerate(documents):
            embedding = self._generate_em.bedding(doc["text"])
            if embedding:
                embeddings.append(embedding)
            ids.append(str(i))

        logger.info(f"Generated {len(embeddings)} embeddings")

        # Add documents to vector store
        from .vector_store import VectorStore

        collection = VectorStore()
        added_docs = collection.add_documents(
            documents=documents,
            embeddings=embeddings,
            metadatas=[doc["metadata"] for doc in documents]
        )

        logger.info(f"Added {len(added_docs)} documents to vector store")

        return added_docs

    def _generate_embedding(self, text: str) -> Optional[Any]:
        """Generate embedding for a text using sentence-transformers.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding or None on error
        """
        try:
            from sentence_transformers import SentenceTransformer

            local_model_path = "./models/local_all-MiniLM-L6-v2"
            model = SentenceTransformer(local_model_path)
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
        from .ollama_client import OllamaClient

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

        user_prompt = f"""Context retrieved from interview database:

{results.documents[0].get('text', '')[:1000] if results.documents else 'No context available'}

Question: {question}"""

        logger.info("Generating response with LLM")
        logger.info(f"User prompt for LLM:\n{user_prompt}")
        try:
            # Generate response using Ollama
            result = self.ollama_client.generate(prompt=user_prompt, model="qwen3.5:4b-mlx")
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
