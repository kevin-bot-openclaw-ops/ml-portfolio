"""Retrieval module - retrieves relevant chunks for a query."""

from typing import List, Dict
import logging

from .embedder import Embedder
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant document chunks using semantic search."""
    
    def __init__(self, embedder: Embedder, vector_store: VectorStore, top_k: int = 5):
        """
        Initialize retriever.
        
        Args:
            embedder: Embedder instance
            vector_store: VectorStore instance
            top_k: Number of chunks to retrieve
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Args:
            query: User query string
        
        Returns:
            List of chunk dictionaries, sorted by relevance
        """
        logger.info(f"Retrieving chunks for query: {query[:50]}...")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, self.top_k)
        
        # Extract chunks (drop distances for simplicity)
        chunks = [chunk for chunk, dist in results]
        
        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks
