"""Vector store module - FAISS-based similarity search."""

from typing import List, Dict, Tuple
import logging
import pickle
from pathlib import Path
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

logger = logging.getLogger(__name__)


class VectorStore:
    """FAISS-based vector store for semantic search."""
    
    def __init__(self, embedding_dim: int, store_path: Path = None):
        """
        Initialize vector store.
        
        Args:
            embedding_dim: Dimension of embeddings
            store_path: Path to save/load the index
        """
        if faiss is None:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        
        self.embedding_dim = embedding_dim
        self.store_path = store_path
        self.index = None
        self.chunks = []  # Store original chunks for retrieval
        
        # Create a flat L2 index (exact search)
        self.index = faiss.IndexFlatL2(embedding_dim)
        logger.info(f"Initialized FAISS index with dimension {embedding_dim}")
    
    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict[str, str]]):
        """
        Add embeddings and associated chunks to the store.
        
        Args:
            embeddings: numpy array of shape (n_chunks, embedding_dim)
            chunks: List of chunk dictionaries
        """
        if len(embeddings) != len(chunks):
            raise ValueError(f"Embeddings ({len(embeddings)}) and chunks ({len(chunks)}) must have same length")
        
        # Normalize embeddings for better cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        self.chunks.extend(chunks)
        
        logger.info(f"Added {len(embeddings)} embeddings. Total: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, str], float]]:
        """
        Search for top-k most similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of (chunk, distance) tuples, sorted by similarity
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Reshape and normalize query
        query_vec = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_vec)
        
        # Search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_vec, k)
        
        # Return chunks with distances
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))
        
        logger.info(f"Retrieved {len(results)} results")
        return results
    
    def save(self):
        """Save index and chunks to disk."""
        if self.store_path is None:
            logger.warning("No store_path specified, cannot save")
            return
        
        store_dir = self.store_path.parent
        store_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = self.store_path.with_suffix('.index')
        faiss.write_index(self.index, str(index_path))
        
        # Save chunks
        chunks_path = self.store_path.with_suffix('.pkl')
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info(f"Saved index to {index_path} and chunks to {chunks_path}")
    
    def load(self):
        """Load index and chunks from disk."""
        if self.store_path is None:
            logger.warning("No store_path specified, cannot load")
            return False
        
        index_path = self.store_path.with_suffix('.index')
        chunks_path = self.store_path.with_suffix('.pkl')
        
        if not index_path.exists() or not chunks_path.exists():
            logger.info("No existing index found")
            return False
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load chunks
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)
        
        logger.info(f"Loaded index with {self.index.ntotal} vectors and {len(self.chunks)} chunks")
        return True
