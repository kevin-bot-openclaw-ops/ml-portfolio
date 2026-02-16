"""Text chunking module - splits documents into overlapping chunks."""

from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class TextChunker:
    """Splits text into overlapping chunks for embedding."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_documents(self, documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents with 'content' and 'metadata'
        
        Returns:
            List of chunks with 'content' and 'metadata' (including chunk_id)
        """
        all_chunks = []
        
        for doc in documents:
            text = doc["content"]
            metadata = doc["metadata"]
            
            chunks = self._split_text(text)
            
            for i, chunk_text in enumerate(chunks):
                chunk = {
                    "content": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                }
                all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _split_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Uses character-based splitting with overlap to maintain context.
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Find a good breaking point (end of sentence if possible)
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(end - 100, start)
                for sep in ['. ', '.\n', '! ', '?\n']:
                    last_sep = text.rfind(sep, search_start, end)
                    if last_sep != -1:
                        end = last_sep + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start forward, accounting for overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)
        
        return chunks
