"""Configuration settings for RAG pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """RAG Pipeline Configuration"""
    
    # Paths
    DATA_DIR: Path = Path("data")
    VECTOR_STORE_DIR: Path = Path("vector_store")
    
    # Embedding settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384  # all-MiniLM-L6-v2 dimension
    
    # Chunking settings
    CHUNK_SIZE: int = 1000  # characters
    CHUNK_OVERLAP: int = 200  # characters
    
    # Retrieval settings
    TOP_K: int = 5  # number of chunks to retrieve
    
    # LLM settings
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 500
    
    # Vector store backend
    VECTOR_STORE_TYPE: str = "faiss"  # "faiss" or "chromadb"
    
    # OpenAI API (optional - set via .env)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.DATA_DIR.mkdir(exist_ok=True)
        self.VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Global config instance
config = Config()
