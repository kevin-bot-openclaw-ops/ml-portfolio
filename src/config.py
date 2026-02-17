"""Configuration settings for RAG pipeline.

All values can be overridden via environment variables or a .env file.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; set env vars manually if missing


class Config:
    """Centralised RAG pipeline configuration."""

    # Paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
    VECTOR_STORE_DIR: Path = Path(os.getenv("VECTOR_STORE_DIR", "vector_store"))

    # Embedding — runs locally, no API key needed
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))    # characters
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Retrieval
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # LLM — provider selected via LLM_PROVIDER env var
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "500"))


config = Config()
