"""Tests for embedding generation."""

import pytest
import numpy as np
from src.embedder import Embedder


@pytest.fixture
def embedder():
    """Create embedder instance for testing."""
    # Use small model for faster tests
    return Embedder("sentence-transformers/all-MiniLM-L6-v2")


def test_embedder_init(embedder):
    """Test embedder initialization."""
    assert embedder.model is not None
    assert embedder.embedding_dim == 384  # all-MiniLM-L6-v2 dimension


def test_embed_single_text(embedder):
    """Test embedding a single text."""
    text = "This is a test sentence."
    embedding = embedder.embed_query(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert not np.isnan(embedding).any()


def test_embed_multiple_texts(embedder):
    """Test embedding multiple texts."""
    texts = [
        "First test sentence.",
        "Second test sentence.",
        "Third test sentence."
    ]
    embeddings = embedder.embed_texts(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)
    assert not np.isnan(embeddings).any()


def test_embed_empty_list(embedder):
    """Test embedding empty list."""
    embeddings = embedder.embed_texts([])
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings) == 0


def test_embedding_similarity(embedder):
    """Test that similar texts have similar embeddings."""
    text1 = "Banking regulations and capital requirements"
    text2 = "Capital requirements for banks and regulations"
    text3 = "The weather is sunny today"
    
    emb1 = embedder.embed_query(text1)
    emb2 = embedder.embed_query(text2)
    emb3 = embedder.embed_query(text3)
    
    # Cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    sim_1_2 = cosine_similarity(emb1, emb2)
    sim_1_3 = cosine_similarity(emb1, emb3)
    
    # Similar banking texts should be more similar than unrelated text
    assert sim_1_2 > sim_1_3
