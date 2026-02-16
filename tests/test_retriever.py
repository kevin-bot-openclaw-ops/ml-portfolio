"""Tests for retrieval functionality."""

import pytest
import numpy as np
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.retriever import Retriever


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "content": "Basel III capital requirements include Common Equity Tier 1 minimum of 4.5%",
            "metadata": {"filename": "basel.txt", "chunk_id": 0}
        },
        {
            "content": "The Liquidity Coverage Ratio ensures banks hold sufficient liquid assets",
            "metadata": {"filename": "basel.txt", "chunk_id": 1}
        },
        {
            "content": "Leverage ratio is a non-risk-based measure set at 3% minimum",
            "metadata": {"filename": "basel.txt", "chunk_id": 2}
        }
    ]


@pytest.fixture
def populated_retriever(sample_chunks):
    """Create retriever with populated vector store."""
    # Initialize embedder
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize vector store
    vector_store = VectorStore(embedding_dim=embedder.embedding_dim)
    
    # Generate and add embeddings
    texts = [chunk["content"] for chunk in sample_chunks]
    embeddings = embedder.embed_texts(texts)
    vector_store.add_embeddings(embeddings, sample_chunks)
    
    # Create retriever
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=2
    )
    
    return retriever


def test_retriever_init():
    """Test retriever initialization."""
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    vector_store = VectorStore(embedding_dim=384)
    retriever = Retriever(embedder, vector_store, top_k=3)
    
    assert retriever.embedder == embedder
    assert retriever.vector_store == vector_store
    assert retriever.top_k == 3


def test_retriever_retrieve(populated_retriever):
    """Test retrieval functionality."""
    query = "What is the capital requirement?"
    results = populated_retriever.retrieve(query)
    
    # Should return top_k chunks
    assert len(results) <= populated_retriever.top_k
    assert len(results) > 0
    
    # Results should be chunk dictionaries
    for chunk in results:
        assert "content" in chunk
        assert "metadata" in chunk


def test_retriever_relevance(populated_retriever):
    """Test that retrieval returns relevant chunks."""
    # Query about capital
    query = "capital requirements"
    results = populated_retriever.retrieve(query)
    
    # Top result should mention capital
    top_chunk = results[0]
    assert "capital" in top_chunk["content"].lower() or "Capital" in top_chunk["content"]


def test_retriever_empty_store():
    """Test retrieval from empty vector store."""
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    vector_store = VectorStore(embedding_dim=384)
    retriever = Retriever(embedder, vector_store, top_k=3)
    
    results = retriever.retrieve("test query")
    assert len(results) == 0
