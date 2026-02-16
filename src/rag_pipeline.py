"""RAG Pipeline - End-to-end orchestration."""

from pathlib import Path
from typing import List, Optional
import logging

from .config import config
from .document_loader import DocumentLoader
from .chunker import TextChunker
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .llm import create_llm_provider, LLMProvider

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG pipeline for document Q&A."""
    
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        embedding_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        top_k: int = 5
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            data_dir: Directory containing documents (defaults to config.DATA_DIR)
            embedding_model: Embedding model name (defaults to config.EMBEDDING_MODEL)
            llm_provider: LLM provider type - "openai", "anthropic", or "ollama"
            llm_model: LLM model name
            top_k: Number of chunks to retrieve (defaults to config.TOP_K)
        """
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.top_k = top_k or config.TOP_K
        
        # Initialize components
        logger.info("Initializing RAG pipeline...")
        
        self.document_loader = DocumentLoader(self.data_dir)
        self.chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        # Initialize embedder
        embedding_model = embedding_model or config.EMBEDDING_MODEL
        self.embedder = Embedder(embedding_model)
        
        # Initialize vector store
        store_path = config.VECTOR_STORE_DIR / "faiss_index"
        self.vector_store = VectorStore(
            embedding_dim=self.embedder.embedding_dim,
            store_path=store_path
        )
        
        # Initialize retriever
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=self.top_k
        )
        
        # Initialize LLM provider (lazy - only when needed)
        self._llm_provider = None
        self._llm_provider_type = llm_provider
        self._llm_model = llm_model
        
        logger.info("RAG pipeline initialized")
    
    @property
    def llm(self) -> LLMProvider:
        """Lazy-load LLM provider."""
        if self._llm_provider is None:
            self._llm_provider = create_llm_provider(
                provider_type=self._llm_provider_type,
                model_name=self._llm_model,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.MAX_TOKENS
            )
        return self._llm_provider
    
    def ingest_documents(self, force: bool = False) -> int:
        """
        Load and index documents from data directory.
        
        Args:
            force: Force re-indexing even if index exists
        
        Returns:
            Number of chunks indexed
        """
        # Try to load existing index
        if not force and self.vector_store.load():
            logger.info("Loaded existing index")
            return len(self.vector_store.chunks)
        
        # Load documents
        logger.info(f"Loading documents from {self.data_dir}")
        documents = self.document_loader.load_documents()
        
        if not documents:
            logger.warning("No documents found")
            return 0
        
        # Chunk documents
        logger.info("Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        
        if not chunks:
            logger.warning("No chunks created")
            return 0
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        texts = [chunk["content"] for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        # Add to vector store
        logger.info("Adding to vector store...")
        self.vector_store.add_embeddings(embeddings, chunks)
        
        # Save index
        self.vector_store.save()
        
        logger.info(f"Ingestion complete: {len(chunks)} chunks indexed")
        return len(chunks)
    
    def query(self, question: str, include_sources: bool = True) -> str:
        """
        Query the RAG system.
        
        Args:
            question: User question
            include_sources: Whether to include source citations
        
        Returns:
            Generated answer (with optional sources)
        """
        logger.info(f"Processing query: {question}")
        
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(question)
        
        if not chunks:
            return "I don't have enough information to answer that question."
        
        # Build context from chunks
        context = self._build_context(chunks)
        
        # Build prompt
        prompt = self._build_prompt(question, context)
        
        # Generate answer
        logger.info("Generating answer with LLM...")
        answer = self.llm.generate(prompt)
        
        # Add sources if requested
        if include_sources:
            sources = self._format_sources(chunks)
            answer = f"{answer}\n\n{sources}"
        
        return answer
    
    def _build_context(self, chunks: List[dict]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk["content"]
            metadata = chunk.get("metadata", {})
            source = metadata.get("filename", "Unknown")
            
            context_parts.append(f"[Document {i}: {source}]\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt for LLM."""
        prompt = f"""You are a helpful assistant answering questions about financial documents.

Use the following context to answer the question. If you cannot answer based on the context, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        return prompt
    
    def _format_sources(self, chunks: List[dict]) -> str:
        """Format source citations."""
        sources = set()
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "Unknown")
            chunk_id = metadata.get("chunk_id", 0)
            sources.add(f"{filename} (chunk {chunk_id})")
        
        sources_list = sorted(sources)
        return "**Sources:**\n" + "\n".join(f"- {s}" for s in sources_list)
