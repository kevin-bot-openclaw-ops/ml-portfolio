# Architecture Deep Dive

## Component Responsibilities

### DocumentLoader
- Reads PDF (via pypdf) and TXT files from a directory
- Returns list of `{"content": str, "metadata": dict}` dicts
- Metadata includes: source path, filename, file type, page count (PDF)

### TextChunker
- Splits documents into overlapping character-based chunks
- Prefers sentence boundaries (looks for `. `, `.\n`, `! `, `?\n`) when splitting
- Configurable `chunk_size` and `chunk_overlap` via `config.py`
- Adds `chunk_id` and `total_chunks` to metadata for traceability

### Embedder
- Wraps `sentence-transformers` — all computation runs locally
- Default model: `all-MiniLM-L6-v2` (384-dim, fast, good quality)
- `embed_texts()` for batch ingestion; `embed_query()` for single query
- Returns numpy arrays for direct FAISS consumption

### VectorStore
- Wraps FAISS `IndexFlatL2` (exact nearest-neighbour search)
- L2-normalises vectors before indexing → equivalent to cosine similarity
- Stores original chunk dicts in parallel list (index position = chunk position)
- `save()` / `load()` persist both FAISS `.index` and chunks `.pkl` to disk
- Re-uses existing index across sessions (skip re-embedding if unchanged)

### Retriever
- Embeds query → calls `vector_store.search()` → returns top-K chunks
- Stateless: just composes Embedder + VectorStore

### LLM Providers
- `OpenAIProvider`: calls `/chat/completions` via official SDK
- `AnthropicProvider`: calls `/messages` via official SDK
- `OllamaProvider`: calls local HTTP API (`/api/generate`)
- `create_llm_provider()` factory reads `LLM_PROVIDER` and `LLM_MODEL` from env
- All providers share the same `generate(prompt: str) -> str` interface

### RAGPipeline
- Orchestrates the full pipeline
- `ingest_documents()`: load → chunk → embed → index → persist
- `query()`: retrieve → build context → build prompt → generate → append sources
- LLM is lazy-loaded (only instantiated on first query call)

## Design Decisions

### Why FAISS over ChromaDB?
FAISS is simpler, faster, and has no server dependency. For a portfolio demo with <10K chunks, flat L2 search is more than adequate. ChromaDB would be a natural next step for larger corpora.

### Why sentence-transformers over OpenAI embeddings?
Zero API cost, zero latency on ingestion, fully offline. `all-MiniLM-L6-v2` is well-benchmarked and fast enough for demo purposes. Easy to swap to `text-embedding-3-small` if needed.

### Why no LangChain?
LangChain adds abstraction and debugging complexity for a demo. Building components from scratch is more transparent for interviews — you can explain every line.

### Chunking strategy
Character-based with sentence boundary preference keeps chunks semantically coherent without requiring a sentence tokeniser dependency. 1000 chars / 200 overlap is a reasonable default for financial prose.
