"""Main entry point for RAG pipeline demo."""

import logging
from pathlib import Path
from dotenv import load_dotenv

from .rag_pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run RAG pipeline demo."""
    # Load environment variables
    load_dotenv()
    
    logger.info("=== RAG Pipeline Demo ===")
    
    # Initialize pipeline
    rag = RAGPipeline(
        data_dir=Path("data"),
        top_k=5
    )
    
    # Ingest documents
    logger.info("Step 1: Ingesting documents...")
    num_chunks = rag.ingest_documents()
    logger.info(f"Indexed {num_chunks} chunks")
    
    if num_chunks == 0:
        logger.error("No documents found. Add PDF or TXT files to the data/ directory.")
        return
    
    # Interactive query loop
    logger.info("\n=== Query Mode ===")
    logger.info("Enter your questions (or 'quit' to exit)\n")
    
    while True:
        try:
            question = input("\nQuestion: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                logger.info("Goodbye!")
                break
            
            # Process query
            answer = rag.query(question)
            print(f"\nAnswer:\n{answer}\n")
            
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")


if __name__ == "__main__":
    main()
