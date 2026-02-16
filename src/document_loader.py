"""Document loading module - handles PDF and text file ingestion."""

from pathlib import Path
from typing import List, Dict
import logging

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads documents from various formats (PDF, TXT)."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize document loader.
        
        Args:
            data_dir: Directory containing documents to load
        """
        self.data_dir = Path(data_dir)
        
    def load_documents(self) -> List[Dict[str, str]]:
        """
        Load all supported documents from data directory.
        
        Returns:
            List of dictionaries with 'content' and 'metadata' keys
        """
        documents = []
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory {self.data_dir} does not exist")
            return documents
        
        # Load PDF files
        for pdf_path in self.data_dir.glob("**/*.pdf"):
            try:
                doc = self._load_pdf(pdf_path)
                documents.append(doc)
                logger.info(f"Loaded PDF: {pdf_path.name}")
            except Exception as e:
                logger.error(f"Error loading {pdf_path}: {e}")
        
        # Load text files
        for txt_path in self.data_dir.glob("**/*.txt"):
            try:
                doc = self._load_text(txt_path)
                documents.append(doc)
                logger.info(f"Loaded TXT: {txt_path.name}")
            except Exception as e:
                logger.error(f"Error loading {txt_path}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents total")
        return documents
    
    def _load_pdf(self, pdf_path: Path) -> Dict[str, str]:
        """Load a PDF file and extract text."""
        if PdfReader is None:
            raise ImportError("pypdf is required for PDF loading. Install with: pip install pypdf")
        
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        
        return {
            "content": text.strip(),
            "metadata": {
                "source": str(pdf_path),
                "filename": pdf_path.name,
                "pages": len(reader.pages),
                "type": "pdf"
            }
        }
    
    def _load_text(self, txt_path: Path) -> Dict[str, str]:
        """Load a text file."""
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return {
            "content": text.strip(),
            "metadata": {
                "source": str(txt_path),
                "filename": txt_path.name,
                "type": "text"
            }
        }
