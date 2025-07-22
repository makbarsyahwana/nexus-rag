import json
import csv
from typing import List, Dict, Any, Union, Iterator, Optional
from pathlib import Path
from dataclasses import dataclass
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of document with metadata."""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    doc_id: str
    chunk_index: int

class DocumentLoader:
    """Handles loading and chunking of JSON and CSV documents."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def load_document(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load a document from file path, detecting the format.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document objects with content and metadata
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            return await self._load_json(file_path)
        elif file_path.suffix.lower() == '.csv':
            return await self._load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    async def _load_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and parse a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                
                # Handle both single object and array of objects
                if isinstance(data, list):
                    return [{"content": str(item), "metadata": {"source": str(file_path)}} 
                           for item in data]
                else:
                    return [{"content": str(data), "metadata": {"source": str(file_path)}}]
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON file {file_path}: {e}")
                raise
    
    async def _load_csv(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and parse a CSV file."""
        documents = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                # Try to detect the dialect
                dialect = csv.Sniffer().sniff(f.read(1024))
                f.seek(0)
                
                reader = csv.DictReader(f, dialect=dialect)
                for i, row in enumerate(reader):
                    # Convert row to string representation for now
                    content = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
                    documents.append({
                        "content": content,
                        "metadata": {
                            "source": str(file_path),
                            "row": i + 1,
                            **{k: v for k, v in row.items() if k and v}
                        }
                    })
                    
            except Exception as e:
                logger.error(f"Failed to parse CSV file {file_path}: {e}")
                raise
        
        return documents
    
    def chunk_document(self, content: str, metadata: Dict[str, Any], doc_id: str) -> List[DocumentChunk]:
        """
        Split a document into chunks with metadata.
        
        Args:
            content: The document content to chunk
            metadata: Metadata to include with each chunk
            doc_id: Unique identifier for the document
            
        Returns:
            List of document chunks
        """
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(content):
            # Calculate end position with overlap
            end = min(start + self.chunk_size, len(content))
            
            # Get the chunk content
            chunk_content = content[start:end]
            
            # Create chunk with metadata
            chunk_id = f"{doc_id}_chunk_{chunk_index}"
            chunk_metadata = {
                **metadata,
                "chunk_index": chunk_index,
                "chunk_start": start,
                "chunk_end": end,
                "chunk_id": chunk_id
            }
            
            chunks.append(DocumentChunk(
                content=chunk_content,
                metadata=chunk_metadata,
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk_index=chunk_index
            ))
            
            # Move to next chunk with overlap
            if end == len(content):
                break
                
            start = end - self.chunk_overlap
            chunk_index += 1
        
        return chunks
