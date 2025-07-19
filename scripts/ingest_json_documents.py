#!/usr/bin/env python3
"""
Script to ingest JSON documents into NexusRAG system.
Supports various JSON formats including single objects, arrays, and JSONL files.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.system import NexusRAGSystem, Document
from config.settings import Settings

class JSONDocumentProcessor:
    """Process various JSON document formats for NexusRAG ingestion."""
    
    def __init__(self, content_fields: List[str] = None, metadata_fields: List[str] = None):
        """
        Initialize processor with field mappings.
        
        Args:
            content_fields: Fields to extract as content (e.g., ['text', 'body', 'content'])
            metadata_fields: Fields to include as metadata (e.g., ['title', 'author', 'timestamp'])
        """
        self.content_fields = content_fields or ['content', 'text', 'body', 'description', 'article', 'message']
        self.metadata_fields = metadata_fields or [
            'title', 'author', 'category', 'tags', 'timestamp', 'source', 'date', 
            'created_at', 'updated_at', 'type', 'id', 'url', 'filename'
        ]
    
    def process_single_json(self, json_data: Dict[str, Any], doc_id: str = None) -> Document:
        """Process a single JSON object into a Document."""
        
        # Extract content
        content_parts = []
        used_content_fields = []
        
        for field in self.content_fields:
            if field in json_data and json_data[field]:
                content_parts.append(str(json_data[field]))
                used_content_fields.append(field)
        
        # If no content fields found, try to construct content from other string fields
        if not content_parts:
            for key, value in json_data.items():
                if (isinstance(value, str) and len(value) > 50 and 
                    key not in self.metadata_fields):
                    content_parts.append(f"{key}: {value}")
                    used_content_fields.append(key)
        
        # If still no content, use JSON string representation
        if not content_parts:
            content_parts.append(json.dumps(json_data, indent=2))
        
        content = "\n\n".join(content_parts)
        
        # Extract metadata
        metadata = {}
        
        # Add specified metadata fields
        for field in self.metadata_fields:
            if field in json_data:
                metadata[field] = json_data[field]
        
        # Add any additional fields as metadata (except content fields)
        for key, value in json_data.items():
            if (key not in used_content_fields and 
                key not in metadata and 
                isinstance(value, (str, int, float, bool, list, dict))):
                metadata[key] = value
        
        # Generate doc_id if not provided
        if not doc_id:
            doc_id = (metadata.get('id') or 
                     metadata.get('doc_id') or 
                     metadata.get('_id') or 
                     f"doc_{hash(content[:100])}")
        
        return Document(
            content=content,
            metadata=metadata,
            doc_id=str(doc_id)
        )
    
    def process_json_file(self, file_path: Path) -> List[Document]:
        """Process a JSON file that can contain single object or array of objects."""
        
        print(f"📄 Processing JSON file: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        if isinstance(data, list):
            # Array of JSON objects
            print(f"   Found array with {len(data)} objects")
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    doc_id = f"{file_path.stem}_{i+1}"
                    doc = self.process_single_json(item, doc_id)
                    documents.append(doc)
                else:
                    print(f"   ⚠️ Skipping non-object item at index {i}")
                    
        elif isinstance(data, dict):
            # Single JSON object
            print("   Found single JSON object")
            doc_id = file_path.stem
            doc = self.process_single_json(data, doc_id)
            documents.append(doc)
        else:
            raise ValueError(f"Unsupported JSON structure in {file_path}. Expected object or array.")
        
        print(f"   ✅ Processed {len(documents)} documents")
        return documents
    
    def process_jsonl_file(self, file_path: Path) -> List[Document]:
        """Process a JSONL (JSON Lines) file."""
        
        print(f"📄 Processing JSONL file: {file_path}")
        
        documents = []
        line_count = 0
        error_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict):
                            doc_id = f"{file_path.stem}_{i}"
                            doc = self.process_single_json(data, doc_id)
                            documents.append(doc)
                            line_count += 1
                        else:
                            print(f"   ⚠️ Skipping non-object at line {i}")
                            error_count += 1
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️ JSON decode error at line {i}: {e}")
                        error_count += 1
        
        print(f"   ✅ Processed {line_count} documents ({error_count} errors)")
        return documents
    
    def process_nested_json(self, data: Dict[str, Any], parent_key: str = "") -> List[Document]:
        """Process nested JSON structures by flattening them into documents."""
        
        documents = []
        
        def extract_documents(obj: Any, path: str = ""):
            if isinstance(obj, dict):
                # Check if this dict looks like a document
                has_content = any(field in obj for field in self.content_fields)
                
                if has_content or len(obj) > 3:  # Looks like a document
                    doc_id = f"{parent_key}_{path}".strip("_") or f"doc_{len(documents)}"
                    doc = self.process_single_json(obj, doc_id)
                    documents.append(doc)
                else:
                    # Recurse into nested structure
                    for key, value in obj.items():
                        new_path = f"{path}_{key}" if path else key
                        extract_documents(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    new_path = f"{path}_{i}" if path else str(i)
                    extract_documents(item, new_path)
        
        extract_documents(data)
        return documents

# Sample JSON documents for testing different formats
SAMPLE_JSON_DOCUMENTS = [
    {
        "title": "Introduction to Vector Databases",
        "content": """Vector databases are specialized databases designed to store and query high-dimensional vectors efficiently. They are essential for applications involving machine learning embeddings, such as similarity search, recommendation systems, and retrieval-augmented generation (RAG) systems.

Key benefits include:
- Fast similarity search using advanced indexing algorithms
- Scalability for billions of vectors
- Integration with ML pipelines
- Support for metadata filtering""",
        "author": "AI Research Team",
        "category": "database",
        "tags": ["vector-database", "machine-learning", "embeddings"],
        "timestamp": "2024-01-15T10:30:00Z",
        "source": "research_paper"
    },
    {
        "id": "doc_002",
        "text": """Graph databases represent data as nodes and edges, making them ideal for storing and querying complex relationships. In RAG systems, graph databases can store knowledge graphs that capture semantic relationships between entities, enabling more sophisticated retrieval strategies.

Applications include:
- Social network analysis
- Recommendation engines  
- Fraud detection
- Knowledge representation""",
        "metadata": {
            "topic": "graph-databases",
            "difficulty": "intermediate",
            "estimated_reading_time": "3 minutes"
        },
        "created_at": "2024-01-16T14:20:00Z",
        "url": "https://example.com/graph-db-guide"
    },
    {
        "document_id": "bedrock_guide",
        "body": """AWS Bedrock provides a unified API to access foundation models from various providers. It supports text generation, embeddings, and image generation through models like Claude 3, Titan, and Stable Diffusion. 

Key features:
- Managed infrastructure
- Multiple model providers
- Enterprise security
- Pay-per-use pricing
- Easy integration with AWS services""",
        "title": "AWS Bedrock Overview",
        "category": "cloud-services",
        "provider": "AWS",
        "last_updated": "2024-01-17T09:15:00Z",
        "tags": ["aws", "bedrock", "llm", "api"]
    },
    {
        "article": """NexusRAG represents the next generation of retrieval-augmented generation systems. By combining vector databases for semantic search with graph databases for relationship modeling, NexusRAG provides more accurate and contextually relevant responses.

Architecture components:
- Vector storage in Qdrant
- Knowledge graphs in Neo4j
- LLM integration via AWS Bedrock
- Hybrid retrieval strategies""",
        "headline": "NexusRAG: Advanced RAG Architecture",
        "author": "Technical Team",
        "publication_date": "2024-01-18",
        "type": "technical_article",
        "complexity": "advanced"
    }
]

async def ingest_json_documents(
    json_input: Union[List[Dict], str, Path, Dict] = None,
    content_fields: List[str] = None,
    metadata_fields: List[str] = None,
    nested: bool = False
):
    """
    Main function to ingest JSON documents.
    
    Args:
        json_input: Can be:
            - List of dict objects
            - Path to JSON file
            - Path to JSONL file
            - Single dict object
            - None (uses sample data)
        content_fields: Fields to use as content
        metadata_fields: Fields to include as metadata
        nested: Whether to process nested JSON structures
    """
    
    print("🚀 Starting JSON document ingestion...")
    
    # Initialize processor and system
    processor = JSONDocumentProcessor(content_fields, metadata_fields)
    settings = Settings()
    system = NexusRAGSystem(settings)
    
    try:
        # Initialize the system
        print("📋 Initializing NexusRAG system...")
        await system.initialize()
        
        # Process JSON data
        documents = []
        
        if json_input is None:
            # Use sample data
            print("📄 Using sample JSON documents...")
            for i, json_obj in enumerate(SAMPLE_JSON_DOCUMENTS):
                doc = processor.process_single_json(json_obj, f"sample_json_{i+1}")
                documents.append(doc)
                
        elif isinstance(json_input, (str, Path)):
            # File path
            file_path = Path(json_input)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.suffix.lower() == '.jsonl':
                documents = processor.process_jsonl_file(file_path)
            else:
                documents = processor.process_json_file(file_path)
                
        elif isinstance(json_input, dict):
            # Single JSON object
            print("📄 Processing single JSON object...")
            if nested:
                documents = processor.process_nested_json(json_input, "input")
            else:
                doc = processor.process_single_json(json_input, "input_doc")
                documents.append(doc)
                
        elif isinstance(json_input, list):
            # List of JSON objects
            print(f"📄 Processing {len(json_input)} JSON objects...")
            for i, json_obj in enumerate(json_input):
                if isinstance(json_obj, dict):
                    doc = processor.process_single_json(json_obj, f"list_doc_{i+1}")
                    documents.append(doc)
                else:
                    print(f"   ⚠️ Skipping non-object item at index {i}")
        else:
            raise ValueError("Invalid json_input type. Expected dict, list, file path, or None.")
        
        if not documents:
            print("❌ No documents were processed")
            return False
        
        # Display processed documents info
        print(f"\n📋 Processed {len(documents)} documents:")
        for i, doc in enumerate(documents[:3]):  # Show first 3
            print(f"  {i+1}. ID: {doc.doc_id}")
            print(f"     Content: {doc.content[:100]}...")
            print(f"     Metadata: {list(doc.metadata.keys())}")
            print()
        
        if len(documents) > 3:
            print(f"     ... and {len(documents) - 3} more documents")
        
        # Ingest documents
        print(f"📚 Ingesting {len(documents)} documents...")
        success = await system.ingest_documents(documents)
        
        if success:
            print("✅ JSON documents ingested successfully!")
            
            # Test query
            test_query = "What are the benefits of vector databases?"
            print(f"\n🔍 Testing query: {test_query}")
            
            results = await system.query(test_query, mode="hybrid", top_k=3)
            if results:
                print(f"   Found {len(results)} relevant results")
                for i, result in enumerate(results, 1):
                    print(f"   {i}. Score: {result.score:.3f} | Source: {result.source}")
                    print(f"      Content: {result.content[:100]}...")
                
                # Generate response
                response = await system.generate_response(test_query, results)
                print(f"\n💬 Generated response:")
                print(f"   {response[:300]}...")
            else:
                print("   No results found")
                
            return True
        else:
            print("❌ Failed to ingest JSON documents")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await system.close()
        print("🔒 System closed")

# Example usage functions
async def ingest_from_file(file_path: str, **kwargs):
    """Ingest from a JSON or JSONL file."""
    return await ingest_json_documents(json_input=file_path, **kwargs)

async def ingest_from_directory(directory_path: str, pattern: str = "*.json", **kwargs):
    """Ingest all JSON files from a directory."""
    dir_path = Path(directory_path)
    if not dir_path.exists():
        print(f"❌ Directory not found: {directory_path}")
        return False
    
    files = list(dir_path.glob(pattern)) + list(dir_path.glob("*.jsonl"))
    if not files:
        print(f"❌ No JSON files found in {directory_path}")
        return False
    
    print(f"📁 Found {len(files)} files to process")
    
    all_success = True
    for file_path in files:
        print(f"\n📄 Processing: {file_path.name}")
        success = await ingest_from_file(str(file_path), **kwargs)
        all_success = all_success and success
    
    return all_success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest JSON documents into NexusRAG")
    parser.add_argument("--file", help="Path to JSON/JSONL file")
    parser.add_argument("--directory", help="Path to directory containing JSON files")
    parser.add_argument("--pattern", default="*.json", help="File pattern for directory search")
    parser.add_argument("--content-fields", nargs="+", 
                       help="Fields to use as content (e.g., --content-fields text body content)")
    parser.add_argument("--metadata-fields", nargs="+", 
                       help="Fields to include as metadata (e.g., --metadata-fields title author date)")
    parser.add_argument("--nested", action="store_true", 
                       help="Process nested JSON structures")
    parser.add_argument("--sample", action="store_true", 
                       help="Use sample JSON documents")
    
    args = parser.parse_args()
    
    # Determine input source
    json_input = None
    if args.sample:
        json_input = None  # Use sample data
    elif args.file:
        json_input = args.file
    elif args.directory:
        # Handle directory processing
        asyncio.run(ingest_from_directory(
            args.directory,
            pattern=args.pattern,
            content_fields=args.content_fields,
            metadata_fields=args.metadata_fields,
            nested=args.nested
        ))
        exit()
    else:
        print("Please specify --file, --directory, or --sample")
        parser.print_help()
        exit(1)
    
    # Run single file or sample processing
    asyncio.run(ingest_json_documents(
        json_input=json_input,
        content_fields=args.content_fields,
        metadata_fields=args.metadata_fields,
        nested=args.nested
    ))
