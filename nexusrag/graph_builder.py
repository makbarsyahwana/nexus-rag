from typing import Dict, List, Any, Optional, Tuple
import json
from dataclasses import dataclass, field
import logging
from .models import Document

logger = logging.getLogger(__name__)

@dataclass
class EntityNode:
    """Represents a node in the knowledge graph."""
    label: str
    properties: Dict[str, Any]
    id_field: str = "id"
    source: str = "document"  # 'document' or 'chunk'
    chunk_id: Optional[str] = None

@dataclass
class Relationship:
    """Represents a relationship between two nodes."""
    source_label: str
    source_id: Any
    target_label: str
    target_id: Any
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_id_field: str = "id"
    target_id_field: str = "id"

class GraphBuilder:
    """Builds a knowledge graph from documents using LLM analysis."""
    
    def __init__(self, llm_client, settings):
        """
        Initialize the GraphBuilder.
        
        Args:
            llm_client: LLM client for analysis
            settings: Application settings
        """
        self.llm_client = llm_client
        self.settings = settings
    
    async def analyze_document(self, document: Document) -> Tuple[List[EntityNode], List[Relationship]]:
        """
        Analyze a complete document to extract entities and relationships.
        
        Args:
            document: The document to analyze
            
        Returns:
            Tuple of (nodes, relationships) extracted from the document
        """
        logger.info(f"Analyzing document-level content for doc_id: {document.doc_id}")
        return await self._analyze_content(
            content=document.content,
            source="document",
            doc_id=document.doc_id
        )
    
    async def analyze_chunk(self, chunk: 'DocumentChunk') -> Tuple[List[EntityNode], List[Relationship]]:
        """
        Analyze a document chunk to extract entities and relationships.
        
        Args:
            chunk: The document chunk to analyze
            
        Returns:
            Tuple of (nodes, relationships) extracted from the chunk
        """
        logger.debug(f"Analyzing chunk {chunk.chunk_id} for doc_id: {chunk.doc_id}")
        return await self._analyze_content(
            content=chunk.content,
            source="chunk",
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id
        )
    
    async def _analyze_content(
        self, 
        content: str, 
        source: str,
        doc_id: str,
        chunk_id: Optional[str] = None
    ) -> Tuple[List[EntityNode], List[Relationship]]:
        """
        Internal method to analyze content and extract graph components.
        
        Args:
            content: The text content to analyze
            source: Source type ('document' or 'chunk')
            doc_id: Document ID
            chunk_id: Optional chunk ID if this is a chunk
            
        Returns:
            Tuple of (nodes, relationships)
        """
        try:
            # Prepare the prompt for LLM analysis
            prompt = self._create_analysis_prompt(content)
            
            # Get LLM response
            response = await self.llm_client.generate(
                model_id=self.settings.llm_model_id,
                prompt=prompt,
                max_tokens=2000,
                temperature=0.3
            )
            
            # Parse the LLM response
            analysis = self._parse_llm_response(response)
            
            # Convert to entity nodes and relationships
            nodes = self._create_entity_nodes(analysis.get("entities", []), source, doc_id, chunk_id)
            relationships = self._create_relationships(analysis.get("relationships", []))
            
            return nodes, relationships
            
        except Exception as e:
            logger.error(f"Failed to analyze content: {e}")
            return [], []
    
    def _create_entity_nodes(
        self, 
        entities: List[dict], 
        source: str,
        doc_id: str,
        chunk_id: Optional[str] = None
    ) -> List[EntityNode]:
        """Convert raw entity data to EntityNode objects."""
        nodes = []
        for entity in entities:
            try:
                node = EntityNode(
                    label=entity["type"],
                    properties={
                        **entity["properties"],
                        "doc_id": doc_id,
                        "source": source
                    },
                    id_field=entity.get("id_field", "id"),
                    source=source,
                    chunk_id=chunk_id
                )
                nodes.append(node)
            except KeyError as e:
                logger.warning(f"Invalid entity format: {e}")
        return nodes
    
    def _create_relationships(self, relationships: List[dict]) -> List[Relationship]:
        """Convert raw relationship data to Relationship objects."""
        rels = []
        for rel in relationships:
            try:
                relationship = Relationship(
                    source_label=rel["source_type"],
                    source_id=rel["source_id"],
                    source_id_field=rel.get("source_id_field", "id"),
                    target_label=rel["target_type"],
                    target_id=rel["target_id"],
                    target_id_field=rel.get("target_id_field", "id"),
                    relationship_type=rel["type"],
                    properties=rel.get("properties", {})
                )
                rels.append(relationship)
            except KeyError as e:
                logger.warning(f"Invalid relationship format: {e}")
        return rels
    
    def _create_analysis_prompt(self, content: str) -> str:
        """Create a prompt for LLM to analyze document structure."""
        return f"""
        Analyze the following content and identify key entities and their relationships.
        Focus on extracting structured information that can be represented as a knowledge graph.
        
        Content:
        {content[:5000]}... [truncated if too long]
        
        Instructions:
        1. Identify distinct entity types (e.g., Person, Organization, Location)
        2. For each entity, extract relevant properties
        3. Identify relationships between entities
        4. Format the response as a JSON object with 'entities' and 'relationships' arrays
        
        Example Output:
        {{
            "entities": [
                {{
                    "type": "Person",
                    "properties": {{"name": "John Doe", "title": "CEO"}},
                    "id_field": "name"
                }}
            ],
            "relationships": [
                {{
                    "source_type": "Person",
                    "source_id": "John Doe",
                    "source_id_field": "name",
                    "target_type": "Organization",
                    "target_id": "Acme Corp",
                    "target_id_field": "name",
                    "type": "WORKS_AT",
                    "properties": {{"since": "2020"}}
                }}
            ]
        }}
        """
    
    def _parse_llm_response(self, response: str) -> dict:
        """Parse the LLM response into structured data."""
        try:
            # Clean and parse the JSON response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start < 0 or json_end <= 0:
                raise ValueError("No JSON object found in response")
                
            json_str = response[json_start:json_end]
            result = json.loads(json_str)
            
            # Validate the structure
            if not isinstance(result, dict):
                raise ValueError("Expected a JSON object")
                
            if "entities" not in result or not isinstance(result["entities"], list):
                result["entities"] = []
                
            if "relationships" not in result or not isinstance(result["relationships"], list):
                result["relationships"] = []
                
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {"entities": [], "relationships": []}
