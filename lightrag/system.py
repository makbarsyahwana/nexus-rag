import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import structlog

from bedrock.bedrock_client import BedrockClient
from qdrant.qdrant_client import QdrantClient
from neo4j.neo4j_client import Neo4jClient
from config.settings import Settings

logger = structlog.get_logger(__name__)

@dataclass
class Document:
    """Document data structure."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None

@dataclass
class RetrievalResult:
    """Retrieval result data structure."""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str  # 'vector', 'graph', or 'hybrid'

class NexusRAGSystem:
    """
    NexusRAG system integrating Bedrock, Qdrant, and Neo4j.
    
    Supports multiple retrieval modes:
    - local: Vector-based retrieval using Qdrant
    - global: Graph-based retrieval using Neo4j
    - hybrid: Combination of vector and graph retrieval
    - mix: Advanced mixing of multiple retrieval strategies
    """
    
    def __init__(self, settings: Settings):
        """Initialize NexusRAG system with all components."""
        self.settings = settings
        
        # Initialize clients
        self.bedrock_client = BedrockClient(
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            session_token=settings.aws_session_token
        )
        
        self.qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout
        )
        
        self.neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
            max_connection_lifetime=settings.neo4j_max_connection_lifetime,
            max_connection_pool_size=settings.neo4j_max_connection_pool_size
        )
        
        logger.info("NexusRAG system initialized")
    
    async def initialize(self):
        """Initialize collections and indexes."""
        try:
            # Create Qdrant collection if it doesn't exist
            if not await self.qdrant_client.collection_exists(self.settings.qdrant_collection_name):
                await self.qdrant_client.create_collection(
                    collection_name=self.settings.qdrant_collection_name,
                    vector_size=self.settings.embedding_dimensions
                )
            
            # Create Neo4j indexes for better performance
            indexes = [
                {"label": "Document", "property": "doc_id"},
                {"label": "Chunk", "property": "chunk_id"},
                {"label": "Entity", "property": "name"},
                {"label": "Concept", "property": "name"}
            ]
            await self.neo4j_client.create_indexes(indexes)
            
            logger.info("NexusRAG system initialization completed")
            
        except Exception as e:
            logger.error("Failed to initialize NexusRAG system", error=str(e))
            raise
    
    async def close(self):
        """Close all database connections."""
        await self.neo4j_client.close()
        logger.info("NexusRAG system closed")
    
    async def ingest_documents(self, documents: List[Document]) -> bool:
        """
        Ingest documents into both vector and graph databases.
        
        Args:
            documents: List of documents to ingest
            
        Returns:
            True if successful
        """
        try:
            for doc in documents:
                await self._ingest_single_document(doc)
            
            logger.info("Documents ingested successfully", count=len(documents))
            return True
            
        except Exception as e:
            logger.error("Failed to ingest documents", error=str(e))
            return False
    
    async def _ingest_single_document(self, document: Document):
        """
        Process a single document through the complete ingestion pipeline:
        1. Document Loading
        2. Text Splitting
        3. Embedding Generation
        4. Vector Storage
        5. Graph Construction
        """
        logger.info("Starting document ingestion", doc_id=document.doc_id)
        
        # 1. Document Loading (already done, document is passed as parameter)
        logger.info("✅ Document loaded", doc_id=document.doc_id, 
                   content_preview=document.content[:100] + "...")
        
        # 2. Text Splitting
        chunks = self._chunk_document(document.content)
        logger.info("✅ Text split into chunks", doc_id=document.doc_id, 
                   num_chunks=len(chunks))
        
        # 3. Generate embeddings for each chunk
        chunk_embeddings = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document.doc_id}_chunk_{i}"
            try:
                # Generate embedding for the chunk
                embedding = await self.bedrock_client.generate_embeddings(
                    text=chunk,
                    model_id=self.settings.embedding_model_id
                )
                chunk_embeddings.append({
                    "id": chunk_id,
                    "vector": embedding,
                    "payload": {
                        "text": chunk,
                        "doc_id": document.doc_id,
                        "chunk_index": i,
                        **document.metadata
                    }
                })
                logger.debug("Generated embedding for chunk", 
                           chunk_id=chunk_id, 
                           embedding_dim=len(embedding))
            except Exception as e:
                logger.error("Failed to generate embedding for chunk", 
                            chunk_id=chunk_id, error=str(e))
                continue
        
        if not chunk_embeddings:
            logger.error("No embeddings generated for document", doc_id=document.doc_id)
            return
            
        logger.info("✅ Generated embeddings", 
                   doc_id=document.doc_id, 
                   num_embeddings=len(chunk_embeddings))
        
        # 4. Store vectors in Qdrant
        try:
            await self.qdrant_client.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=chunk_embeddings
            )
            logger.info("✅ Stored vectors in Qdrant", 
                       doc_id=document.doc_id,
                       collection=self.settings.qdrant_collection_name)
        except Exception as e:
            logger.error("Failed to store vectors in Qdrant", 
                        doc_id=document.doc_id, error=str(e))
            return
        
        # 5. Build and store knowledge graph in Neo4j
        try:
            await self._store_document_graph(document, chunks)
            logger.info("✅ Built knowledge graph in Neo4j", 
                       doc_id=document.doc_id)
        except Exception as e:
            logger.error("Failed to build knowledge graph", 
                        doc_id=document.doc_id, error=str(e))
            return
        
        logger.info("✅ Document ingestion completed successfully", 
                   doc_id=document.doc_id, 
                   num_chunks=len(chunks))
    
    def _chunk_document(self, content: str) -> List[str]:
        """Simple document chunking."""
        chunk_size = self.settings.chunk_size
        chunk_overlap = self.settings.chunk_overlap
        
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
            
            if start >= len(content):
                break
        
        return chunks
    
    async def _store_document_graph(self, document: Document, chunks: List[str]):
        """Store document and chunks in Neo4j graph."""
        # Create document node
        await self.neo4j_client.create_node(
            label="Document",
            properties={
                "doc_id": document.doc_id,
                "content": document.content[:1000],  # Truncate for storage
                **document.metadata
            }
        )
        
        # Create chunk nodes and relationships
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document.doc_id}_chunk_{i}"
            
            # Create chunk node
            await self.neo4j_client.create_node(
                label="Chunk",
                properties={
                    "chunk_id": chunk_id,
                    "content": chunk,
                    "chunk_index": i,
                    "doc_id": document.doc_id
                }
            )
            
            # Create relationship between document and chunk
            await self.neo4j_client.create_relationship(
                start_node_label="Document",
                start_node_properties={"doc_id": document.doc_id},
                end_node_label="Chunk",
                end_node_properties={"chunk_id": chunk_id},
                relationship_type="HAS_CHUNK",
                relationship_properties={"chunk_index": i}
            )
    
    async def query(
        self,
        query_text: str,
        mode: str = "hybrid",
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Query the RAG system with different modes.
        
        Args:
            query_text: Query text
            mode: Retrieval mode ('local', 'global', 'hybrid', 'mix')
            top_k: Number of results to return
            score_threshold: Minimum relevance score
            
        Returns:
            List of retrieval results
        """
        top_k = top_k or self.settings.retrieval_top_k
        score_threshold = score_threshold or self.settings.retrieval_score_threshold
        
        logger.info("Processing query", mode=mode, query_preview=query_text[:100])
        
        if mode == "local":
            return await self._vector_retrieval(query_text, top_k, score_threshold)
        elif mode == "global":
            return await self._graph_retrieval(query_text, top_k)
        elif mode == "hybrid":
            return await self._hybrid_retrieval(query_text, top_k, score_threshold)
        elif mode == "mix":
            return await self._mix_retrieval(query_text, top_k, score_threshold)
        else:
            raise ValueError(f"Unsupported query mode: {mode}")
    
    async def _vector_retrieval(
        self,
        query_text: str,
        top_k: int,
        score_threshold: float
    ) -> List[RetrievalResult]:
        """Perform vector-based retrieval using Qdrant."""
        # Generate query embedding
        query_embeddings = await self.bedrock_client.generate_embeddings(
            texts=[query_text],
            model_id=self.settings.embedding_model_id,
            dimensions=self.settings.embedding_dimensions
        )
        
        # Search in Qdrant
        search_results = await self.qdrant_client.search_vectors(
            collection_name=self.settings.qdrant_collection_name,
            query_vector=query_embeddings[0],
            limit=top_k,
            score_threshold=score_threshold
        )
        
        # Convert to RetrievalResult format
        results = []
        for result in search_results:
            results.append(RetrievalResult(
                content=result["payload"]["content"],
                score=result["score"],
                metadata=result["payload"],
                source="vector"
            ))
        
        logger.info("Vector retrieval completed", results_count=len(results))
        return results
    
    async def _graph_retrieval(self, query_text: str, top_k: int) -> List[RetrievalResult]:
        """Perform graph-based retrieval using Neo4j."""
        # Simple keyword-based graph search
        # In production, you'd want more sophisticated entity extraction and graph traversal
        
        # Search for chunks containing query keywords
        keywords = query_text.lower().split()
        
        results = []
        for keyword in keywords[:3]:  # Limit to first 3 keywords
            # Find chunks containing the keyword
            cypher_query = """
            MATCH (c:Chunk)
            WHERE toLower(c.content) CONTAINS $keyword
            RETURN c.content as content, c.chunk_id as chunk_id, c
            LIMIT $limit
            """
            
            graph_results = await self.neo4j_client.execute_query(
                cypher_query,
                {"keyword": keyword, "limit": top_k}
            )
            
            for result in graph_results:
                results.append(RetrievalResult(
                    content=result["content"],
                    score=0.8,  # Fixed score for graph results
                    metadata={"chunk_id": result["chunk_id"], "keyword": keyword},
                    source="graph"
                ))
        
        # Remove duplicates and limit results
        unique_results = {}
        for result in results:
            chunk_id = result.metadata.get("chunk_id")
            if chunk_id not in unique_results:
                unique_results[chunk_id] = result
        
        final_results = list(unique_results.values())[:top_k]
        
        logger.info("Graph retrieval completed", results_count=len(final_results))
        return final_results
    
    async def _hybrid_retrieval(
        self,
        query_text: str,
        top_k: int,
        score_threshold: float
    ) -> List[RetrievalResult]:
        """Combine vector and graph retrieval results."""
        # Get results from both methods
        vector_results = await self._vector_retrieval(query_text, top_k // 2, score_threshold)
        graph_results = await self._graph_retrieval(query_text, top_k // 2)
        
        # Combine and deduplicate results
        all_results = vector_results + graph_results
        
        # Remove duplicates based on content similarity
        unique_results = []
        seen_content = set()
        
        for result in all_results:
            content_hash = hash(result.content[:100])  # Use first 100 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # Sort by score and limit
        unique_results.sort(key=lambda x: x.score, reverse=True)
        final_results = unique_results[:top_k]
        
        logger.info("Hybrid retrieval completed", results_count=len(final_results))
        return final_results
    
    async def _mix_retrieval(
        self,
        query_text: str,
        top_k: int,
        score_threshold: float
    ) -> List[RetrievalResult]:
        """Advanced mixing of multiple retrieval strategies."""
        # For now, this is the same as hybrid retrieval
        # In production, you could implement more sophisticated mixing algorithms
        return await self._hybrid_retrieval(query_text, top_k, score_threshold)
    
    async def generate_response(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        model_id: Optional[str] = None
    ) -> str:
        """
        Generate a response using Bedrock LLM with retrieved context.
        
        Args:
            query: Original query
            retrieval_results: Retrieved context
            model_id: Optional model ID override
            
        Returns:
            Generated response
        """
        # Prepare context from retrieval results
        context_parts = []
        for i, result in enumerate(retrieval_results[:5]):  # Limit to top 5 results
            context_parts.append(f"Context {i+1} (score: {result.score:.3f}):\n{result.content}")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
        
        # Generate response using Bedrock
        response = await self.bedrock_client.generate_text(
            prompt=prompt,
            model_id=model_id or self.settings.llm_model_id,
            max_tokens=self.settings.max_tokens,
            temperature=self.settings.temperature
        )
        
        logger.info("Response generated", query_preview=query[:100], response_length=len(response))
        return response
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all system components."""
        bedrock_health = await self.bedrock_client.health_check()
        qdrant_health = await self.qdrant_client.health_check()
        neo4j_health = await self.neo4j_client.health_check()
        
        overall_status = "healthy" if all([
            bedrock_health.get("status") == "healthy",
            qdrant_health.get("status") == "healthy",
            neo4j_health.get("status") == "healthy"
        ]) else "unhealthy"
        
        return {
            "status": overall_status,
            "components": {
                "bedrock": bedrock_health,
                "qdrant": qdrant_health,
                "neo4j": neo4j_health
            }
        }
