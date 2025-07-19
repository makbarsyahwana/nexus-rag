from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import structlog
import asyncio
from contextlib import asynccontextmanager

from nexusrag.system import NexusRAGSystem, Document, RetrievalResult
from config.settings import Settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(30),
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global variables
nexusrag_system: Optional[NexusRAGSystem] = None
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global nexusrag_system
    
    # Startup
    logger.info("Starting NexusRAG API server")
    nexusrag_system = NexusRAGSystem(settings)
    await nexusrag_system.initialize()
    logger.info("NexusRAG system initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down NexusRAG API server")
    if nexusrag_system:
        await nexusrag_system.close()
    logger.info("NexusRAG system closed")

# Create FastAPI app
app = FastAPI(
    title="NexusRAG API",
    description="Advanced RAG system with Bedrock, Qdrant, and Neo4j",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DocumentInput(BaseModel):
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    doc_id: Optional[str] = Field(None, description="Document ID")

class QueryInput(BaseModel):
    query: str = Field(..., description="Query text")
    mode: str = Field(default="hybrid", description="Retrieval mode: local, global, hybrid, mix")
    top_k: Optional[int] = Field(None, description="Number of results to return")
    score_threshold: Optional[float] = Field(None, description="Minimum relevance score")
    generate_response: bool = Field(default=True, description="Whether to generate a response")

class QueryResponse(BaseModel):
    query: str
    mode: str
    retrieval_results: List[Dict[str, Any]]
    response: Optional[str] = None
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, Any]

# Dependency to get NexusRAG system
async def get_nexusrag_system() -> NexusRAGSystem:
    if nexusrag_system is None:
        raise HTTPException(status_code=503, detail="NexusRAG system not initialized")
    return nexusrag_system

# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "NexusRAG API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(system: NexusRAGSystem = Depends(get_nexusrag_system)):
    """Health check endpoint."""
    try:
        health_data = await system.health_check()
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=503, detail="Health check failed")

@app.post("/ingest")
async def ingest_documents(
    documents: List[DocumentInput],
    background_tasks: BackgroundTasks,
    system: NexusRAGSystem = Depends(get_nexusrag_system)
):
    """Ingest documents into the system."""
    try:
        # Convert input to Document objects
        doc_objects = []
        for i, doc_input in enumerate(documents):
            doc_id = doc_input.doc_id or f"doc_{i}_{hash(doc_input.content[:100])}"
            doc_objects.append(Document(
                content=doc_input.content,
                metadata=doc_input.metadata,
                doc_id=doc_id
            ))
        
        # Ingest documents in background
        background_tasks.add_task(system.ingest_documents, doc_objects)
        
        logger.info("Document ingestion started", count=len(documents))
        
        return {
            "message": f"Started ingesting {len(documents)} documents",
            "document_ids": [doc.doc_id for doc in doc_objects]
        }
        
    except Exception as e:
        logger.error("Document ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_system(
    query_input: QueryInput,
    system: NexusRAGSystem = Depends(get_nexusrag_system)
):
    """Query the RAG system."""
    try:
        # Perform retrieval
        retrieval_results = await system.query(
            query_text=query_input.query,
            mode=query_input.mode,
            top_k=query_input.top_k,
            score_threshold=query_input.score_threshold
        )
        
        # Convert retrieval results to dict format
        retrieval_dicts = []
        for result in retrieval_results:
            retrieval_dicts.append({
                "content": result.content,
                "score": result.score,
                "metadata": result.metadata,
                "source": result.source
            })
        
        # Generate response if requested
        response_text = None
        if query_input.generate_response and retrieval_results:
            response_text = await system.generate_response(
                query=query_input.query,
                retrieval_results=retrieval_results
            )
        
        logger.info("Query processed", mode=query_input.mode, results_count=len(retrieval_results))
        
        return QueryResponse(
            query=query_input.query,
            mode=query_input.mode,
            retrieval_results=retrieval_dicts,
            response=response_text,
            metadata={
                "results_count": len(retrieval_results),
                "top_k": query_input.top_k or settings.retrieval_top_k,
                "score_threshold": query_input.score_threshold or settings.retrieval_score_threshold
            }
        )
        
    except Exception as e:
        logger.error("Query processing failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/collections/info")
async def get_collection_info(system: NexusRAGSystem = Depends(get_nexusrag_system)):
    """Get information about Qdrant collections."""
    try:
        collection_info = await system.qdrant_client.get_collection_info(
            settings.qdrant_collection_name
        )
        return {
            "collection_name": settings.qdrant_collection_name,
            "info": collection_info
        }
    except Exception as e:
        logger.error("Failed to get collection info", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {str(e)}")

@app.get("/graph/stats")
async def get_graph_stats(system: NexusRAGSystem = Depends(get_nexusrag_system)):
    """Get Neo4j graph statistics."""
    try:
        # Get node and relationship counts
        node_stats = await system.neo4j_client.execute_query(
            "MATCH (n) RETURN labels(n) as label, count(n) as count"
        )
        
        rel_stats = await system.neo4j_client.execute_query(
            "MATCH ()-[r]->() RETURN type(r) as relationship_type, count(r) as count"
        )
        
        return {
            "nodes": {stat["label"][0] if stat["label"] else "Unknown": stat["count"] for stat in node_stats},
            "relationships": {stat["relationship_type"]: stat["count"] for stat in rel_stats}
        }
        
    except Exception as e:
        logger.error("Failed to get graph stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get graph stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )
