import asyncio
from typing import Dict, List, Optional, Any, Tuple
import uuid
from qdrant_client import QdrantClient as QdrantClientSDK
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
import structlog

logger = structlog.get_logger(__name__)

class QdrantClient:
    """Qdrant vector database client for semantic search."""
    
    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        timeout: int = 60
    ):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.url = url
        self.client = QdrantClientSDK(
            url=url,
            api_key=api_key,
            timeout=timeout
        )
        
        logger.info("Qdrant client initialized", url=url)
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1024,
        distance: str = "Cosine"
    ) -> bool:
        """
        Create a collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric (Cosine, Dot, Euclid)
            
        Returns:
            True if successful
        """
        try:
            distance_map = {
                "Cosine": models.Distance.COSINE,
                "Dot": models.Distance.DOT,
                "Euclid": models.Distance.EUCLID
            }
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=distance_map.get(distance, models.Distance.COSINE)
                    )
                )
            )
            
            logger.info("Collection created", collection=collection_name, vector_size=vector_size)
            return True
            
        except Exception as e:
            logger.error("Failed to create collection", collection=collection_name, error=str(e))
            return False
    
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(
                None,
                lambda: self.client.get_collections()
            )
            
            collection_names = [col.name for col in collections.collections]
            return collection_name in collection_names
            
        except Exception as e:
            logger.error("Failed to check collection existence", collection=collection_name, error=str(e))
            return False
    
    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Upsert vectors into collection.
        
        Args:
            collection_name: Target collection
            vectors: List of embedding vectors
            payloads: List of metadata for each vector
            ids: Optional list of IDs (auto-generated if None)
            
        Returns:
            True if successful
        """
        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in vectors]
            
            points = [
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ]
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.upsert(
                    collection_name=collection_name,
                    points=points
                )
            )
            
            logger.info("Vectors upserted", collection=collection_name, count=len(vectors))
            return True
            
        except Exception as e:
            logger.error("Failed to upsert vectors", collection=collection_name, error=str(e))
            return False
    
    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            collection_name: Target collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional metadata filters
            
        Returns:
            List of search results with scores and payloads
        """
        try:
            search_filter = None
            if filter_conditions:
                search_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                        for key, value in filter_conditions.items()
                    ]
                )
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    query_filter=search_filter
                )
            )
            
            formatted_results = [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                }
                for result in results
            ]
            
            logger.info("Vector search completed", collection=collection_name, results_count=len(formatted_results))
            return formatted_results
            
        except Exception as e:
            logger.error("Failed to search vectors", collection=collection_name, error=str(e))
            return []
    
    async def delete_vectors(
        self,
        collection_name: str,
        ids: List[str]
    ) -> bool:
        """
        Delete vectors by IDs.
        
        Args:
            collection_name: Target collection
            ids: List of vector IDs to delete
            
        Returns:
            True if successful
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(
                        points=ids
                    )
                )
            )
            
            logger.info("Vectors deleted", collection=collection_name, count=len(ids))
            return True
            
        except Exception as e:
            logger.error("Failed to delete vectors", collection=collection_name, error=str(e))
            return False
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection information."""
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                None,
                lambda: self.client.get_collection(collection_name)
            )
            
            return {
                "status": info.status,
                "vector_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value
            }
            
        except Exception as e:
            logger.error("Failed to get collection info", collection=collection_name, error=str(e))
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Qdrant service health."""
        try:
            loop = asyncio.get_event_loop()
            collections = await loop.run_in_executor(
                None,
                lambda: self.client.get_collections()
            )
            
            return {
                "status": "healthy",
                "url": self.url,
                "collections_count": len(collections.collections)
            }
            
        except Exception as e:
            logger.error("Qdrant health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }
