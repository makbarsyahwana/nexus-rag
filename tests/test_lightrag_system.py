import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.system import NexusRAGSystem, Document
from config.settings import Settings

@pytest.fixture
def settings():
    """Test settings fixture."""
    return Settings(
        aws_region="us-east-1",
        qdrant_url="http://localhost:6333",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="test"
    )

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(
            content="This is a test document about artificial intelligence.",
            metadata={"topic": "AI", "category": "test"},
            doc_id="test_doc_1"
        ),
        Document(
            content="This document discusses machine learning algorithms.",
            metadata={"topic": "ML", "category": "test"},
            doc_id="test_doc_2"
        )
    ]

@pytest.fixture
async def mock_nexusrag_system(settings):
    """Mock NexusRAG system for testing."""
    with patch('nexusrag.system.BedrockClient') as mock_bedrock, \
         patch('nexusrag.system.QdrantClient') as mock_qdrant, \
         patch('nexusrag.system.Neo4jClient') as mock_neo4j:
        
        # Setup mock clients
        mock_bedrock_instance = AsyncMock()
        mock_qdrant_instance = AsyncMock()
        mock_neo4j_instance = AsyncMock()
        
        mock_bedrock.return_value = mock_bedrock_instance
        mock_qdrant.return_value = mock_qdrant_instance
        mock_neo4j.return_value = mock_neo4j_instance
        
        # Create system
        system = NexusRAGSystem(settings)
        
        # Setup mock responses
        mock_bedrock_instance.generate_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_bedrock_instance.generate_text.return_value = "Test response"
        mock_bedrock_instance.health_check.return_value = {"status": "healthy"}
        
        mock_qdrant_instance.collection_exists.return_value = True
        mock_qdrant_instance.search_vectors.return_value = [
            {
                "id": "test_id",
                "score": 0.9,
                "payload": {"content": "Test content", "doc_id": "test_doc_1"}
            }
        ]
        mock_qdrant_instance.upsert_vectors.return_value = True
        mock_qdrant_instance.health_check.return_value = {"status": "healthy"}
        
        mock_neo4j_instance.create_node.return_value = None
        mock_neo4j_instance.create_relationship.return_value = True
        mock_neo4j_instance.execute_query.return_value = [
            {"content": "Test graph content", "chunk_id": "test_chunk_1"}
        ]
        mock_neo4j_instance.health_check.return_value = {"status": "healthy"}
        
        yield system

class TestNexusRAGSystem:
    """Test cases for NexusRAG system."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_nexusrag_system):
        """Test system initialization."""
        await mock_nexusrag_system.initialize()
        
        # Verify clients were called
        mock_nexusrag_system.qdrant_client.collection_exists.assert_called_once()
        mock_nexusrag_system.neo4j_client.create_indexes.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_document_ingestion(self, mock_nexusrag_system, sample_documents):
        """Test document ingestion."""
        result = await mock_nexusrag_system.ingest_documents(sample_documents)
        
        assert result is True
        
        # Verify embedding generation was called
        assert mock_nexusrag_system.bedrock_client.generate_embeddings.call_count == len(sample_documents)
        
        # Verify vector storage was called
        assert mock_nexusrag_system.qdrant_client.upsert_vectors.call_count == len(sample_documents)
        
        # Verify graph storage was called
        assert mock_nexusrag_system.neo4j_client.create_node.call_count >= len(sample_documents)
    
    @pytest.mark.asyncio
    async def test_vector_retrieval(self, mock_nexusrag_system):
        """Test vector-based retrieval."""
        results = await mock_nexusrag_system._vector_retrieval("test query", 5, 0.7)
        
        assert len(results) == 1
        assert results[0].content == "Test content"
        assert results[0].score == 0.9
        assert results[0].source == "vector"
        
        # Verify embedding generation and search were called
        mock_nexusrag_system.bedrock_client.generate_embeddings.assert_called_once()
        mock_nexusrag_system.qdrant_client.search_vectors.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graph_retrieval(self, mock_nexusrag_system):
        """Test graph-based retrieval."""
        results = await mock_nexusrag_system._graph_retrieval("test query", 5)
        
        assert len(results) > 0
        assert results[0].source == "graph"
        
        # Verify Neo4j query was called
        mock_nexusrag_system.neo4j_client.execute_query.assert_called()
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval(self, mock_nexusrag_system):
        """Test hybrid retrieval combining vector and graph."""
        results = await mock_nexusrag_system._hybrid_retrieval("test query", 10, 0.7)
        
        assert len(results) > 0
        
        # Should have called both vector and graph retrieval
        mock_nexusrag_system.bedrock_client.generate_embeddings.assert_called()
        mock_nexusrag_system.qdrant_client.search_vectors.assert_called()
        mock_nexusrag_system.neo4j_client.execute_query.assert_called()
    
    @pytest.mark.asyncio
    async def test_response_generation(self, mock_nexusrag_system):
        """Test response generation with retrieved context."""
        from nexusrag.system import RetrievalResult
        
        retrieval_results = [
            RetrievalResult(
                content="Test content for response generation",
                score=0.9,
                metadata={"doc_id": "test_doc_1"},
                source="vector"
            )
        ]
        
        response = await mock_nexusrag_system.generate_response(
            "test query",
            retrieval_results
        )
        
        assert response == "Test response"
        mock_nexusrag_system.bedrock_client.generate_text.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_nexusrag_system):
        """Test system health check."""
        health = await mock_nexusrag_system.health_check()
        
        assert health["status"] == "healthy"
        assert "components" in health
        assert "bedrock" in health["components"]
        assert "qdrant" in health["components"]
        assert "neo4j" in health["components"]
    
    @pytest.mark.asyncio
    async def test_query_modes(self, mock_nexusrag_system):
        """Test different query modes."""
        query = "test query"
        
        # Test all supported modes
        for mode in ["local", "global", "hybrid", "mix"]:
            results = await mock_nexusrag_system.query(query, mode=mode)
            assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_document_chunking(self, mock_nexusrag_system):
        """Test document chunking functionality."""
        long_content = "This is a test. " * 100  # Create long content
        chunks = mock_nexusrag_system._chunk_document(long_content)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= mock_nexusrag_system.settings.chunk_size + 50 for chunk in chunks)

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in various scenarios."""
    settings = Settings()
    
    with patch('nexusrag.system.BedrockClient') as mock_bedrock:
        mock_bedrock_instance = AsyncMock()
        mock_bedrock_instance.generate_embeddings.side_effect = Exception("API Error")
        mock_bedrock.return_value = mock_bedrock_instance
        
        system = NexusRAGSystem(settings)
        
        # Test that errors are properly handled
        with pytest.raises(Exception):
            await system.bedrock_client.generate_embeddings(["test"])
