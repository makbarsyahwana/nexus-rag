#!/usr/bin/env python3
"""
Sample script to ingest documents into NexusRAG system.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.system import NexusRAGSystem, Document
from config.settings import Settings

# Sample documents
SAMPLE_DOCUMENTS = [
    {
        "content": """
        Artificial Intelligence (AI) is a broad field of computer science that aims to create 
        systems capable of performing tasks that typically require human intelligence. These 
        tasks include learning, reasoning, problem-solving, perception, and language understanding.
        
        Machine Learning is a subset of AI that focuses on the development of algorithms and 
        statistical models that enable computers to improve their performance on a specific 
        task through experience. Deep Learning, a subset of machine learning, uses neural 
        networks with multiple layers to model and understand complex patterns in data.
        
        Natural Language Processing (NLP) is another important area of AI that deals with the 
        interaction between computers and human language. It involves developing algorithms 
        that can understand, interpret, and generate human language in a valuable way.
        """,
        "metadata": {"topic": "AI", "category": "technology", "source": "educational"}
    },
    {
        "content": """
        Retrieval-Augmented Generation (RAG) is a technique that combines the capabilities of 
        large language models with external knowledge retrieval systems. This approach allows 
        AI systems to access and utilize information from external databases or document 
        collections when generating responses.
        
        The RAG process typically involves three main steps: retrieval, augmentation, and 
        generation. First, relevant information is retrieved from an external knowledge base 
        using the input query. Then, this retrieved information is used to augment the input 
        to the language model. Finally, the language model generates a response based on both 
        the original query and the retrieved context.
        
        RAG systems often use vector databases like Qdrant or Pinecone to store and search 
        through embeddings of documents. Graph databases like Neo4j can also be used to 
        represent and traverse knowledge relationships.
        """,
        "metadata": {"topic": "RAG", "category": "technology", "source": "research"}
    },
    {
        "content": """
        AWS Bedrock is Amazon's fully managed service that provides access to foundation models 
        from leading AI companies through a single API. It offers models from Anthropic, AI21 Labs, 
        Cohere, Meta, Stability AI, and Amazon's own Titan models.
        
        Bedrock allows developers to build and scale generative AI applications without managing 
        infrastructure. It provides capabilities for text generation, embeddings, image generation, 
        and more. The service includes features for fine-tuning models, implementing guardrails, 
        and monitoring model performance.
        
        Some popular models available through Bedrock include Claude 3 from Anthropic for 
        conversational AI, Titan Text for embeddings and text generation, and Stable Diffusion 
        for image generation. Bedrock integrates well with other AWS services and provides 
        enterprise-grade security and compliance features.
        """,
        "metadata": {"topic": "AWS Bedrock", "category": "cloud_services", "source": "documentation"}
    },
    {
        "content": """
        Qdrant is an open-source vector database designed for high-performance similarity search 
        and recommendations. It's specifically built for machine learning applications that work 
        with high-dimensional vectors, such as embeddings from neural networks.
        
        Key features of Qdrant include fast similarity search using HNSW (Hierarchical Navigable 
        Small World) algorithm, payload filtering capabilities, and support for multiple distance 
        metrics including cosine similarity, dot product, and Euclidean distance. It provides 
        both REST API and gRPC interfaces for integration.
        
        Qdrant supports clustering for horizontal scaling and offers features like snapshots 
        for backup and recovery. It can handle billions of vectors efficiently and provides 
        real-time indexing capabilities. The database is written in Rust, ensuring memory 
        safety and high performance.
        """,
        "metadata": {"topic": "Qdrant", "category": "database", "source": "documentation"}
    },
    {
        "content": """
        Neo4j is a leading graph database management system that stores data in nodes and 
        relationships rather than tables or documents. It uses the Cypher query language, 
        which is specifically designed for working with graph data structures.
        
        Graph databases like Neo4j excel at handling complex relationships between data points. 
        They are particularly useful for applications like social networks, recommendation 
        engines, fraud detection, and knowledge graphs. Neo4j supports ACID transactions 
        and provides both read and write scalability.
        
        In the context of RAG systems, Neo4j can be used to store knowledge graphs that 
        represent entities and their relationships. This allows for more sophisticated 
        retrieval strategies that consider the semantic relationships between concepts, 
        not just similarity based on embeddings.
        """,
        "metadata": {"topic": "Neo4j", "category": "database", "source": "documentation"}
    }
]

async def main():
    """Main function to ingest sample documents."""
    print("🚀 Starting document ingestion...")
    
    # Initialize settings and system
    settings = Settings()
    system = NexusRAGSystem(settings)
    
    try:
        # Initialize the system
        print("📋 Initializing NexusRAG system...")
        await system.initialize()
        
        # Create Document objects
        documents = []
        for i, doc_data in enumerate(SAMPLE_DOCUMENTS):
            doc = Document(
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                doc_id=f"sample_doc_{i+1}"
            )
            documents.append(doc)
        
        # Ingest documents
        print(f"📚 Ingesting {len(documents)} documents...")
        success = await system.ingest_documents(documents)
        
        if success:
            print("✅ Documents ingested successfully!")
            
            # Test queries with different modes
            test_queries = [
                "What is Retrieval-Augmented Generation?",
                "How does AWS Bedrock work?",
                "What are the benefits of using graph databases?"
            ]
            
            for query in test_queries:
                print(f"\n🔍 Testing query: {query}")
                
                # Test different retrieval modes
                for mode in ["local", "global", "hybrid"]:
                    print(f"  Mode: {mode}")
                    results = await system.query(query, mode=mode, top_k=3)
                    print(f"    Retrieved {len(results)} results")
                    
                    if results:
                        # Generate response
                        response = await system.generate_response(query, results)
                        print(f"    Response preview: {response[:150]}...")
        else:
            print("❌ Failed to ingest documents")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Close system
        await system.close()
        print("🔒 System closed")

if __name__ == "__main__":
    asyncio.run(main())
