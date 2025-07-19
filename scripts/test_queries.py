#!/usr/bin/env python3
"""
Sample script to test NexusRAG system queries.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nexusrag.system import NexusRAGSystem
from config.settings import Settings

# Test queries for different scenarios
TEST_QUERIES = [
    {
        "query": "What is artificial intelligence and how does it relate to machine learning?",
        "modes": ["local", "global", "hybrid", "mix"]
    },
    {
        "query": "How can I use AWS Bedrock for building AI applications?",
        "modes": ["local", "hybrid"]
    },
    {
        "query": "What are the advantages of using vector databases like Qdrant?",
        "modes": ["local", "global", "hybrid"]
    },
    {
        "query": "Explain the relationship between RAG systems and graph databases",
        "modes": ["hybrid", "mix"]
    },
    {
        "query": "What models are available in AWS Bedrock?",
        "modes": ["local", "hybrid"]
    }
]

async def test_query(system: NexusRAGSystem, query_data: dict):
    """Test a single query with different modes."""
    query = query_data["query"]
    modes = query_data["modes"]
    
    print(f"\n{'='*80}")
    print(f"🔍 Query: {query}")
    print(f"{'='*80}")
    
    results_by_mode = {}
    
    for mode in modes:
        print(f"\n📊 Mode: {mode.upper()}")
        print("-" * 40)
        
        try:
            # Perform retrieval
            retrieval_results = await system.query(
                query_text=query,
                mode=mode,
                top_k=5,
                score_threshold=0.5
            )
            
            print(f"Retrieved {len(retrieval_results)} results")
            
            # Display retrieval results
            for i, result in enumerate(retrieval_results[:3], 1):
                print(f"\n  Result {i} (Score: {result.score:.3f}, Source: {result.source}):")
                print(f"    {result.content[:200]}...")
                if result.metadata:
                    print(f"    Metadata: {result.metadata}")
            
            # Generate response
            if retrieval_results:
                print(f"\n💬 Generating response...")
                response = await system.generate_response(query, retrieval_results)
                print(f"\nResponse ({mode}):")
                print("-" * 20)
                print(response)
                
                results_by_mode[mode] = {
                    "retrieval_count": len(retrieval_results),
                    "response": response,
                    "retrieval_results": [
                        {
                            "content": r.content[:100] + "...",
                            "score": r.score,
                            "source": r.source
                        }
                        for r in retrieval_results[:3]
                    ]
                }
            else:
                print("No results found")
                results_by_mode[mode] = {
                    "retrieval_count": 0,
                    "response": None,
                    "retrieval_results": []
                }
                
        except Exception as e:
            print(f"❌ Error in mode {mode}: {e}")
            results_by_mode[mode] = {
                "error": str(e)
            }
    
    return results_by_mode

async def main():
    """Main function to test queries."""
    print("🚀 Starting NexusRAG query testing...")
    
    # Initialize settings and system
    settings = Settings()
    system = NexusRAGSystem(settings)
    
    try:
        # Initialize the system
        print("📋 Initializing NexusRAG system...")
        await system.initialize()
        
        # Check system health
        print("🏥 Checking system health...")
        health = await system.health_check()
        print(f"System status: {health['status']}")
        
        if health['status'] != 'healthy':
            print("⚠️ System is not healthy. Check your configuration and services.")
            for component, status in health['components'].items():
                print(f"  {component}: {status['status']}")
            return
        
        # Run test queries
        all_results = {}
        
        for query_data in TEST_QUERIES:
            try:
                results = await test_query(system, query_data)
                all_results[query_data["query"]] = results
            except Exception as e:
                print(f"❌ Failed to test query: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results to file
        output_file = project_root / "test_results.json"
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 Results saved to: {output_file}")
        print("\n✅ Query testing completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Close system
        await system.close()
        print("🔒 System closed")

if __name__ == "__main__":
    asyncio.run(main())
