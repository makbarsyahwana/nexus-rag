#!/usr/bin/env python3
"""
Usage examples for JSON document ingestion in NexusRAG.
This script demonstrates various ways to ingest JSON documents.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.ingest_json_documents import ingest_json_documents, ingest_from_file, ingest_from_directory

async def example_1_sample_data():
    """Example 1: Ingest sample JSON documents (built-in)."""
    print("=" * 60)
    print("Example 1: Ingesting sample JSON documents")
    print("=" * 60)
    
    success = await ingest_json_documents()
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    return success

async def example_2_json_file():
    """Example 2: Ingest from JSON file."""
    print("\n" + "=" * 60)
    print("Example 2: Ingesting from JSON file")
    print("=" * 60)
    
    file_path = project_root / "data" / "sample_articles.json"
    
    success = await ingest_from_file(
        str(file_path),
        content_fields=['content'],
        metadata_fields=['title', 'author', 'category', 'tags', 'published_date']
    )
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    return success

async def example_3_jsonl_file():
    """Example 3: Ingest from JSONL file."""
    print("\n" + "=" * 60)
    print("Example 3: Ingesting from JSONL file")
    print("=" * 60)
    
    file_path = project_root / "data" / "news_articles.jsonl"
    
    success = await ingest_from_file(
        str(file_path),
        content_fields=['body'],
        metadata_fields=['headline', 'author', 'publication', 'timestamp', 'category']
    )
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    return success

async def example_4_nested_json():
    """Example 4: Ingest nested JSON structure."""
    print("\n" + "=" * 60)
    print("Example 4: Ingesting nested JSON structure")
    print("=" * 60)
    
    file_path = project_root / "data" / "research_collection.json"
    
    success = await ingest_from_file(
        str(file_path),
        content_fields=['abstract', 'full_text', 'content', 'description'],
        metadata_fields=['title', 'authors', 'venue', 'year', 'keywords'],
        nested=True
    )
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    return success

async def example_5_custom_json():
    """Example 5: Ingest custom JSON objects."""
    print("\n" + "=" * 60)
    print("Example 5: Ingesting custom JSON objects")
    print("=" * 60)
    
    custom_documents = [
        {
            "product_name": "Smart Home Assistant",
            "description": "An AI-powered device that helps manage your smart home ecosystem. Features voice recognition, natural language processing, and integration with popular IoT devices.",
            "manufacturer": "TechCorp",
            "price": 299.99,
            "features": ["voice control", "AI assistant", "IoT integration"],
            "release_date": "2024-02-15"
        },
        {
            "recipe_name": "Machine Learning Model Training",
            "instructions": "1. Prepare your dataset by cleaning and preprocessing the data. 2. Split data into training, validation, and test sets. 3. Choose appropriate algorithms and hyperparameters. 4. Train the model and monitor performance metrics. 5. Evaluate on test set and deploy if satisfactory.",
            "chef": "Data Science Team",
            "difficulty": "intermediate",
            "prep_time": "2-4 hours",
            "tools_needed": ["Python", "scikit-learn", "pandas", "numpy"]
        }
    ]
    
    success = await ingest_json_documents(
        json_input=custom_documents,
        content_fields=['description', 'instructions'],
        metadata_fields=['product_name', 'recipe_name', 'manufacturer', 'chef', 'price', 'difficulty']
    )
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    return success

async def example_6_directory_batch():
    """Example 6: Batch ingest from directory."""
    print("\n" + "=" * 60)
    print("Example 6: Batch ingesting from directory")
    print("=" * 60)
    
    data_dir = project_root / "data"
    
    success = await ingest_from_directory(
        str(data_dir),
        pattern="*.json",
        content_fields=['content', 'body', 'abstract', 'description', 'instructions'],
        metadata_fields=['title', 'author', 'category', 'tags', 'timestamp', 'headline']
    )
    print(f"Result: {'✅ Success' if success else '❌ Failed'}")
    return success

async def run_all_examples():
    """Run all examples sequentially."""
    print("🚀 NexusRAG JSON Ingestion Examples")
    print("=" * 60)
    
    examples = [
        ("Sample Data", example_1_sample_data),
        ("JSON File", example_2_json_file),
        ("JSONL File", example_3_jsonl_file),
        ("Nested JSON", example_4_nested_json),
        ("Custom Objects", example_5_custom_json),
        ("Directory Batch", example_6_directory_batch),
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            success = await example_func()
            results[name] = success
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, success in results.items():
        status = "✅ Success" if success else "❌ Failed"
        print(f"{name:<20}: {status}")
    
    total_success = sum(results.values())
    total_examples = len(results)
    print(f"\nOverall: {total_success}/{total_examples} examples succeeded")

async def interactive_mode():
    """Interactive mode for testing specific examples."""
    print("🚀 NexusRAG JSON Ingestion - Interactive Mode")
    print("=" * 60)
    
    examples = {
        "1": ("Sample Data", example_1_sample_data),
        "2": ("JSON File", example_2_json_file),
        "3": ("JSONL File", example_3_jsonl_file),
        "4": ("Nested JSON", example_4_nested_json),
        "5": ("Custom Objects", example_5_custom_json),
        "6": ("Directory Batch", example_6_directory_batch),
        "a": ("All Examples", run_all_examples),
    }
    
    while True:
        print("\nAvailable examples:")
        for key, (name, _) in examples.items():
            print(f"  {key}. {name}")
        print("  q. Quit")
        
        choice = input("\nSelect an example (1-6, a, or q): ").strip().lower()
        
        if choice == 'q':
            print("Goodbye! 👋")
            break
        elif choice in examples:
            name, example_func = examples[choice]
            print(f"\n🏃 Running: {name}")
            try:
                await example_func()
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NexusRAG JSON ingestion examples")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode")
    parser.add_argument("--example", "-e", type=int, choices=range(1, 7),
                       help="Run specific example (1-6)")
    
    args = parser.parse_args()
    
    if args.interactive:
        asyncio.run(interactive_mode())
    elif args.example:
        example_funcs = {
            1: example_1_sample_data,
            2: example_2_json_file,
            3: example_3_jsonl_file,
            4: example_4_nested_json,
            5: example_5_custom_json,
            6: example_6_directory_batch,
        }
        asyncio.run(example_funcs[args.example]())
    else:
        # Run all examples by default
        asyncio.run(run_all_examples())
