#!/usr/bin/env python3
"""
Environment setup and validation script for NexusRAG.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from config.validation import load_and_validate_settings, print_validation_results

def create_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📋 Creating .env file from template...")
        env_file.write_text(env_example.read_text())
        print("✅ .env file created. Please edit it with your actual values.")
        return True
    elif not env_file.exists():
        print("❌ .env.example not found. Cannot create .env file.")
        return False
    else:
        print("✅ .env file already exists.")
        return True

def main():
    """Main setup function."""
    print("🚀 NexusRAG Environment Setup")
    print("=" * 50)
    
    # Create .env file if needed
    if not create_env_file():
        sys.exit(1)
    
    # Load and validate settings
    print("\n📊 Loading and validating settings...")
    settings, validation_results = load_and_validate_settings()
    
    # Print results
    print_validation_results(validation_results)
    
    # Print current configuration
    print(f"\n⚙️  Current Configuration:")
    print(f"  App: {settings.app_name} v{settings.app_version}")
    print(f"  Debug: {settings.debug}")
    print(f"  AWS Region: {settings.aws_region}")
    print(f"  LLM Model: {settings.llm_model_id}")
    print(f"  Embedding Model: {settings.embedding_model_id}")
    print(f"  Qdrant URL: {settings.qdrant_url}")
    print(f"  Neo4j URI: {settings.neo4j_uri}")
    
    # Environment variable check
    print(f"\n🔍 Environment Variables Check:")
    env_vars = [
        "AWS_REGION", "LLM_MODEL_ID", "QDRANT_URL", "NEO4J_URI"
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        status = "✅" if value else "❌"
        print(f"  {var}: {status} {value or 'Not set'}")
    
    if not validation_results["valid"]:
        print("\n❌ Please fix the validation errors before proceeding.")
        sys.exit(1)
    else:
        print("\n✅ Environment setup complete!")

if __name__ == "__main__":
    main()