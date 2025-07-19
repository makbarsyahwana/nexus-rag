#!/bin/bash

# NexusRAG System Setup Script
# This script helps set up the NexusRAG development environment

set -e

echo "🚀 NexusRAG System Setup"
echo "========================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+ first."
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Copy environment file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "⚠️  Please edit .env file with your AWS credentials and preferences"
else
    echo "✅ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p data
echo "✅ Directories created"

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d qdrant neo4j

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🏥 Checking service health..."

# Check Qdrant
if curl -f http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant is healthy"
else
    echo "⚠️  Qdrant is not ready yet"
fi

# Check Neo4j (might take longer to start)
echo "⏳ Waiting for Neo4j to be ready..."
sleep 30

if docker exec nexusrag-neo4j cypher-shell -u neo4j -p nexusrag123 "RETURN 1" > /dev/null 2>&1; then
    echo "✅ Neo4j is healthy"
else
    echo "⚠️  Neo4j is not ready yet. It might need more time to start."
fi

echo ""
echo "🎉 Setup completed!"
echo ""
echo "Next steps:"
echo "1. Edit the .env file with your AWS credentials"
echo "2. Run 'source venv/bin/activate' to activate the virtual environment"
echo "3. Run 'python scripts/ingest_sample_data.py' to ingest sample data"
echo "4. Run 'python -m uvicorn api.main:app --reload' to start the API server"
echo "5. Visit http://localhost:8000/docs for API documentation"
echo ""
echo "For Docker deployment:"
echo "- Run 'docker-compose up -d' to start all services"
echo "- Run 'docker-compose logs -f nexusrag-api' to view logs"
echo ""
echo "Enjoy using NexusRAG! 🎯"
