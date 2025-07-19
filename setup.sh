#!/bin/bash

# NexusRAG System Setup Script
# This script helps set up the NexusRAG development environment

set -e

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo "🔍 Loading environment variables from .env file"
    # Export all non-comment, non-empty lines
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
else
    echo "⚠️  .env file not found. Creating from template..."
    if [ -f ".env.template" ]; then
        cp .env.template .env
        echo "✅ Created .env file from template"
        echo "⚠️  Please edit .env file with your configuration and run this script again"
        exit 1
    else
        echo "❌ Error: .env.template not found. Please create it first."
        exit 1
    fi
fi

# Set default values if not set in .env
: ${NEO4J_CONTAINER_NAME:=nexusrag-neo4j}
: ${NEO4J_USER:=neo4j}
: ${NEO4J_PASSWORD:=nexusrag123}
: ${QDRANT_URL:=localhost}
: ${QDRANT_PORT:=6333}
: ${API_HOST:=0.0.0.0}
: ${API_PORT:=8000}
: ${AWS_REGION:=us-east-1}
: ${DEBUG:=false}
: ${LOG_LEVEL:=INFO}

echo "🚀 NexusRAG System Setup"
echo "========================="

# Function to check Qdrant connection
check_qdrant_connection() {
    local url=$1
    local api_key=${2:-}
    echo "🔍 Checking Qdrant connection at $url..."
    
    # Prepare curl command with API key if provided
    local curl_cmd="curl -s -f -L"
    if [ -n "$api_key" ]; then
        curl_cmd+=" -H \"api-key: $api_key\""
    fi
    
    # Ensure URL ends with exactly one slash
    url="${url%/}/"
    
    # Check connection to root endpoint
    if eval "$curl_cmd ${url}" > /dev/null 2>&1; then
        echo "✅ Successfully connected to Qdrant at $url"
        return 0
    else
        echo "❌ Failed to connect to Qdrant at $url"
        return 1
    fi
}

# Function to check Neo4j connection
check_neo4j_connection() {
    local uri=$1
    local user=$2
    local password=$3

    # Convert bolt/neo4j URI to HTTP/HTTPS
    if [[ "$uri" == bolt* ]] || [[ "$uri" == neo4j* ]]; then
        uri="https://${uri#*//}"  # Replace bolt:// or neo4j:// with https://
    fi
    
    # Ensure URI has protocol
    if [[ "$uri" != http* ]]; then
        uri="https://$uri"
    fi
    
    # Remove any trailing slashes and add db/data/ path
    local base_uri="${uri%%/}"
    
    echo "🔍 Checking Neo4j connection at $base_uri..."
    
    # Test connection by making a simple query
    local response=$(curl -s -o /dev/null -w "%{http_code}" -X GET \
        -H "Accept: application/json" \
        -H "Authorization: Basic $(echo -n "$user:$password" | base64)" \
        "$base_uri" 2>/dev/null)

    if [ "$response" = "200" ]; then
        echo "✅ Successfully connected to Neo4j at $base_uri"
        return 0
    else
        echo "❌ Failed to connect to Neo4j at $base_uri (HTTP $response)"
        return 1
    fi
}

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

# Check if Docker Compose is available
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please make sure Docker Desktop is running."
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

# Check if we should use local Docker containers
USE_LOCAL_QDRANT=false
USE_LOCAL_NEO4J=false

# Check if Qdrant is configured to use localhost
if [[ "$QDRANT_URL" == "localhost" || "$QDRANT_URL" == "127.0.0.1" ]]; then
    USE_LOCAL_QDRANT=true
    echo "🔍 Qdrant is configured to use local Docker container"
else
    echo "ℹ️  Qdrant is configured to use external service at $QDRANT_URL"
    # Test connection to external Qdrant
    check_qdrant_connection "$QDRANT_URL" "$QDRANT_API_KEY" || {
        echo "⚠️  Warning: Could not connect to external Qdrant service. Please check your configuration."
    }
fi

# Check if Neo4j is configured to use localhost
if [[ "$NEO4J_URI" == *"localhost"* || "$NEO4J_URI" == *"127.0.0.1"* ]]; then
    USE_LOCAL_NEO4J=true
    echo "🔍 Neo4j is configured to use local Docker container"
else
    echo "ℹ️  Neo4j is configured to use external service at $NEO4J_URI"
    # Test connection to external Neo4j
    check_neo4j_connection "$NEO4J_URI" "$NEO4J_USER" "$NEO4J_PASSWORD" || {
        echo "⚠️  Warning: Could not connect to external Neo4j service. Please check your configuration."
    }
fi

# Start local Docker services if needed
if [ "$USE_LOCAL_QDRANT" = true ] || [ "$USE_LOCAL_NEO4J" = true ]; then
    echo "🐳 Starting Docker services..."
    
    # Start only the services that are configured to use local Docker
    SERVICES=()
    [ "$USE_LOCAL_QDRANT" = true ] && SERVICES+=("qdrant")
    [ "$USE_LOCAL_NEO4J" = true ] && SERVICES+=("neo4j")
    
    docker compose up -d "${SERVICES[@]}"
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    echo "🏥 Checking service health..."
    
    # Check Qdrant health if using local Qdrant
    if [ "$USE_LOCAL_QDRANT" = true ]; then
        if curl -f http://${QDRANT_HOST}:${QDRANT_PORT}/health > /dev/null 2>&1; then
            echo "✅ Qdrant is healthy"
        else
            echo "⚠️  Qdrant is not ready yet"
        fi
    fi
    
    # Check Neo4j health if using local Neo4j
    if [ "$USE_LOCAL_NEO4J" = true ]; then
        echo "⏳ Waiting for Neo4j to be ready..."
        sleep 30
        
        if docker exec ${NEO4J_CONTAINER_NAME} cypher-shell -u ${NEO4J_USER} -p ${NEO4J_PASSWORD} "RETURN 1" > /dev/null 2>&1; then
            echo "✅ Neo4j is healthy"
        else
            echo "⚠️  Neo4j is not ready yet. It might need more time to start."
        fi
    fi
else
    echo "ℹ️  No local Docker services to start"
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
echo "NexusRAG! Ready 🎯"