# NexusRAG - Advanced RAG System

A production-ready Retrieval-Augmented Generation (RAG) system integrating AWS Bedrock, Qdrant vector database, and Neo4j graph database.

## Features

- **Multi-modal Retrieval**: Supports vector-based, graph-based, and hybrid retrieval strategies
- **AWS Bedrock Integration**: Uses Claude 3 Sonnet for LLM and Titan Text Embeddings V2 for embeddings
- **Vector Search**: High-performance similarity search with Qdrant
- **Knowledge Graphs**: Rich relationship modeling with Neo4j
- **Async Architecture**: Built with asyncio for high concurrency
- **RESTful API**: FastAPI-based API with comprehensive endpoints
- **Docker Support**: Full containerization with Docker Compose
- **Production Ready**: Includes logging, monitoring, health checks, and error handling

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   NexusRAG      │    │   AWS Bedrock   │
│   Web API       │◄──►│   System        │◄──►│   LLM & Embed   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    │                       │
            ┌───────▼────────┐    ┌────────▼────────┐
            │   Qdrant       │    │   Neo4j         │
            │   Vector DB    │    │   Graph DB      │
            └────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- AWS Account with Bedrock access
- AWS credentials configured

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nexus-rag
```

2. Copy environment variables:
```bash
cp .env.example .env
```

3. Edit `.env` file with your AWS credentials and preferences.

4. Start with Docker Compose:
```bash
docker-compose up -d
```

5. Install Python dependencies (for local development):
```bash
pip install -r requirements.txt
```

### Running the System

#### Using Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f nexusrag-api

# Stop services
docker-compose down
```

#### Local Development
```bash
# Start Qdrant and Neo4j
docker-compose up -d qdrant neo4j

# Run the API locally
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

## Usage Examples

### 1. Ingest Sample Documents

```bash
# Ingest sample text documents
python scripts/ingest_sample_data.py

# Ingest sample JSON documents
python scripts/ingest_json_documents.py --sample
```

### 2. Ingest JSON Documents

```bash
# Ingest from JSON file
python scripts/ingest_json_documents.py --file data/sample_articles.json

# Ingest from JSONL file
python scripts/ingest_json_documents.py --file data/news_articles.jsonl

# Ingest from directory (all JSON files)
python scripts/ingest_json_documents.py --directory data/

# Custom field mapping
python scripts/ingest_json_documents.py --file data/custom.json \
  --content-fields text body description \
  --metadata-fields title author date category

# Process nested JSON structures
python scripts/ingest_json_documents.py --file data/nested.json --nested
```

### 3. JSON Ingestion Examples

```bash
# Run all examples
python scripts/json_ingestion_examples.py

# Interactive mode
python scripts/json_ingestion_examples.py --interactive

# Run specific example
python scripts/json_ingestion_examples.py --example 2
```

### 4. Test Queries

```bash
python scripts/test_queries.py
```

### 3. API Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Ingest Documents
```bash
# Text documents
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '[{
    "content": "Your document content here",
    "metadata": {"topic": "example", "source": "api"},
    "doc_id": "example_doc_1"
  }]'

# JSON documents (will be processed by the API)
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '[{
    "content": "AI and machine learning are transforming industries...",
    "metadata": {
      "title": "AI Revolution", 
      "author": "Tech Expert",
      "category": "technology"
    },
    "doc_id": "ai_article_1"
  }]'
```

#### Query the System
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is artificial intelligence?",
    "mode": "hybrid",
    "top_k": 5,
    "generate_response": true
  }'
```

## API Documentation

Once the system is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region for Bedrock | `us-east-1` |
| `AWS_ACCESS_KEY_ID` | AWS access key | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Required |
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | `nexusrag123` |

### JSON Document Formats

The system supports various JSON document formats:

#### Simple JSON Array
```json
[
  {
    "title": "Document Title",
    "content": "Document content here...",
    "author": "Author Name",
    "category": "technology"
  }
]
```

#### JSONL (JSON Lines)
```
{"headline": "News Title", "body": "News content...", "date": "2024-01-01"}
{"headline": "Another News", "body": "More content...", "date": "2024-01-02"}
```

#### Nested JSON
```json
{
  "articles": {
    "article_1": {
      "title": "Title",
      "text": "Content...",
      "metadata": {"author": "Name"}
    }
  }
}
```

#### Custom Field Mapping
Use `--content-fields` and `--metadata-fields` to specify which JSON fields to use:
- **Content fields**: `content`, `text`, `body`, `description`, `article`
- **Metadata fields**: `title`, `author`, `category`, `tags`, `timestamp`, `source`

### Query Modes

- **local**: Vector-based retrieval using Qdrant embeddings
- **global**: Graph-based retrieval using Neo4j relationships
- **hybrid**: Combination of vector and graph retrieval
- **mix**: Advanced mixing of multiple strategies

## Development

### Project Structure

```
nexus-rag/
├── api/                    # FastAPI application
├── bedrock/               # AWS Bedrock client
├── qdrant/                # Qdrant vector database client
├── neo4j/                 # Neo4j graph database client
├── nexusrag/              # Core NexusRAG system
├── config/                # Configuration management
├── scripts/               # Utility scripts
│   ├── ingest_sample_data.py          # Ingest sample text documents
│   ├── ingest_json_documents.py       # Ingest JSON/JSONL documents
│   ├── json_ingestion_examples.py     # JSON ingestion examples
│   └── test_queries.py                # Test query functionality
├── data/                  # Sample data files
│   ├── sample_articles.json           # Sample JSON articles
│   ├── news_articles.jsonl            # Sample JSONL news
│   └── research_collection.json       # Nested JSON structure
├── tests/                 # Test cases
├── docker-compose.yml     # Docker composition
├── Dockerfile            # Container definition
└── requirements.txt      # Python dependencies
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-mock

# Run tests
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Deployment

### AWS ECS Fargate (Recommended)

1. Build and push Docker image to ECR
2. Create ECS task definition with proper IAM roles
3. Deploy using AWS CDK or CloudFormation templates
4. Configure load balancer and auto-scaling

### Kubernetes

```yaml
# Example deployment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nexusrag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nexusrag-api
  template:
    metadata:
      labels:
        app: nexusrag-api
    spec:
      containers:
      - name: nexusrag-api
        image: your-registry/nexusrag:latest
        ports:
        - containerPort: 8000
        env:
        - name: AWS_REGION
          value: "us-east-1"
        # Add other environment variables
```

## Monitoring and Observability

- **Structured Logging**: JSON-formatted logs with structured fields
- **Health Checks**: Comprehensive health monitoring for all components
- **Metrics**: Integration with Prometheus (can be extended)
- **Tracing**: Ready for distributed tracing integration

## Security

- **IAM Roles**: Use AWS IAM roles for Bedrock access
- **Secret Management**: Environment-based secret configuration
- **Network Security**: Containerized deployment with network isolation
- **Rate Limiting**: Built-in API rate limiting

## Performance Optimization

- **Async Architecture**: Non-blocking I/O operations
- **Connection Pooling**: Efficient database connection management
- **Caching**: Results caching (can be extended with Redis)
- **Batch Processing**: Efficient document ingestion

## Troubleshooting

### Common Issues

1. **AWS Bedrock Access**: Ensure your AWS credentials have Bedrock permissions
2. **Database Connections**: Check Qdrant and Neo4j are running and accessible
3. **Memory Usage**: Monitor memory usage with large document ingestion
4. **Rate Limits**: Be aware of AWS Bedrock rate limits

### Logs

```bash
# View application logs
docker-compose logs -f nexusrag-api

# View database logs
docker-compose logs -f qdrant neo4j
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## License

[Add your license information here]

## Support

For questions and support:
- Create an issue in the repository
- Check the documentation
- Review the test cases for usage examples
