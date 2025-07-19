from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application configuration settings loaded from environment variables."""
    
    # App settings
    app_name: str = "NexusRAG System"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    
    # AWS Bedrock settings
    aws_region: str = "ap-southeast-1"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    
    # Bedrock model settings
    llm_model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    embedding_model_id: str = "amazon.titan-embed-text-v2:0"
    embedding_dimensions: int = 1024
    max_tokens: int = 1000
    temperature: float = 0.7
    
    # Qdrant settings
    qdrant_url: str = None
    qdrant_api_key: Optional[str] = None
    qdrant_timeout: int = 60
    qdrant_collection_name: str = None
    
    # Neo4j settings
    neo4j_uri: str = None
    neo4j_user: str = None
    neo4j_password: str = None
    neo4j_database: str = None
    neo4j_max_connection_lifetime: int = 3600
    neo4j_max_connection_pool_size: int = 50
    
    # NexusRAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_top_k: int = 10
    retrieval_score_threshold: float = 0.7
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Additional LLM parameters
    top_p: float = 0.9
    top_k: int = 50
    system_prompt: Optional[str] = None
    
    # Health check settings
    health_check_timeout: int = 30
    health_check_interval: int = 60
    
    # Security settings
    api_key: Optional[str] = None
    cors_origins: str = "*"
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }
