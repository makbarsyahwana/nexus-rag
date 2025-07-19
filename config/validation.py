import os
from pathlib import Path
from typing import Dict, Any, List
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from .settings import Settings

class SettingsValidator:
    """Validate configuration settings and connections."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.validation_errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "aws": self.validate_aws_credentials(),
            "models": self.validate_model_settings(),
            "databases": {
                "qdrant": self.validate_qdrant_settings(),
                "neo4j": self.validate_neo4j_settings()
            }
        }
        
        results["errors"] = self.validation_errors
        results["warnings"] = self.warnings
        results["valid"] = len(self.validation_errors) == 0
        
        return results
    
    def validate_aws_credentials(self) -> bool:
        """Validate AWS credentials and Bedrock access."""
        try:
            session = boto3.Session(
                region_name=self.settings.aws_region,
                aws_access_key_id=self.settings.aws_access_key_id,
                aws_secret_access_key=self.settings.aws_secret_access_key,
                aws_session_token=self.settings.aws_session_token
            )
            
            # Test credentials
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            
            # Test Bedrock access
            bedrock = session.client('bedrock', region_name=self.settings.aws_region)
            bedrock.list_foundation_models()
            
            return True
            
        except NoCredentialsError:
            self.warnings.append("No AWS credentials found. Using IAM role or environment.")
            return True  # This might be fine if using IAM roles
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'UnauthorizedOperation':
                self.validation_errors.append("AWS credentials lack Bedrock permissions")
            else:
                self.validation_errors.append(f"AWS error: {error_code}")
            return False
            
        except Exception as e:
            self.validation_errors.append(f"AWS validation failed: {str(e)}")
            return False
    
    def validate_model_settings(self) -> bool:
        """Validate model IDs and settings."""
        valid = True
        
        # Check LLM model format
        if not self.settings.llm_model_id:
            self.validation_errors.append("LLM model ID is required")
            valid = False
        
        # Check embedding model format
        if not self.settings.embedding_model_id:
            self.validation_errors.append("Embedding model ID is required")
            valid = False
        
        # Validate model-specific settings
        if self.settings.temperature < 0 or self.settings.temperature > 1:
            self.validation_errors.append("Temperature must be between 0 and 1")
            valid = False
        
        if self.settings.max_tokens <= 0:
            self.validation_errors.append("Max tokens must be positive")
            valid = False
        
        return valid
    
    def validate_qdrant_settings(self) -> bool:
        """Validate Qdrant connection settings."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.settings.qdrant_url)
            
            if not parsed.scheme or not parsed.netloc:
                self.validation_errors.append("Invalid Qdrant URL format")
                return False
            
            if not self.settings.qdrant_collection_name:
                self.validation_errors.append("Qdrant collection name is required")
                return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Qdrant validation failed: {str(e)}")
            return False
    
    def validate_neo4j_settings(self) -> bool:
        """Validate Neo4j connection settings."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(self.settings.neo4j_uri)
            
            if not parsed.scheme or not parsed.netloc:
                self.validation_errors.append("Invalid Neo4j URI format")
                return False
            
            if not self.settings.neo4j_user or not self.settings.neo4j_password:
                self.validation_errors.append("Neo4j credentials are required")
                return False
            
            return True
            
        except Exception as e:
            self.validation_errors.append(f"Neo4j validation failed: {str(e)}")
            return False

def load_and_validate_settings() -> tuple[Settings, Dict[str, Any]]:
    """Load settings and run validation."""
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Copy .env.example to .env and configure.")
        print("Using default settings and environment variables.")
    
    # Load settings
    settings = Settings()
    
    # Validate settings
    validator = SettingsValidator(settings)
    validation_results = validator.validate_all()
    
    return settings, validation_results

def print_validation_results(results: Dict[str, Any]) -> None:
    """Print validation results in a user-friendly format."""
    
    if results["valid"]:
        print("✅ All settings validation passed!")
    else:
        print("❌ Settings validation failed!")
        
    if results["errors"]:
        print("\n🚨 Errors:")
        for error in results["errors"]:
            print(f"  - {error}")
    
    if results["warnings"]:
        print("\n⚠️  Warnings:")
        for warning in results["warnings"]:
            print(f"  - {warning}")
    
    print(f"\n📊 Validation Summary:")
    print(f"  AWS: {'✅' if results['aws'] else '❌'}")
    print(f"  Models: {'✅' if results['models'] else '❌'}")
    print(f"  Qdrant: {'✅' if results['databases']['qdrant'] else '❌'}")
    print(f"  Neo4j: {'✅' if results['databases']['neo4j'] else '❌'}")