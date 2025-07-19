import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import structlog

logger = structlog.get_logger(__name__)

class BedrockClient:
    """AWS Bedrock client for LLM and embedding operations."""
    
    def __init__(
        self,
        region_name: str = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None
    ):
        """
        Initialize Bedrock client.
        
        Args:
            region_name: AWS region
            aws_access_key_id: AWS access key (optional if using IAM roles)
            aws_secret_access_key: AWS secret key (optional if using IAM roles)
            session_token: AWS session token (optional)
        """
        self.region_name = region_name
        
        session_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            })
            if session_token:
                session_kwargs["aws_session_token"] = session_token
        
        self.session = boto3.Session(**session_kwargs)
        self.bedrock_runtime = self.session.client("bedrock-runtime")
        
        logger.info("Bedrock client initialized", region=region_name)
    
    async def generate_text(
        self,
        prompt: str,
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate text using Bedrock LLM.
        
        Args:
            prompt: Input prompt
            model_id: Bedrock model ID
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional model parameters
            
        Returns:
            Generated text
        """
        try:
            # Prepare request body based on model family
            if "anthropic.claude" in model_id:
                body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}]
                }
                # Add optional parameters for Claude
                if "top_p" in kwargs:
                    body["top_p"] = kwargs["top_p"]
                if "system" in kwargs:
                    body["system"] = kwargs["system"]
                
            elif "amazon.titan-text" in model_id:
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature,
                        "topP": kwargs.get("top_p", 0.9),
                        "stopSequences": kwargs.get("stop_sequences", [])
                    }
                }
                
            elif "ai21.j2" in model_id:
                # AI21 Jurassic-2 models
                body = {
                    "prompt": prompt,
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": kwargs.get("top_p", 1.0),
                    "stopSequences": kwargs.get("stop_sequences", []),
                    "countPenalty": {
                        "scale": kwargs.get("count_penalty", 0.0)
                    },
                    "presencePenalty": {
                        "scale": kwargs.get("presence_penalty", 0.0)
                    },
                    "frequencyPenalty": {
                        "scale": kwargs.get("frequency_penalty", 0.0)
                    }
                }
                
            elif "cohere.command" in model_id:
                # Cohere Command models
                body = {
                    "message": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "p": kwargs.get("top_p", 0.75),
                    "k": kwargs.get("top_k", 0),
                    "stop_sequences": kwargs.get("stop_sequences", []),
                    "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                    "presence_penalty": kwargs.get("presence_penalty", 0.0),
                    "seed": kwargs.get("seed"),
                    "return_likelihoods": kwargs.get("return_likelihoods", "NONE")
                }
                
            elif "meta.llama2" in model_id or "meta.llama3" in model_id:
                # Meta Llama models
                body = {
                    "prompt": prompt,
                    "max_gen_len": max_tokens,
                    "temperature": temperature,
                    "top_p": kwargs.get("top_p", 0.9)
                }
                
            elif "mistral" in model_id:
                # Mistral models
                body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": kwargs.get("top_p", 0.7),
                    "top_k": kwargs.get("top_k", 50),
                    "stop": kwargs.get("stop_sequences", [])
                }
                
            elif "amazon.nova" in model_id:
                # Amazon Nova models (if available)
                body = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature,
                        "topP": kwargs.get("top_p", 0.9)
                    }
                }
                
            else:
                # Generic fallback for unknown models
                logger.warning(f"Unknown model family for {model_id}, using generic format")
                body = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": kwargs.get("top_p", 0.9)
                }
        
            # Make async call to Bedrock
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.bedrock_runtime.invoke_model(
                    modelId=model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body)
                )
            )
            
            # Parse response based on model family
            response_body = json.loads(response["body"].read())
            
            if "anthropic.claude" in model_id:
                return response_body["content"][0]["text"]
            elif "amazon.titan-text" in model_id or "amazon.nova" in model_id:
                return response_body["results"][0]["outputText"]
            elif "ai21.j2" in model_id:
                return response_body["completions"][0]["data"]["text"]
            elif "cohere.command" in model_id:
                return response_body["text"]
            elif "meta.llama" in model_id:
                return response_body["generation"]
            elif "mistral" in model_id:
                return response_body["outputs"][0]["text"]
            else:
                # Generic fallback
                # Try common response formats
                if "text" in response_body:
                    return response_body["text"]
                elif "generation" in response_body:
                    return response_body["generation"]
                elif "completions" in response_body:
                    return response_body["completions"][0]["text"]
                elif "outputs" in response_body:
                    return response_body["outputs"][0]["text"]
                else:
                    logger.error(f"Unknown response format for model {model_id}")
                    raise ValueError(f"Unable to parse response for model {model_id}")
        
        except (ClientError, BotoCoreError) as e:
            logger.error("Bedrock API error", error=str(e), model_id=model_id)
            raise
        except Exception as e:
            logger.error("Unexpected error in text generation", error=str(e))
            raise
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model_id: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,
        normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings using Bedrock embedding model.
        
        Args:
            texts: List of texts to embed
            model_id: Bedrock embedding model ID
            dimensions: Embedding dimensions
            normalize: Whether to normalize embeddings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            
            for text in texts:
                body = {
                    "inputText": text,
                    "dimensions": dimensions,
                    "normalize": normalize
                }
                
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self.bedrock_runtime.invoke_model(
                        modelId=model_id,
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(body)
                    )
                )
                
                response_body = json.loads(response["body"].read())
                embeddings.append(response_body["embedding"])
            
            logger.info("Generated embeddings", count=len(embeddings), dimensions=dimensions)
            return embeddings
            
        except (ClientError, BotoCoreError) as e:
            logger.error("Bedrock embedding error", error=str(e), model_id=model_id)
            raise
        except Exception as e:
            logger.error("Unexpected error in embedding generation", error=str(e))
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Bedrock service health."""
        try:
            # List available models to test connectivity
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.session.client("bedrock").list_foundation_models()
            )
            
            return {
                "status": "healthy",
                "region": self.region_name,
                "available_models": len(response.get("modelSummaries", []))
            }
        except Exception as e:
            logger.error("Bedrock health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e)
            }
