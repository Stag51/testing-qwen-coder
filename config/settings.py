"""
Configuration settings for Federated Multi-Modal Diagnostic Agentic Swarm
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional


class FederatedConfig(BaseSettings):
    """Federated learning configuration"""
    num_clients: int = Field(default=5, description="Number of hospital clients")
    aggregation_rounds: int = Field(default=100, description="Number of federated rounds")
    local_epochs: int = Field(default=5, description="Local training epochs per round")
    batch_size: int = Field(default=32, description="Training batch size")
    learning_rate: float = Field(default=0.001, description="Learning rate")
    differential_privacy_epsilon: float = Field(default=1.0, description="DP epsilon")
    differential_privacy_delta: float = Field(default=1e-5, description="DP delta")
    
    class Config:
        env_prefix = "FED_"


class QdrantConfig(BaseSettings):
    """Qdrant vector database configuration"""
    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant port")
    grpc_port: int = Field(default=6334, description="Qdrant gRPC port")
    radiology_collection: str = Field(default="radiology_embeddings", description="Radiology collection name")
    genomics_collection: str = Field(default="genomics_embeddings", description="Genomics collection name")
    fusion_collection: str = Field(default="fusion_embeddings", description="Multi-modal fusion collection name")
    embedding_dim: int = Field(default=768, description="Embedding dimension")
    
    class Config:
        env_prefix = "QDRANT_"


class APIConfig(BaseSettings):
    """FastAPI configuration"""
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    api_title: str = Field(default="Federated Diagnostic Swarm API", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    
    class Config:
        env_prefix = "API_"


class BioNeMoConfig(BaseSettings):
    """NVIDIA BioNeMo configuration"""
    api_key: Optional[str] = Field(default=None, description="BioNeMo API key")
    base_url: str = Field(default="https://api.bionemo.nvidia.com/v1", description="BioNeMo API URL")
    model_name: str = Field(default="nvidia/bionemo-large", description="Default BioNeMo model")
    max_tokens: int = Field(default=512, description="Max generation tokens")
    temperature: float = Field(default=0.7, description="Generation temperature")
    
    class Config:
        env_prefix = "BIONEMO_"


class FHIRConfig(BaseSettings):
    """FHIR server configuration"""
    server_url: str = Field(default="http://localhost:8080/fhir", description="FHIR server URL")
    username: Optional[str] = Field(default=None, description="FHIR username")
    password: Optional[str] = Field(default=None, description="FHIR password")
    resource_types: List[str] = Field(
        default=["Patient", "Observation", "DiagnosticReport", "ImagingStudy", "MolecularSequence"],
        description="FHIR resource types to access"
    )
    
    class Config:
        env_prefix = "FHIR_"


class AppConfig(BaseSettings):
    """Main application configuration"""
    federated: FederatedConfig = Field(default_factory=FederatedConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    bionemo: BioNeMoConfig = Field(default_factory=BioNeMoConfig)
    fhir: FHIRConfig = Field(default_factory=FHIRConfig)
    
    # General settings
    log_level: str = Field(default="INFO", description="Logging level")
    data_dir: str = Field(default="./data", description="Data directory")
    model_dir: str = Field(default="./models", description="Model checkpoint directory")
    cache_dir: str = Field(default="./cache", description="Cache directory")
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


# Global config instance
config = AppConfig()
