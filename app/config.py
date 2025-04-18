"""Configuration module"""

import os
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name) - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load env vars
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

class MistralConfig(BaseModel):
    """Config for Mistral API"""
    api_key: str = Field(..., description="Mistral API key")

    def validate_api_key(self) -> bool:
        # Check that the api key is not empty and == mistral api key
        return bool(self.api_key and self.api_key != os.getenv("MISTRAL_API_KEY"))


class PineconeConfig(BaseModel):
    """Configuration for Pinecone vector database."""

    api_key: str = Field(..., description="PINECONE_API_KEY")
    environment: str = Field(default="gcp-starter", description="Pinecone environment")
    index_name: str = Field(default="receipt-index", description="Pinecone index name")

    def validate_api_key(self) -> bool:
        # Check that the api key is not empty and == pinecone api key
        return bool(self.api_key and self.api_key != os.getenv("PINECONE_API_KEY"))



class AppConfig(BaseModel):
    """App configuration"""

    mistral: MistralConfig
    pinecone: PineconeConfig
    log_level: str = Field(default="INFO", description="Logging level")

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load config from env vars"""
        try:
            return cls(
                mistral=MistralConfig(
                    api_key=os.getenv("MISTRAL_API_KEY", ""),
                ),
                pinecone=PineconeConfig(
                    api_key=os.getenv("PINECONE_API_KEY", ""),
                    environment=os.getenv("PINECONE_ENVIRONMENT", "gcp-starter"),
                    index_name=os.getenv("PINECONE_INDEX_NAME", "receipt-index"),
                ),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
            )
        except ValidationError as e:
            logger.error(f"Configuration error: {e}")
            raise


# Global configuration instance
_config_instance: Optional[AppConfig] = None


# Singleton configuration instance
_config: Optional[AppConfig] = None

def get_config() -> AppConfig:
    """Get application configuration"""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig.from_env()

        # Set log level
        log_level = getattr(logging, _config_instance.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)

        logger.info("Configuration loaded successfully")

    return _config_instance




