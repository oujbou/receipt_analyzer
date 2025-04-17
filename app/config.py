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

class AppConfig(BaseModel):
    """App configuration"""

    mistral: MistralConfig
    log_level: str = Field(default="INFO", description="Logging level")

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load config from env vars"""
        try:
            return cls(
                mistral=MistralConfig(
                    api_key=os.getenv("MISTRAL_API_KEY", ""),
                ),
                log_level=os.getenv("LOG_LEVEL", "INFO"),
            )
        except ValidationError as e:
            logger.error(f"Configuration error: {e}")
            raise

def get_config() -> AppConfig:
    """Get the app config"""
    config = AppConfig.from_env()

    # Validate Mistral api key
    if not config.mistral.validate_api_key():
        logger.error("Invalid or missing Mistral API key")
        raise ValueError(
            "Invalid or missing Mistral API key. "
            "Please set the MISTRAL_API_KEY environment variable."
        )
    log_level = getattr(logging, config.log_level.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)

    return config

# Singleton configuration instance
_config: Optional[AppConfig] = None

def get_settings() -> AppConfig:
    """Get application settings, loading them if needed."""
    global _config
    if _config is None:
        _config = get_config()
    return _config




