"""Config module"""

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

class AppConfig(BaseModel):
    mistral: MistralConfig
    log_level: str = Field(default="INFO", description="Logging level")

    @classmethod
    def from_env(cls) -> "AppConfig":
