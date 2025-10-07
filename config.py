"""
Configuration management for NovoMD
"""
import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Authentication
    api_key: str = os.getenv("NOVOMD_API_KEY", "")

    # Server Configuration
    port: int = int(os.getenv("PORT", "8010"))
    host: str = os.getenv("HOST", "0.0.0.0")

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Feature Flags
    enable_rdkit: bool = True
    enable_openbabel: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Validate that API key is set
if not settings.api_key:
    import logging
    logging.warning("⚠️  NOVOMD_API_KEY not set. API authentication will not work.")
    logging.warning("   Set NOVOMD_API_KEY environment variable or create .env file")
