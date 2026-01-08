"""
Configuration management for NovoMD
"""

import os
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Authentication
    api_key: str = os.getenv("NOVOMD_API_KEY", "")

    # Server Configuration
    port: int = int(os.getenv("PORT", "8010"))
    host: str = os.getenv("HOST", "0.0.0.0")  # nosec B104

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # CORS Configuration
    # Comma-separated list of allowed origins, or "*" for all (not recommended for production)
    cors_origins: str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")

    # Rate Limiting
    rate_limit: str = os.getenv("RATE_LIMIT", "100/minute")

    # Feature Flags
    enable_rdkit: bool = True
    enable_openbabel: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from comma-separated string"""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


# Global settings instance
settings = Settings()

# Validate that API key is set
if not settings.api_key:
    import logging

    logging.warning("NOVOMD_API_KEY not set. API authentication will not work.")
    logging.warning("   Set NOVOMD_API_KEY environment variable or create .env file")
