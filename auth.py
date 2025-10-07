"""
Simple API key authentication for NovoMD
"""
from fastapi import Header, HTTPException
from config import settings
import logging

logger = logging.getLogger(__name__)

async def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")) -> str:
    """
    Verify API key from request header

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not x_api_key:
        logger.warning("API request without X-API-Key header")
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header in your request."
        )

    if not settings.api_key:
        logger.error("NOVOMD_API_KEY not configured on server")
        raise HTTPException(
            status_code=500,
            detail="Server authentication not configured"
        )

    if x_api_key != settings.api_key:
        logger.warning(f"Invalid API key attempt: {x_api_key[:8]}...")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

    return x_api_key
