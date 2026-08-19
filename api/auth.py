"""
Simple API-key gating for the REST layer. Not a substitute for a full
OAuth/SSO setup in a real enterprise deployment, but it stops the API
from being wide open — which was one of the flagged gaps.

Keys are read from the API_KEYS env var (comma-separated) and passed by
clients via the `X-API-Key` header.
"""
from fastapi import Header, HTTPException, status

from rag import config


async def require_api_key(x_api_key: str = Header(default="")) -> str:
    if not config.REQUIRE_API_KEY:
        return "auth-disabled"

    if not config.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server misconfigured: no API_KEYS set. See .env.example.",
        )

    if x_api_key not in config.API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )
    return x_api_key
