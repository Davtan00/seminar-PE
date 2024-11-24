from fastapi import HTTPException, Header
from app.config import get_settings

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != get_settings().API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return x_api_key 