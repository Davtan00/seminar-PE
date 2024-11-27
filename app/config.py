from pydantic_settings import BaseSettings 
from functools import lru_cache
import os
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Prompt Engineering Seminar"
    PROJECT_VERSION: str = "1.0.0"
    OPENAI_API_KEY: str 
    API_KEY: str
    TC_API_URL: str
    HUGGINGFACE_API_KEY: str
    MODEL_CACHE_DIR: str = "./model_cache"
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    CORS_METHODS: List[str] = ["*"]
    CORS_HEADERS: List[str] = ["Content-Type", "Authorization", "X-API-Key"]
    CORS_CREDENTIALS: bool = True
    CORS_MAX_AGE: int = 3600
    class Config:
        env_file = ".env"
        extra = "allow"  
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

# Ensure cache directory exists
os.makedirs(get_settings().MODEL_CACHE_DIR, exist_ok=True)
# Set environment variable for transformers
os.environ['TRANSFORMERS_CACHE'] = get_settings().MODEL_CACHE_DIR
