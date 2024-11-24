from pydantic_settings import BaseSettings 
from functools import lru_cache
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Prompt Engineering Seminar"
    PROJECT_VERSION: str = "1.0.0"
    OPENAI_API_KEY: str
    API_KEY: str
    TC_API_URL: str
    HUGGINGFACE_API_KEY: str
    MODEL_CACHE_DIR: str = "./model_cache"
    class Config:
        env_file = ".env"
        extra = "allow"  

@lru_cache()
def get_settings():
    return Settings()

# Ensure cache directory exists
os.makedirs(get_settings().MODEL_CACHE_DIR, exist_ok=True)
# Set environment variable for transformers
os.environ['TRANSFORMERS_CACHE'] = get_settings().MODEL_CACHE_DIR
