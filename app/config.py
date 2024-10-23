from pydantic_settings import BaseSettings 
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "Prompt Engineering Seminar"
    PROJECT_VERSION: str = "1.0.0"
    OPENAI_API_KEY: str
    API_KEY: str  
    
    class Config:
        env_file = ".env"
        extra = "allow"  

@lru_cache()
def get_settings():
    return Settings()
