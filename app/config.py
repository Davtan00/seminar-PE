from pydantic_settings import BaseSettings 
from functools import lru_cache
from cryptography.fernet import Fernet
from hmac import HMAC, compare_digest
import hashlib
import base64
import json
from typing import Dict, Any

class SecurityManager:
    def __init__(self, encryption_key: str, signing_secret: str):
        self.encryption_key = encryption_key
        self.signing_secret = signing_secret
        
        if not self.encryption_key or not self.signing_secret:
            raise ValueError("Encryption key and signing secret must be set")

    def decrypt_api_key(self, encrypted_key: str) -> str:
        try:
            fernet_key = base64.urlsafe_b64encode(
                hashlib.sha256(self.encryption_key.encode()).digest()
            )
            cipher = Fernet(fernet_key)
            return cipher.decrypt(encrypted_key.encode()).decode()
        except Exception as e:
            raise ValueError(f"Failed to decrypt API key: {str(e)}")

    def verify_signature(self, timestamp: int, payload: Dict[str, Any], signature: str) -> bool:
        string_to_sign = f"{timestamp}.{json.dumps(payload)}"
        print(f"Backend string to sign: {string_to_sign}")  # For debugging
        
        expected_signature = base64.b64encode(
            HMAC(
                key=self.signing_secret.encode(),
                msg=string_to_sign.encode(),
                digestmod=hashlib.sha256
            ).digest()
        ).decode()
        
        print(f"Expected signature: {expected_signature}")  # For debugging
        print(f"Received signature: {signature}")  # For debugging
        
        return compare_digest(signature, expected_signature)

class Settings(BaseSettings):
    PROJECT_NAME: str = "Prompt Engineering Seminar"
    PROJECT_VERSION: str = "1.0.0"
    OPENAI_API_KEY: str
    API_KEY: str
    ENCRYPTION_KEY: str
    SIGNING_SECRET: str
    
    class Config:
        env_file = ".env"
        extra = "allow"

@lru_cache()
def get_settings():
    return Settings()

@lru_cache()
def get_security_manager():
    settings = get_settings()
    return SecurityManager(settings.ENCRYPTION_KEY, settings.SIGNING_SECRET)
