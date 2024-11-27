import base64
## If we actually deploy onto hosted services use better encryption and signatures 
def decrypt_api_key(encrypted_key: str) -> str:
    """Simple Base64 decoding for development purposes"""
    try:
        return base64.b64decode(encrypted_key).decode('utf-8')
    except:
        raise ValueError("Invalid encrypted key format") 