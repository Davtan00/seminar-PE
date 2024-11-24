import os
import shutil
from fastapi import APIRouter
from pathlib import Path
from app.config import get_settings

## Store the models and put the directory pathin the .env MODEL_CACHE_DIR

router = APIRouter(
    prefix="/cache",
    tags=["cache-management"],
    responses={404: {"description": "Not found"}}
)

class ModelCacheManager:
    def __init__(self):
        self.cache_dir = Path(get_settings().MODEL_CACHE_DIR)
        
    def clear_cache(self):
        """Clear the entire model cache"""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir)
            
    def get_cache_size(self):
        """Get total size of cached models"""
        total = 0
        for path in self.cache_dir.rglob('*'):
            if path.is_file():
                total += path.stat().st_size
        return total / (1024 * 1024)  # Return size in MB
        
    def list_cached_models(self):
        """List all cached models"""
        if not self.cache_dir.exists():
            return []
        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]

# Cache management endpoints
@router.get("/info")
async def get_cache_info():
    cache_manager = ModelCacheManager()
    return {
        "cache_directory": str(cache_manager.cache_dir),
        "cache_size_mb": round(cache_manager.get_cache_size(), 2),
        "cached_models": cache_manager.list_cached_models()
    }

@router.post("/clear")
async def clear_cache():
    cache_manager = ModelCacheManager()
    cache_manager.clear_cache()
    return {"message": "Cache cleared successfully"} 