from pydantic import BaseModel
from typing import List, Dict, Optional, Any

class HuggingFaceRequest(BaseModel):
    generated_data: List[Dict[str, Any]]
    batch_size: Optional[int] = 16
    stream_progress: Optional[bool] = False
    save_progress: Optional[bool] = False
    summary: Optional[Dict[str, Any]] = None

class LargeAnalysisRequest(HuggingFaceRequest):
    batch_id: Optional[str] = None
    save_progress: bool = True
    stream_response: bool = False
    stream_progress: bool = True
    start_index: Optional[int] = 0 