from typing import List, Dict, Optional,Any
import requests
import asyncio
from app.config import get_settings

class HFGenerationChain:
    def __init__(self):
        settings = get_settings()
        self.api_url = "https://api-inference.huggingface.co/models/gpt2"
        self.headers = {"Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}"}

    async def generate(
        self,
        domain: str,
        count: int,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        prompt_template = f"""Generate a detailed and realistic review about {domain}. 
        The review should be specific and include both factual and emotional aspects."""

        async def generate_batch(batch_count: int) -> List[Dict[str, str]]:
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json={
                        "inputs": prompt_template,
                        "parameters": {
                            "max_length": 150,
                            "num_return_sequences": batch_count,
                            "temperature": 0.9,
                            "do_sample": True,
                            "return_full_text": False
                        }
                    }
                )
                
                if response.status_code == 503:
                    await asyncio.sleep(20)
                    return await generate_batch(batch_count)
                
                response.raise_for_status()
                generated_texts = response.json()
                
                return [{"text": text} for text in generated_texts]
                
            except Exception as e:
                print(f"Error in batch generation: {str(e)}")
                return []

        # Process in batches
        results = []
        current_id = 1
        
        for i in range(0, count, batch_size):
            batch_count = min(batch_size, count - i)
            batch_results = await generate_batch(batch_count)
            
            for result in batch_results:
                results.append({"id": current_id, **result})
                current_id += 1
            
            await asyncio.sleep(1)

        return {
            "generated_data": results,
            "summary": {
                "total_generated": len(results),
                "domain": domain
            }
        } 