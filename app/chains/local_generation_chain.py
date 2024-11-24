from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Optional, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.config import get_settings

class LocalGenerationChain:
    def __init__(self):
        ## TODO change this gargbage model 
        self.model_name = "facebook/opt-350m"
        self.cache_dir = get_settings().MODEL_CACHE_DIR
        
        print(f"Loading model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir,
            use_fast=True  # Use faster tokenizer implementation
        )
        
        # Load model with CPU optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better CPU performance
            low_cpu_mem_usage=True       # Optimize memory usage
        )
        
        # Enable CPU threading optimizations
        torch.set_num_threads(8)  # Set to number of physical cores
        torch.set_num_interop_threads(8)  # Set to number of physical cores
        print(f"Model loaded with CPU optimizations")

    async def generate(
        self,
        domain: str,
        count: int,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        prompt_template = f"""Write a short, specific customer review for a {domain}. Focus on personal experience, quality, and service. Be genuine and concise.

Review: """
        
        async def generate_single_review() -> Dict[str, str]:
            inputs = self.tokenizer(
                prompt_template, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            with torch.inference_mode():  
                outputs = self.model.generate(
                    **inputs,
                    max_length=100,            
                    min_length=30,             
                    num_return_sequences=1,
                    temperature=0.7,           
                    top_p=0.9,                
                    do_sample=True,
                    no_repeat_ngram_size=3,    
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.2,    # Discourage repetitive text
                    early_stopping=True
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            review_text = generated_text.replace(prompt_template, "").strip()
            
            # Clean up any incomplete sentences
            if review_text.count('.') > 0:
                review_text = '. '.join(review_text.split('.')[:-1]) + '.'
            
            return {
                "text": review_text
            }

        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(max_workers, 8)) as executor:  # Limit to physical cores
            results = []
            loop = asyncio.get_event_loop()
            
            # Create batches of tasks for better CPU utilization
            tasks = [
                loop.run_in_executor(executor, lambda: asyncio.run(generate_single_review()))
                for _ in range(count)
            ]
            
            # Wait for all tasks to complete
            completed_tasks = await asyncio.gather(*tasks)
            
            for i, result in enumerate(completed_tasks):
                results.append({"id": i + 1, **result})
        
        return {
            "generated_data": results,
            "summary": {
                "total_generated": len(results),
                "domain": domain,
                "model": self.model_name
            }
        } 