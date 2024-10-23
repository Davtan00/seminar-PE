from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from app.prompts.generation_prompts import create_generation_prompt, create_bad_generation_prompt
from app.config import get_settings
from fastapi import HTTPException
from typing import Dict, Any, List
from openai import OpenAIError

MAX_RECORDS = 1000  # Safety limit since still not project API key

class DataGenerationChain:
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            temperature=0.7,   # Higher temperature for more variety, to make the results a bit more interesting
            model="gpt-3.5-turbo",
            request_timeout=60,
            openai_api_key=settings.OPENAI_API_KEY
        )
    
    async def generate(self, domain: str, count: int, use_bad_prompt: bool = False) -> List[Dict[str, Any]]:
        if count > MAX_RECORDS:
            raise ValueError(f"Requested count exceeds maximum limit of {MAX_RECORDS}")
            
        try:
            prompt = create_bad_generation_prompt() if use_bad_prompt else create_generation_prompt(domain, count)
            chain = (
                {"domain": RunnablePassthrough(), "count": lambda x: count}
                | prompt
                | self.llm
                | JsonOutputParser()
            )
            
            results = await chain.ainvoke(domain)
            return [{"id": i + 1, **item} for i, item in enumerate(results)]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
