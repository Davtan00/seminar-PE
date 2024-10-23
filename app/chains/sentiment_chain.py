from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from app.prompts.sentiment_prompts import create_sentiment_prompt, create_bad_prompt
from app.utils.cost_tracker import track_cost
from app.config import get_settings
from fastapi import HTTPException
import json
import uuid
from typing import Dict, Any
from openai import OpenAIError

class SentimentAnalysisChain:
    def __init__(self):
        settings = get_settings()
        try:
            self.llm = ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo", # Using 3.5 for cost efficiency since this is a personal key
                request_timeout=30,
                openai_api_key=settings.OPENAI_API_KEY
            )
        except Exception as e:
            print(f"Error initializing ChatOpenAI: {e}")
            raise
        
    async def analyze(self, text: str, domain: str, use_bad_prompt: bool = False, index: int = 0) -> Dict[str, Any]:
        try:
            prompt = create_bad_prompt() if use_bad_prompt else create_sentiment_prompt(domain)
            chain = (
                {"text": RunnablePassthrough(), "domain": lambda _: domain} 
                | prompt 
                | self.llm 
                | JsonOutputParser()
            )
            
            result = await chain.ainvoke(text)
            
            return {
                "id": index + 1,  # Simple enumeration
                "text": text,
                **result
            }
        except OpenAIError as e:
            raise HTTPException(status_code=500, detail=f"OpenAI API Error: {str(e)}")
        except json.JSONDecodeError as e:
            return {
                "id": index + 1,
                "text": text,
                "error": "Failed to get structured output",
                "raw_response": str(e)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error during analysis: {str(e)}")
