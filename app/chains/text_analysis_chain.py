from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from app.prompts.text_analysis_prompts import create_text_analysis_prompt
from app.config import get_settings
from fastapi import HTTPException
from typing import Dict, Any
from openai import OpenAIError

class AnalysisResult(BaseModel):
    text: str = Field(description="The original text")
    sentiment: str = Field(description="One of: positive, neutral, negative")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class TextAnalysisChain:
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            temperature=0.1,  
            model="gpt-4o-mini",  # Good price/performance model choice with 128k context window
            request_timeout=1000,  # Increased timeout
            max_retries=5,  # Increased max retries
            max_tokens=16384,  
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        self.parser = JsonOutputParser(pydantic_object=AnalysisResult)
        self.prompt = create_text_analysis_prompt()

    async def analyze_text(self, text: str, index: int = 0) -> Dict[str, Any]:
        try:
            chain = (
                {"text": RunnablePassthrough()}
                | self.prompt
                | self.llm
                | self.parser
            )
            
            result_dict = await chain.ainvoke(text)
            
            return {
                "id": index + 1,
                "text": text,
                "sentiment": result_dict["sentiment"],
                "confidence": result_dict["confidence"]
            }
            
        except OpenAIError as e:
            raise HTTPException(
                status_code=500,
                detail=f"OpenAI API error: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing text: {str(e)}"
            )
