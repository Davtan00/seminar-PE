from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from app.prompts.sentiment_prompts import create_sentiment_prompt
from app.utils.cost_tracker import track_cost
from app.config import get_settings
import json
import uuid
from typing import Dict, Any

class SentimentAnalysisChain:
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-3.5-turbo",  # Using 3.5 for cost efficiency since this is my own api key
            request_timeout=30,
            openai_api_key=settings.OPENAI_API_KEY
        )
        
    async def analyze(self, text: str, domain: str) -> Dict[str, Any]:
        prompt = create_sentiment_prompt(domain)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        
        with track_cost() as cost:
            result = await chain.arun(text=text, domain=domain)
            
        try:
            analysis = json.loads(result)
            return {
                "id": f"rev_{str(uuid.uuid4())[:8]}",
                "text": text,
                **analysis,
                "cost_info": cost
            }
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response")
