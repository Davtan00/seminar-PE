from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

TEXT_ANALYSIS_SYSTEM_TEMPLATE = """You are an expert sentiment analysis model.
Your task is to analyze text and classify it into one of three categories: positive, neutral, or negative.
CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON without any markdown formatting or comments
2. Each analysis must include EXACTLY one sentiment label and a confidence score
3. Sentiment MUST be one of: positive, neutral, negative
4. Confidence score MUST be a number between 0.0 and 1.0
5. Be consistent in your classifications"""

TEXT_ANALYSIS_HUMAN_TEMPLATE = """Analyze the sentiment of this text:
"{text}"

Return ONLY a JSON object with these EXACT fields:
{{
    "sentiment": "one of [positive, neutral, negative]",
    "confidence": number between 0.0 and 1.0
}}"""

def create_text_analysis_prompt() -> ChatPromptTemplate:
    messages = [
        SystemMessagePromptTemplate.from_template(TEXT_ANALYSIS_SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(TEXT_ANALYSIS_HUMAN_TEMPLATE)
    ]
    
    return ChatPromptTemplate.from_messages(messages)
