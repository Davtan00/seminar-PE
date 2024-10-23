from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

SYSTEM_TEMPLATE = """You are an expert sentiment analysis model specializing in {domain} feedback.
Your task is to analyze text and provide detailed sentiment classification with metadata.
Always maintain consistent output format and provide confidence scores based on certainty."""

HUMAN_TEMPLATE = """Analyze the following text and provide sentiment analysis with metadata:
Text: {text}

Return the analysis in valid JSON format with:
- sentiment (positive/neutral/negative)
- confidence_score (0.0-1.0)
- metadata including domain, sub_category, key_aspects, and text_stats"""

def create_sentiment_prompt(domain: str) -> ChatPromptTemplate:
    system_message_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
    human_message_prompt = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
    
    return ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])

# Add these after the existing prompts
BASIC_SYSTEM_TEMPLATE = """You are a sentiment analysis model."""

BASIC_HUMAN_TEMPLATE = """What is the sentiment of this text? Text: {text}
Return the result in JSON format."""

def create_basic_prompt(domain: str) -> ChatPromptTemplate:
    system_message_prompt = SystemMessagePromptTemplate.from_template(BASIC_SYSTEM_TEMPLATE)
    human_message_prompt = HumanMessagePromptTemplate.from_template(BASIC_HUMAN_TEMPLATE)
    
    return ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])


BAD_TEMPLATE = """Clean up this data and return it as JSON for sentiment analysis:
{text}"""

def create_bad_prompt() -> ChatPromptTemplate:
    # No system message, just a basic human prompt - like most non-engineers would do
    return ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(BAD_TEMPLATE)
    ])
