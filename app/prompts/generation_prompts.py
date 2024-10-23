from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

SYSTEM_TEMPLATE = """You are an expert in generating realistic {domain} feedback data.
Your task is to create diverse, balanced, and authentic-looking sentiment data that mirrors real-world patterns."""

HUMAN_TEMPLATE = """Generate {count} unique records for sentiment analysis in the {domain} domain.

Each record should be realistic and varied in:
- Length and complexity
- Vocabulary and expression style
- Sentiment distribution (maintain natural proportions)
- Common issues and praises in the {domain} field

Return the data as a JSON array where each item must have:
- text: the generated review/feedback text
- sentiment: (positive/neutral/negative)
- confidence_score: (0.0-1.0)
- metadata: {{
    "domain": "{domain}",
    "sub_category": "<relevant subcategory>",
    "key_aspects": ["<aspect1>", "<aspect2>"],
    "text_stats": {{
        "word_count": 0,
        "character_count": 0,
        "average_word_length": 0.0
    }}
}}"""

BAD_TEMPLATE = """Give me {count} rows of {domain} reviews for sentiment analysis."""

def create_generation_prompt(domain: str, count: int) -> ChatPromptTemplate:
    system_message = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
    human_message = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
    return ChatPromptTemplate.from_messages([system_message, human_message])

def create_bad_generation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(BAD_TEMPLATE)
    ])
