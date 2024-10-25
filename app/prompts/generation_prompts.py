from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import Optional, Dict

SYSTEM_TEMPLATE = """You are an expert in generating realistic {domain} feedback data.
Your task is to create diverse, balanced, and authentic-looking sentiment data that mirrors real-world patterns.
CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON array without any markdown formatting or comments
2. You MUST generate EXACTLY {count} records
3. You MUST STRICTLY follow these sentiment ratios:
{sentiment_distribution}"""

HUMAN_TEMPLATE = """Generate EXACTLY {count} records for {domain} sentiment analysis with this EXACT distribution:
{sentiment_distribution_details}

Required counts:
{distribution_examples}

Each record must have:
- text: the generated review/feedback text
- sentiment: EXACTLY one of (positive/neutral/negative) matching the required distribution
- confidence_score: (0.0-1.0)
- metadata: {{
    "domain": "{domain}",
    "sub_category": "<relevant subcategory>",
    "key_aspects": ["<aspect1>", "<aspect2>"],
    "text_stats": {{
        "word_count": <integer>,
        "character_count": <integer>,
        "average_word_length": <float>
    }}
}}"""

BAD_TEMPLATE = """Give me {count} rows of {domain} reviews for sentiment analysis."""

def create_generation_prompt(
    domain: str,
    count: int,
    sentiment_distribution: Optional[Dict[str, float]] = None,
) -> ChatPromptTemplate:
    if sentiment_distribution:
        # Calculate exact numbers for each sentiment
        sentiment_counts = {
            sentiment: int(ratio * count)
            for sentiment, ratio in sentiment_distribution.items()
        }
        
        # Format detailed distribution instructions
        dist_details = "\n".join([
            f"- EXACTLY {sentiment_counts[sentiment]} {sentiment} records ({ratio*100:.1f}%)"
            for sentiment, ratio in sentiment_distribution.items()
        ])
        
        # Add concrete examples
        examples = "\n".join([
            f"- {sentiment}: {sentiment_counts[sentiment]} records"
            for sentiment in sentiment_counts
        ])
        
        dist_str = f"Strict distribution requirements:\n{dist_details}"
    else:
        dist_str = "Maintain a natural distribution across positive, neutral, and negative sentiments"
        dist_details = "Natural distribution across sentiments"
        examples = "Balanced distribution of sentiments"

    messages = [
        SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
    ]

    return ChatPromptTemplate.from_messages(messages).partial(
        domain=domain,
        count=count,
        sentiment_distribution=dist_str,
        sentiment_distribution_details=dist_details,
        distribution_examples=examples
    )

def create_bad_generation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template(BAD_TEMPLATE)
    ])
