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

SIMPLE_SYSTEM_TEMPLATE = """You are an expert in generating realistic {domain} feedback data.
Your task is to create diverse, balanced, and authentic-looking sentiment data.
CRITICAL REQUIREMENTS:
1. Return ONLY valid JSON array with EXACTLY {count} records
2. Each record MUST be complete and valid JSON
3. Use ONLY double quotes (") for strings, never single quotes (')
4. Generate ALL {count} records in a single response
5. You MUST STRICTLY follow these sentiment ratios:
{sentiment_distribution}

IMPORTANT: Your response MUST contain EXACTLY {count} records, no more, no less."""

SIMPLE_HUMAN_TEMPLATE = """Generate EXACTLY {count} records for {domain} sentiment analysis with this EXACT distribution:
{sentiment_distribution_details}

Required counts:
{distribution_examples}

Each record must ONLY have:
- text: a realistic and varied feedback text
- sentiment: EXACTLY one of (positive/neutral/negative) matching the required distribution"""


SIMPLE_BASIC_TEMPLATE = """You are a specialized AI trained to generate large quantities of realistic {domain} reviews efficiently.

Your core objectives:
1. Generate EXACTLY {count} reviews
2. Maintain high quality while prioritizing speed
3. Focus on realistic, diverse content
4. Keep responses concise but authentic
5. Create a natural balance of positive, neutral, and negative experiences

Guidelines for generation:
- Aim for roughly equal distribution of positive, neutral, and negative reviews
- Vary the length naturally (short, medium, and long reviews)
- Use authentic language and expressions
- Include specific details about {domain}
- Avoid repetitive patterns or phrases
- Consider common complaints and praise points for {domain}
- Include both emotional and factual reviews
- Mix different aspects (price, quality, service, etc.)

Output Requirements:
1. Return ONLY a valid JSON array
2. Each object must have EXACTLY one field: "text"
3. Use double quotes (") for strings
4. Generate ALL {count} reviews in one response
5. No additional formatting or comments

Example format:
[
    {{"text": "Excellent service and great value for money. Would definitely recommend!"}},
    {{"text": "Average experience. Nothing special but nothing terrible either."}},
    {{"text": "Disappointing quality and overpriced. Would not purchase again."}}
]"""

SIMPLE_BASIC_HUMAN_TEMPLATE = """Create {count} diverse {domain} reviews.

Focus areas:
1. Quantity: EXACTLY {count} reviews
2. Balance: Mix of positive, neutral, and negative experiences
3. Authenticity: Real-world language and scenarios
4. Diversity: Varied perspectives and aspects
5. Specificity: {domain}-relevant details

Remember: Only the "text" field is needed."""

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

def create_simple_generation_prompt(
    domain: str,
    count: int,
    sentiment_distribution: Optional[Dict[str, float]] = None,
) -> ChatPromptTemplate:
    # Reuse the distribution calculation logic from create_generation_prompt
    if sentiment_distribution:
        sentiment_counts = {
            sentiment: int(ratio * count)
            for sentiment, ratio in sentiment_distribution.items()
        }
        dist_details = "\n".join([
            f"- EXACTLY {sentiment_counts[sentiment]} {sentiment} records ({ratio*100:.1f}%)"
            for sentiment, ratio in sentiment_distribution.items()
        ])
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
        SystemMessagePromptTemplate.from_template(SIMPLE_SYSTEM_TEMPLATE),
        HumanMessagePromptTemplate.from_template(SIMPLE_HUMAN_TEMPLATE)
    ]

    return ChatPromptTemplate.from_messages(messages).partial(
        domain=domain,
        count=count,
        sentiment_distribution=dist_str,
        sentiment_distribution_details=dist_details,
        distribution_examples=examples
    )

def create_simple_generation_prompt_basic(
    domain: str,
    count: int,
) -> ChatPromptTemplate:
    messages = [
        SystemMessagePromptTemplate.from_template(SIMPLE_BASIC_TEMPLATE),
        HumanMessagePromptTemplate.from_template(SIMPLE_BASIC_HUMAN_TEMPLATE)
    ]

    return ChatPromptTemplate.from_messages(messages).partial(
        domain=domain,
        count=count
    )
