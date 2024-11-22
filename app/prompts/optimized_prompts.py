from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from typing import Dict

OPTIMIZED_GENERATION_TEMPLATE = """You are an expert in generating {domain} feedback data.

STYLE CONFIGURATION (all parameters 0.0-1.0):
- Realism: {realism} (authenticity of scenarios)
- Domain Relevance: {domain_relevance} (industry-specific terminology)
- Cultural Sensitivity: {cultural_sensitivity} (awareness of cultural contexts)
- Formality: {formality} (professional vs casual language)
- Lexical Complexity: {lexical_complexity} (vocabulary sophistication)
- Diversity: {diversity} (variety in perspectives/experiences)
- Privacy Level: {privacy_level} (personal detail abstraction)

CONTENT GUIDELINES:
High Realism + Domain Relevance Example:
"The ML model's inference speed improved by 40% after optimization, though we noticed increased memory usage on edge devices. ROI exceeded expectations."

High Cultural Sensitivity + Formality Example:
"The platform's inclusive design and professional support team made our international collaboration seamless. Documentation is well-structured."

High Diversity + Lexical Complexity Example:
"The implementation exhibited remarkable versatility across heterogeneous environments, although the learning curve was initially steep."

REQUIREMENTS:
1. Generate EXACTLY {count} records
2. Return ONLY a JSON array of objects with 'text' field
3. Adapt content based on ALL style parameters
4. Maintain natural language patterns
5. NO additional metadata

Expected format:
[{{"text": "your text here"}}, ...]"""

OPTIMIZED_ANALYSIS_TEMPLATE = """You are an expert sentiment analyzer for {domain} content.

ANALYSIS PARAMETERS:
- Sentiment Intensity: {sentiment_intensity} (0-1, higher = stronger classification)
- Bias Control: {bias_control} (0-1, higher = more neutral interpretation)
- Temporal Relevance: {temporal_relevance} (0-1, higher = more weight on current context)
- Noise Level: {noise_level} (0-1, higher = more tolerance for ambiguity)

TARGET DISTRIBUTION:
Positive: {positive}%, Neutral: {neutral}%, Negative: {negative}%

CALIBRATION EXAMPLES:
[High Intensity, Low Noise]
Text: "This solution completely transformed our workflow!"
→ Positive (0.95)

[High Bias Control, Medium Intensity]
Text: "Performance metrics showed 15% improvement with some tradeoffs"
→ Neutral (0.80)

[High Temporal Relevance]
Text: "Latest update broke compatibility with legacy systems"
→ Negative (0.85)

ANALYZE THIS TEXT:
{text}

RETURN ONLY:
{
    "sentiment": "positive|neutral|negative",
    "confidence": float (0.0-1.0)
}"""

def create_optimized_prompts(config: Dict) -> tuple[ChatPromptTemplate, ChatPromptTemplate]:
    generation_messages = [
        SystemMessagePromptTemplate.from_template(OPTIMIZED_GENERATION_TEMPLATE)
    ]
    
    analysis_messages = [
        SystemMessagePromptTemplate.from_template(OPTIMIZED_ANALYSIS_TEMPLATE)
    ]
    
    return (
        ChatPromptTemplate.from_messages(generation_messages),
        ChatPromptTemplate.from_messages(analysis_messages)
    ) 