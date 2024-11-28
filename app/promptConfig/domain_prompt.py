from typing import Dict, Any
from app.promptConfig.domain_configs import DOMAIN_CONFIGS
from app.classes.models import ReviewParameters
import logging

logger = logging.getLogger(__name__)

SENTIMENT_EXAMPLES = {
    'restaurant': {
        'positive': "The wagyu steak was cooked to perfection, and the attentive sommelier recommended an excellent wine pairing. The ambient lighting and soft jazz created an intimate atmosphere perfect for our anniversary dinner.",
        'negative': "Despite the high prices, the portion sizes were disappointingly small, and we had to wait over an hour for our main course to arrive. The food was barely lukewarm when served.",
        'neutral': "The restaurant offers standard fare at typical prices. Service was neither particularly good nor bad, and the atmosphere was acceptable for a casual dining experience."
    },
        'technology': {
            'positive': "The processor handles multiple demanding applications smoothly, and the cooling system keeps everything running at optimal temperatures. The build quality is exceptional for this price point.",
            'negative': "The battery drains completely within 4 hours of normal use, and the charging port became loose after just two weeks. The promised software updates never materialized.",
            'neutral': "The device performs basic functions as expected. Battery life is average, and the build quality is comparable to similar products in this price range."
        },
        'software': {
            'positive': "The cloud integration works flawlessly with our existing tools, and the automated backup feature has already saved us from several potential data losses. Customer support is incredibly responsive.",
            'negative': "The latest update completely overhauled the user interface without any warning. Critical features are now buried in submenus, and the new workflow is frustratingly counterintuitive.",
            'neutral': "The software provides the expected functionality without any standout features. The interface is functional, though not particularly innovative."
        },
        'healthcare': {
            'positive': "The medical staff was exceptionally thorough and caring. The doctor spent ample time explaining my condition and treatment options, and the facility was immaculately clean.",
            'negative': "Had to wait three months for an appointment, only to be rushed through a five-minute consultation. The doctor dismissed my concerns and barely looked at my test results.",
            'neutral': "The appointment process was straightforward, and wait times were reasonable. The staff was professional, providing standard care as expected."
        },
        'hotel': {
            'positive': "The room exceeded our expectations with its modern amenities and stunning view. The staff went above and beyond, especially the concierge who provided excellent local recommendations.",
            'negative': "The room had visible mold in the bathroom, and the air conditioning was extremely loud. Despite multiple complaints, the management failed to address any of our concerns.",
            'neutral': "The hotel provided basic accommodations at the expected level for its category. Check-in was efficient, and the room was clean but basic."
        },
        'education': {
            'positive': "The instructor's expertise and engaging teaching style made complex concepts easy to grasp. The online learning platform was intuitive, and the course materials were comprehensive.",
            'negative': "The course content was outdated, and the instructor was consistently unprepared. Technical issues frequently disrupted live sessions, and support was unresponsive.",
            'neutral': "The course covered the basic material as outlined in the syllabus. The teaching methods were conventional, and the resources provided were adequate."
        },
        'ecommerce': {
            'positive': "The website's search filters made finding exactly what I needed effortless. Checkout was smooth, and my order arrived ahead of schedule in perfect condition.",
            'negative': "The checkout process kept glitching, and customer service was unreachable. When my order finally arrived, it was completely different from what I ordered.",
            'neutral': "The ordering process was straightforward, and delivery occurred within the expected timeframe. Product selection and prices were typical for this type of online store."
        },
        'social_media': {
            'positive': "The new interface makes navigating between features seamless. The privacy controls are comprehensive, and the content recommendation algorithm is remarkably accurate.",
            'negative': "The platform is plagued with bugs after the latest update. Posts aren't showing up in chronological order, and the app crashes frequently during video playback.",
        'neutral': "The platform offers standard social networking features. Content sharing works as expected, and the user interface is functional though unremarkable."
    }
}

def create_review_prompt_detailed(
    domain: str,
    parameters: ReviewParameters,
    batch_params: dict,
    use_compact_prompt: bool = True
) -> str:
    """Create prompt with option for compact or detailed version"""
    
    if use_compact_prompt:
        return f"""Generate {batch_params['count']} {batch_params['sentiment']} {domain} reviews.
Parameters: {parameters.get_prompt_description(use_compact=True)}
Format: JSON with text and sentiment fields."""
    
    # Original detailed prompt
    domain_config = DOMAIN_CONFIGS.get(domain, {})
    current_sentiment = batch_params['sentiment']
    
    return f"""You are a specialized review generator creating authentic, domain-specific content.
Generate {batch_params['count']} {current_sentiment} reviews that strictly follow this JSON structure:

{{
  "reviews": [
    {{
      "text": "The actual review content",
      "sentiment": "{current_sentiment}"
    }}
  ]
}}

Domain Context:
You are generating {domain_config['review_type']}s {domain_config['context_prefix']}.
Key aspects to consider: {', '.join(domain_config['aspects'])}
{domain_config['specific_prompts']['high_relevance']}

Style Parameters:
{parameters.get_prompt_description(use_compact=False)}

Critical Requirements:
1. Generate exactly {batch_params['count']} reviews
2. Each review MUST have exactly "text" and "sentiment" fields
3. All reviews MUST maintain {current_sentiment} sentiment
4. Reviews must be complete sentences
5. Include domain-specific terminology
6. Reference the provided key aspects
7. Maintain the exact JSON structure shown above"""