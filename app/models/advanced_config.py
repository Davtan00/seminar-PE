from pydantic import BaseModel, Field
from typing import Optional

class SentimentDistribution(BaseModel):
    positive: int = Field(
        ge=0, 
        le=100, 
        description="Percentage of positive reviews (0-100)"
    )
    negative: int = Field(
        ge=0, 
        le=100, 
        description="Percentage of negative reviews (0-100)"
    )
    neutral: int = Field(
        ge=0, 
        le=100, 
        description="Percentage of neutral reviews (0-100)"
    )

    def validate_total(self):
        total = self.positive + self.negative + self.neutral
        if total != 100:
            raise ValueError("Sentiment distribution percentages must sum to 100")

class AdvancedConfig(BaseModel):
    # Basic configuration
    sentimentDistribution: SentimentDistribution
    rowCount: int = Field(
        gt=0, 
        le=100000, 
        description="Number of reviews to generate"
    )
    domain: str = Field(
        description="Domain/topic for the generated content"
    )

    # Model parameters 
    model: str = Field(
        default="gpt-4o-mini",
        description="Model to use for generation"
    )
    temperature: float = Field(
        ge=0, 
        le=1, 
        default=0.7,
        description="Controls randomness in generation (0-1)"
    )
    topP: float = Field(
        ge=0, 
        le=1, 
        default=0.5,
        description="Controls diversity via nucleus sampling (0-1)"
    )
    maxTokens: int = Field(
        ge=1, 
        le=2000, 
        default=100,
        description="Maximum tokens to generate"
    )
    frequencyPenalty: float = Field(
        ge=0, 
        le=2, 
        default=0,
        description="Penalizes frequent token usage (0-2)"
    )
    presencePenalty: float = Field(
        ge=0, 
        le=2, 
        default=0,
        description="Penalizes token presence (0-2)"
    )

    # Content control parameters
    privacyLevel: float = Field(
        ge=0, 
        le=1, 
        default=0.5,
        description="Controls inclusion of personal/sensitive information (0-1)"
    )
    biasControl: float = Field(
        ge=0, 
        le=1, 
        default=0.5,
        description="Controls potential bias in generated content (0-1)"
    )
    sentimentIntensity: float = Field(
        ge=0, 
        le=1, 
        default=0.5,
        description="Controls strength of sentiment expression (0-1)"
    )

    # Quality parameters
    realism: float = Field(
        ge=0, 
        le=1, 
        default=0.7,
        description="Controls how realistic the content should be (0-1)"
    )
    domainRelevance: float = Field(
        ge=0, 
        le=1, 
        default=0.8,
        description="Controls relevance to specified domain (0-1)"
    )
    diversity: float = Field(
        ge=0, 
        le=1, 
        default=0.6,
        description="Controls variety in generated content (0-1)"
    )
    temporalRelevance: float = Field(
        ge=0, 
        le=1, 
        default=0.5,
        description="Controls temporal context awareness (0-1)"
    )
    noiseLevel: float = Field(
        ge=0, 
        le=1, 
        default=0.3,
        description="Controls introduction of natural language noise (0-1)"
    )

    # Style parameters
    culturalSensitivity: float = Field(
        ge=0, 
        le=1, 
        default=0.8,
        description="Controls cultural awareness and sensitivity (0-1)"
    )
    formality: float = Field(
        ge=0, 
        le=1, 
        default=0.5,
        description="Controls formal vs informal tone (0-1)"
    )
    lexicalComplexity: float = Field(
        ge=0, 
        le=1, 
        default=0.5,
        description="Controls vocabulary complexity (0-1)"
    )

class AdvancedGenerationRequest(BaseModel):
    encryptedKey: str = Field(
        description="Base64 encoded OpenAI API key"
    )
    config: AdvancedConfig