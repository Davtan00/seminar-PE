from dataclasses import dataclass
@dataclass
class ReviewParameters:
    realism: float = 0.7
    domainRelevance: float = 0.8
    formality: float = 0.5
    lexicalComplexity: float = 0.5
    culturalSensitivity: float = 0.8
    privacyLevel: float = 0.8
    biasControl: float = 0.7
    sentimentIntensity: float = 0.5
    diversity: float = 0.6
    temporalRelevance: float = 0.5
    noiseLevel: float = 0.3

    def get_prompt_description(self, use_compact: bool = True) -> str:
        """Generate parameter description based on compact flag"""
        if use_compact:
            return (
                f"Style parameters: "
                f"realism={self.realism:.1f}, "
                f"relevance={self.domainRelevance:.1f}, "
                f"formality={self.formality:.1f}, "
                f"complexity={self.lexicalComplexity:.1f}, "
                f"sentiment_strength={self.sentimentIntensity:.1f}, "
                f"privacy={self.privacyLevel:.1f}"
            )