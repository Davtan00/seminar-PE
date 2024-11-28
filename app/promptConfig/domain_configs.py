DOMAIN_CONFIGS = {
    'restaurant': {
        'review_type': 'dining experience',
        'aspects': ['food quality', 'service', 'ambiance', 'price-to-value', 'cleanliness'],
        'context_prefix': 'about this restaurant',
        'specific_prompts': {
            'high_relevance': 'Include specific details about the food, service quality, or dining atmosphere',
            'low_relevance': 'Focus on the overall experience'
        }
    },
    'hotel': {
        'review_type': 'stay experience',
        'aspects': ['room quality', 'service', 'amenities', 'location', 'value', 'cleanliness', 'comfort'],
        'context_prefix': 'about this hotel',
        'specific_prompts': {
            'high_relevance': 'Include specific details about the accommodation, services, and facilities',
            'low_relevance': 'Focus on overall stay satisfaction'
        }
    },
    'software': {
        'review_type': 'software experience',
        'aspects': ['functionality', 'user interface', 'performance', 'reliability', 'support', 'updates', 'integration'],
        'context_prefix': 'about this software',
        'specific_prompts': {
            'high_relevance': 'Include specific features, technical aspects, or functionality details',
            'low_relevance': 'Focus on general software usability'
        }
    },
    'ecommerce': {
        'review_type': 'shopping experience',
        'aspects': ['website usability', 'product selection', 'checkout process', 'delivery speed', 'customer service', 'return policy'],
        'context_prefix': 'about this online store',
        'specific_prompts': {
            'high_relevance': 'Include specific details about the shopping process, delivery, or customer service',
            'low_relevance': 'Focus on overall shopping satisfaction'
        }
    },
    'social_media': {
        'review_type': 'platform experience',
        'aspects': ['user interface', 'features', 'content quality', 'community interaction', 'privacy settings', 'app performance'],
        'context_prefix': 'about this social media platform',
        'specific_prompts': {
            'high_relevance': 'Include specific platform features, community aspects, or user experience details',
            'low_relevance': 'Focus on overall platform satisfaction'
        }
    },
    'technology': {
        'review_type': 'tech product experience',
        'aspects': ['performance', 'build quality', 'features', 'innovation', 'ease of use', 'compatibility', 'value'],
        'context_prefix': 'about this tech product',
        'specific_prompts': {
            'high_relevance': 'Include specific technical features, performance metrics, or usage scenarios',
            'low_relevance': 'Focus on overall product satisfaction'
        }
    },
    'education': {
        'review_type': 'learning experience',
        'aspects': ['course content', 'teaching quality', 'learning resources', 'support services', 'engagement', 'practical value'],
        'context_prefix': 'about this educational service',
        'specific_prompts': {
            'high_relevance': 'Include specific details about the learning materials, teaching methods, or educational outcomes',
            'low_relevance': 'Focus on overall learning satisfaction'
        }
    },
    'healthcare': {
        'review_type': 'healthcare experience',
        'aspects': ['medical care quality', 'staff professionalism', 'facility cleanliness', 'wait times', 'communication', 'follow-up care'],
        'context_prefix': 'about this healthcare service',
        'specific_prompts': {
            'high_relevance': 'Include specific details about the medical care, staff interaction, or facility conditions',
            'low_relevance': 'Focus on overall care satisfaction'
        }
    }
} 

