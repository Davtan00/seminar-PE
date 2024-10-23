# Seminar-PE

# Sentiment Analysis and Data Generation API


## Features

- Sentiment Analysis with metadata and confidence scores
- Synthetic data generation for various domains
- Cost tracking for OpenAI API usage (Will be removed eventually)
- Demonstration of prompt engineering best practices(WIP)
- API key authentication

I didnt fully grasp what exactly the input would be either generate or receive data , cleanse and then return in a nice format, so I simply did both variants

## API Endpoints

### Sentiment Analysis

- **POST `/analyze`**
  - Analyzes sentiment of multiple texts with detailed metadata
  - Includes confidence scores and domain-specific insights

- **POST `/clean-and-analyze`**
  - Processes and analyzes raw text data
  - Provides sentiment distribution statistics
  - Returns detailed metadata and confidence scores

- **POST `/bad-prompt`**
  - Demonstrates poor prompt engineering practices
  - Shows the impact of basic prompts on analysis quality

### Data Generation

- **POST `/generate-data`**
  - Generates synthetic domain-specific data
  - Uses well-engineered prompts for realistic output
  - Includes sentiment and detailed metadata

- **POST `/bad-generate-data`**
  - Demonstrates basic data generation
  - Shows the difference in quality with simple prompts

### Utility

- **GET `/health`**
  - Health check endpoint
  - Returns service status

## Documentation

Interactive swagger API documentation is available at `/docs` when the server is running.

## Authentication

All endpoints (except `/health`) require an API key to be passed in the `x-api-key` header.

## Response Format

json

{
"id": 1,
"text": "Sample text",
"sentiment": "positive",
"confidence_score": 0.95,
"metadata": {
"domain": "example_domain",
"sub_category": "category",
"key_aspects": ["aspect1", "aspect2"],
"text_stats": {
"word_count": 10,
"character_count": 50,
"average_word_length": 5.0
        }
    }
}
## Cost Tracking

All endpoints include cost tracking for OpenAI API usage, returned in the response summary.