# Seminar-PE

# Sentiment Analysis and Data Generation API

A FastAPI-based service for sentiment analysis and synthetic data generation

## Features

- Sentiment Analysis with metadata and confidence scores
- Multiple data generation approaches (full metadata, simple sentiment, basic reviews)
- Cost tracking for OpenAI API usage
- Configurable batch processing
- API key authentication
- Rate limit handling
- Extensive error handling and validation

I didnt fully grasp what exactly the input would be either generate or receive data , cleanse and then return in a nice format, so I simply did both variants

## API Endpoints

### Sentiment Analysis

- **POST `/analyze`**
  - Analyzes sentiment of multiple texts with detailed metadata
  - Includes confidence scores and domain-specific insights
  - Batch processing support

- **POST `/clean-and-analyze`**
  - Processes and analyzes raw text data
  - Provides sentiment distribution statistics
  - Returns detailed metadata and confidence scores

### Data Generation

- **POST `/generate-data`**
  - Generates synthetic domain-specific data with full metadata
  - Configurable sentiment distribution
  - Supports verbose mode for detailed metadata
  - Example request:
  ```json
  {
    "domain": "Restaurants",
    "count": 50,
    "sentiment_distribution": {
      "positive": 0.33,
      "neutral": 0.33,
      "negative": 0.34
    },
    "verbose": true
  }
  ```

- **POST `/generate-simple`**
  - Generates basic review data without sentiment analysis
  - Optimized for high-volume generation
  - Example request:
  ```json
  {
    "domain": "Shoes",
    "count": 100
  }
  ```

- **POST `/bad-generate-data`**
  - Demonstration endpoint for basic generation
  - Shows impact of prompt engineering on output quality

### Utility

- **GET `/health`**
  - Health check endpoint
  - Returns service status

## Technical Details

### Models
- Uses gpt-4o-mini for optimal performance
- Configurable token limits and batch sizes
- Automatic retry mechanisms for API failures

### Authentication
All endpoints (except `/health`) require an API key: