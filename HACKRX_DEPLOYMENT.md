# HackRx 6.0 PDF Chatbot API - Deployment Guide

## Quick Start for HackRx Competition

This API implements the exact endpoint specification required for HackRx 6.0 competition.

### Endpoint Specification

- **Method:** POST
- **Path:** `/hackrx/run`
- **Authentication:** Bearer Token
- **Content-Type:** application/json

### 1. Environment Setup

Create a `.env` file with your credentials:

```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional configurations
PINECONE_ENVIRONMENT=gcp-starter
HACKRX_API_KEY=hackrx_2024_secret_key
LOG_LEVEL=INFO
```

### 2. Installation & Running

#### Option A: Docker (Recommended)

```bash
# Clone/download the project
# Set up your .env file

# Start the API service
docker-compose up hackrx-api

# The API will be available at http://localhost:8000
```

#### Option B: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python api.py

# The API will be available at http://localhost:8000
```

### 3. API Usage

#### Authentication Header
```
Authorization: Bearer hackrx_2024_secret_key
```

#### Request Format
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the main topic of this document?",
        "What are the key findings?"
    ]
}
```

#### Response Format
```json
{
    "answers": [
        "The main topic is artificial intelligence in healthcare...",
        "The key findings include improved diagnostic accuracy..."
    ]
}
```

### 4. Test the API

Use the provided test script:

```bash
python test_hackrx_api.py
```

Or test manually with curl:

```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer hackrx_2024_secret_key" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is this document about?"]
  }'
```

### 5. Features

✅ **Exact API Contract Compliance** - Matches HackRx specification exactly
✅ **Bearer Token Authentication** - Secure API access
✅ **PDF URL Processing** - Downloads and processes PDFs from URLs
✅ **Persistent Vector Storage** - Caches processed documents for efficiency
✅ **Batch Question Processing** - Handles multiple questions in one request
✅ **Error Handling** - Comprehensive error responses
✅ **Logging** - Detailed logging for debugging
✅ **Docker Support** - Easy deployment with containers

### 6. Error Responses

#### 401 Unauthorized
```json
{
    "detail": "Invalid authentication credentials"
}
```

#### 400 Bad Request
```json
{
    "detail": "Document URL is required"
}
```

#### 500 Internal Server Error
```json
{
    "detail": "Internal server error: <error details>"
}
```

### 7. API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### 8. Performance Notes

- First request for a new document will take longer (processing time)
- Subsequent requests for the same document are much faster (cached)
- Processing time depends on document size and number of questions
- Typical response time: 10-60 seconds for new documents, 2-10 seconds for cached

### 9. Configuration

#### Environment Variables

- `HACKRX_API_KEY`: Bearer token for authentication (default: hackrx_2024_secret_key)
- `OPENAI_API_KEY`: Required for LLM and embeddings
- `PINECONE_API_KEY`: Required for vector storage
- `PINECONE_ENVIRONMENT`: Pinecone environment (default: gcp-starter)
- `LOG_LEVEL`: Logging level (default: INFO)

#### Docker Configuration

- Port 8000 is exposed for the API
- Logs are stored in `./logs` directory
- Temporary PDFs are stored in `./temp_pdfs` directory
- Health checks are configured

### 10. Troubleshooting

#### Common Issues

1. **401 Unauthorized**: Check your API key in the Authorization header
2. **PDF Download Fails**: Ensure the PDF URL is accessible and valid
3. **Processing Timeout**: Large documents may take longer to process
4. **API Keys Invalid**: Verify OpenAI and Pinecone keys are correct

#### Debug Steps

1. Check logs: `docker-compose logs hackrx-api`
2. Test health endpoint: `curl http://localhost:8000/`
3. Verify environment variables are loaded
4. Test with the provided test script

### 11. Competition Submission

For HackRx 6.0 submission:

1. Deploy the API to your preferred cloud platform
2. Update the `HACKRX_API_KEY` to your competition key
3. Provide the endpoint URL: `https://your-domain.com/hackrx/run`
4. Include the Bearer token for authentication
5. Test with the competition's sample documents

### Support

For issues or questions, check the logs and ensure all environment variables are properly configured.
