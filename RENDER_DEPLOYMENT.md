# Deploying PDF Chatbot to Render

This guide will help you deploy your PDF chatbot application to Render.

## Prerequisites

1. A Render account (sign up at https://render.com)
2. Your code pushed to a Git repository (GitHub, GitLab, or Bitbucket)
3. Required API keys:
   - OpenAI API Key
   - Pinecone API Key

## Deployment Steps

### Method 1: Using Render Dashboard (Recommended)

1. **Connect Your Repository**
   - Log into your Render dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub/GitLab account if not already connected
   - Select your repository containing this PDF bot code

2. **Configure Your Service**
   - **Name**: `pdf-chatbot-api` (or any name you prefer)
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Instance Type**: Free (or paid for better performance)

3. **Set Environment Variables**
   Go to the "Environment" tab and add these variables:
   
   **Required:**
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PINECONE_API_KEY`: Your Pinecone API key
   
   **Recommended for External Testing:**
   - `ENVIRONMENT`: `external_testing` (optimized CORS for remote testing)
   
   **Optional (with defaults):**
   - `PINECONE_ENVIRONMENT`: `gcp-starter` (default)
   - `HACKRX_API_KEY`: `hackrx_2024_secret_key` (default)
   - `LOG_LEVEL`: `INFO` (default)

4. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - Wait for the build to complete (usually 3-5 minutes)

### Method 2: Using render.yaml (Infrastructure as Code)

1. The `render.yaml` file is already created in your project
2. Push your code with this file to your repository
3. In Render dashboard, click "New +" → "Blueprint"
4. Connect your repository and Render will use the YAML configuration
5. Update the environment variables in the Render dashboard after deployment

## Post-Deployment

### Testing Your Deployment

Once deployed, your API will be available at: `https://your-service-name.onrender.com`

Test the endpoints:

1. **Health Check**:
   ```bash
   curl https://your-service-name.onrender.com/
   ```

2. **API Test**:
   ```bash
   curl -X POST "https://your-service-name.onrender.com/hackrx/run" \
        -H "Content-Type: application/json" \
        -d '{
          "documents": "https://example.com/sample.pdf",
          "questions": ["What is this document about?"]
        }'
   ```

### Important Notes

1. **Free Tier Limitations**:
   - Service spins down after 15 minutes of inactivity
   - 750 hours per month limit
   - Cold starts take 30-60 seconds

2. **Performance Optimization**:
   - Consider upgrading to a paid plan for production use
   - Free tier may timeout on large PDF processing
   - Consider implementing caching for better performance

3. **Monitoring**:
   - Check logs in Render dashboard if issues occur
   - Monitor resource usage and response times

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | Your OpenAI API key for AI processing |
| `PINECONE_API_KEY` | Yes | - | Your Pinecone API key for vector storage |
| `PINECONE_ENVIRONMENT` | No | `gcp-starter` | Pinecone environment region |
| `HACKRX_API_KEY` | No | `hackrx_2024_secret_key` | API key for authentication |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |

## Troubleshooting

### Common Issues

1. **Build Fails**:
   - Check that all dependencies in `requirements.txt` are compatible
   - Verify Python version compatibility

2. **Service Won't Start**:
   - Check environment variables are set correctly
   - Review logs in Render dashboard
   - Ensure start command is correct

3. **API Timeouts**:
   - Large PDFs may cause timeouts on free tier
   - Consider upgrading to a paid plan
   - Implement request size limits

4. **Memory Issues**:
   - Free tier has limited memory (512MB)
   - Large documents may cause out-of-memory errors
   - Consider implementing document size limits

### Getting Help

- Check Render documentation: https://render.com/docs
- Review application logs in Render dashboard
- Test locally first with `uvicorn api:app --reload`

## API Documentation

Once deployed, your API documentation will be available at:
- Swagger UI: `https://your-service-name.onrender.com/docs`
- ReDoc: `https://your-service-name.onrender.com/redoc`
