# External Remote Testing Deployment Guide

## Optimized Configuration for External Testers

Your API is now configured with `ENVIRONMENT=external_testing` which provides:

### 🌐 **Maximum Compatibility CORS Settings**
- **Origins**: `["*"]` - Any domain can access your API
- **Methods**: `["GET", "POST", "OPTIONS", "HEAD"]` - All common HTTP methods
- **Headers**: `["*"]` - All headers allowed for maximum compatibility
- **Max Age**: `3600` seconds - Reduces preflight requests for better performance

### 🚀 **Quick Deployment to Render**

1. **Deploy to Render**:
   ```bash
   # Push your code to GitHub first
   git add .
   git commit -m "Configure for external remote testing"
   git push origin main
   ```

2. **Create Web Service on Render**:
   - Go to https://render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Use these settings:
     - **Name**: `pdf-chatbot-external-test`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn api:app --host 0.0.0.0 --port $PORT --workers 1`

3. **Set Environment Variables**:
   ```
   ENVIRONMENT=external_testing
   OPENAI_API_KEY=your_actual_openai_key
   PINECONE_API_KEY=your_actual_pinecone_key
   PINECONE_ENVIRONMENT=gcp-starter
   HACKRX_API_KEY=hackrx_2024_secret_key
   LOG_LEVEL=INFO
   ```

### 🧪 **Testing Your Deployed API**

Once deployed (usually at `https://your-service-name.onrender.com`), test with:

#### **1. Health Check**
```bash
curl https://your-service-name.onrender.com/
```

#### **2. API Documentation**
Visit in browser:
- Swagger UI: `https://your-service-name.onrender.com/docs`
- ReDoc: `https://your-service-name.onrender.com/redoc`

#### **3. Full API Test**
```bash
curl -X POST "https://your-service-name.onrender.com/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
    "questions": ["What is this document about?", "What is the main content?"]
  }'
```

#### **4. Browser-based Test**
Open browser console on any website and run:
```javascript
fetch('https://your-service-name.onrender.com/hackrx/run', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    documents: 'https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf',
    questions: ['What is this document about?']
  })
})
.then(response => response.json())
.then(data => console.log('Success:', data))
.catch(error => console.error('Error:', error));
```

### 🔧 **For Remote Testers**

Share this information with your external testers:

#### **API Endpoint**
```
POST https://your-service-name.onrender.com/hackrx/run
```

#### **Request Format**
```json
{
  "documents": "https://example.com/path/to/document.pdf",
  "questions": [
    "What is this document about?",
    "What are the key points?",
    "Who is the author?"
  ]
}
```

#### **Response Format**
```json
{
  "answers": [
    "This document is about...",
    "The key points are...",
    "The author is..."
  ]
}
```

#### **Headers Required**
```
Content-Type: application/json
```

#### **Optional Authentication**
```
Authorization: Bearer hackrx_2024_secret_key
```

### 📋 **Testing Checklist for External Testers**

- [ ] ✅ Health check endpoint responds
- [ ] ✅ API documentation is accessible
- [ ] ✅ Can upload test PDF via URL
- [ ] ✅ Can ask questions about the PDF
- [ ] ✅ Receives appropriate answers
- [ ] ✅ Error handling works (invalid URLs, malformed requests)
- [ ] ✅ CORS works from different domains/tools

### ⚡ **Performance Notes for External Testing**

1. **First Request**: May take 30-60 seconds (cold start on free tier)
2. **Subsequent Requests**: Much faster (service is warm)
3. **Large PDFs**: May timeout on free tier (consider upgrading for production)
4. **Concurrent Requests**: Limited on free tier

### 🛠️ **Common Testing Tools**

Your API is compatible with:
- **Postman** - Import the OpenAPI spec from `/docs`
- **Insomnia** - REST client testing
- **curl** - Command line testing
- **Browser** - Direct JavaScript fetch requests
- **Python requests** - Programmatic testing
- **Automated testing frameworks** - Jest, pytest, etc.

### 🚨 **Important Notes**

1. **Free Tier Limitations**:
   - Service sleeps after 15 minutes of inactivity
   - Cold starts add latency
   - 750 hours/month limit

2. **Rate Limiting**: 
   - No artificial limits set
   - Limited by OpenAI/Pinecone API limits

3. **File Size Limits**:
   - Large PDFs may cause timeouts
   - Consider 10MB limit for reliable processing

### 📞 **Support for External Testers**

If testers encounter issues:
1. Check the live logs in Render dashboard
2. Verify the API is not sleeping (send health check first)
3. Test with smaller PDF files first
4. Ensure proper JSON formatting

Your API is now optimized for external remote testing with maximum compatibility!
