# CORS Issues and Solutions Guide

## Current CORS Issues in Your Project

### 1. **Security Vulnerability - Overly Permissive Configuration**

**Problem**: Your current CORS setup allows ANY website to access your API:
```python
allow_origins=["*"]  # ⚠️ SECURITY RISK
allow_methods=["*"]  # ⚠️ TOO PERMISSIVE  
allow_headers=["*"]  # ⚠️ TOO PERMISSIVE
```

**Impact**:
- Any malicious website can call your API
- Potential for abuse and unauthorized usage
- API key exposure risk
- Rate limiting bypass attempts

### 2. **Inconsistent Configuration Across Files**

**Problem**: Different CORS configs in different files:
- `api.py`: `allow_methods=["*"]`
- `api_vercel.py`: `allow_methods=["GET", "POST", "OPTIONS"]`

**Impact**: Confusing behavior depending on deployment method

### 3. **Missing Environment-Specific Configuration**

**Problem**: Same permissive config for development and production

**Impact**: Production security vulnerabilities

## Solutions Implemented

### 1. **Environment-Based CORS Configuration**

Now your API uses different CORS settings based on the `ENVIRONMENT` variable:

```python
# Development (permissive for testing)
ENVIRONMENT=development  

# Production (secure, specific domains only)
ENVIRONMENT=production   

# Competition (permissive for judges)
ENVIRONMENT=hackrx       
```

### 2. **Secure Production Configuration**

For production, CORS is now restricted to:
- Specific allowed domains only
- Limited HTTP methods (GET, POST, OPTIONS)
- Specific headers only
- No credentials allowed

### 3. **Competition-Friendly Configuration**

For HackRx competition:
- Allows all origins (for judges to test)
- Limited to safe HTTP methods
- Specific headers only

## Deployment-Specific CORS Settings

### For Render Deployment

Set this environment variable in Render:
```
ENVIRONMENT=hackrx
```

This will use the competition-friendly CORS settings.

### For Production Deployment

1. Set environment variable:
   ```
   ENVIRONMENT=production
   ```

2. Update `cors_config.py` with your actual domains:
   ```python
   CORS_PROD = {
       "allow_origins": [
           "https://yourdomain.com",
           "https://your-frontend.vercel.app",
           # Add your real domains here
       ],
       # ... rest of config
   }
   ```

## Common CORS Errors and Solutions

### 1. **"Access to fetch has been blocked by CORS policy"**

**Cause**: Frontend domain not in `allow_origins`

**Solution**: 
- Add your frontend domain to the allowed origins list
- Or use `ENVIRONMENT=development` for testing

### 2. **"Request header 'authorization' is not allowed"**

**Cause**: Authorization header not in `allow_headers`

**Solution**: Already fixed - Authorization is included in all configs

### 3. **"Method 'PUT' is not allowed"**

**Cause**: PUT method not in `allow_methods`

**Solution**: Add "PUT" to `allow_methods` if needed, or use POST

### 4. **Preflight OPTIONS Request Failing**

**Cause**: Server not handling OPTIONS requests properly

**Solution**: 
- "OPTIONS" is included in all configs
- FastAPI handles this automatically with CORSMiddleware

## Testing CORS Configuration

### 1. **Test with curl**
```bash
# Test preflight request
curl -X OPTIONS \
  -H "Origin: https://example.com" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  https://your-api.onrender.com/hackrx/run

# Test actual request
curl -X POST \
  -H "Origin: https://example.com" \
  -H "Content-Type: application/json" \
  -d '{"documents":"https://example.com/test.pdf","questions":["test?"]}' \
  https://your-api.onrender.com/hackrx/run
```

### 2. **Test from Browser Console**
```javascript
// Test from browser developer console
fetch('https://your-api.onrender.com/hackrx/run', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    documents: 'https://example.com/test.pdf',
    questions: ['What is this document about?']
  })
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('CORS Error:', error));
```

## Best Practices

### 1. **Environment-Specific Configuration**
- Development: Permissive for easy testing
- Production: Restrictive for security
- Competition: Balanced for judging

### 2. **Specific Domain Allowlist**
```python
# ✅ Good - Specific domains
allow_origins=["https://myapp.com", "https://api.myapp.com"]

# ❌ Bad - Wildcard
allow_origins=["*"]
```

### 3. **Minimal Required Headers**
```python
# ✅ Good - Only required headers
allow_headers=["Content-Type", "Authorization"]

# ❌ Bad - All headers
allow_headers=["*"]
```

### 4. **Limited HTTP Methods**
```python
# ✅ Good - Only needed methods
allow_methods=["GET", "POST", "OPTIONS"]

# ❌ Bad - All methods
allow_methods=["*"]
```

## Environment Variables for Render

When deploying to Render, set these environment variables:

```
ENVIRONMENT=hackrx                    # Use competition-friendly CORS
OPENAI_API_KEY=your_openai_key       # Required
PINECONE_API_KEY=your_pinecone_key   # Required
PINECONE_ENVIRONMENT=gcp-starter     # Optional
HACKRX_API_KEY=hackrx_2024_secret_key # Optional
LOG_LEVEL=INFO                       # Optional
```

This configuration balances security with the needs of a competition environment.
