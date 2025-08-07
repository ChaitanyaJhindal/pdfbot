"""
Vercel entry point for the PDF Chatbot API.
"""

# Temporarily use test API to debug CORS
from test_api import app

# Vercel expects the FastAPI app to be available as 'app'

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
