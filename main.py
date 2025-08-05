"""
Vercel entry point for the PDF Chatbot API.
"""

from api_vercel import app

# Vercel expects the FastAPI app to be available as 'app'
# This is imported from api_vercel.py which is optimized for serverless

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
