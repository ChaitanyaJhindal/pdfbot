"""
Simple test version without authentication to debug CORS
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Simple FastAPI app
app = FastAPI(title="PDF Chatbot API - Test")

# Simple CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TestRequest(BaseModel):
    documents: str
    questions: List[str]

class TestResponse(BaseModel):
    answers: List[str]

@app.get("/")
async def root():
    return {"message": "Test API is working"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/hackrx/run", response_model=TestResponse)
async def test_endpoint(request: TestRequest):
    """Simple test endpoint without authentication."""
    return TestResponse(answers=["Test answer for: " + q for q in request.questions])

# For Vercel
handler = app
