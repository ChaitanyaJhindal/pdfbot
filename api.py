"""
FastAPI web service for the PDF Chatbot - HackRx 6.0 Competition.
Optimized for hackathon judging with exact API specification compliance.
"""

import os
import shutil
import logging
import requests
import tempfile
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
import time
from datetime import datetime
from advanced_pdf_bot import PDFChatbot, setup_logging
from cors_config import CORS_DEV, CORS_PROD, CORS_HACKRX, CORS_EXTERNAL_TESTING

# Setup logging optimized for hackathon
def setup_hackathon_logging():
    """Setup logging optimized for hackathon debugging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    # Reduce noise from external libraries during hackathon
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

setup_hackathon_logging()
logger = logging.getLogger(__name__)

# Hackathon-optimized FastAPI app
app = FastAPI(
    title="HackRx 6.0 PDF Chatbot API",
    description="Competition-ready PDF question-answering service with exact specification compliance",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Environment-based CORS configuration
def get_cors_config():
    """Get CORS configuration based on environment."""
    env = os.getenv("ENVIRONMENT", "hackrx").lower()
    
    if env == "production":
        return CORS_PROD
    elif env == "external_testing" or env == "remote_testing":
        return CORS_EXTERNAL_TESTING
    else:
        return CORS_HACKRX  # Default to hackathon config

# Add CORS middleware with hackathon-optimized configuration
cors_config = get_cors_config()
app.add_middleware(CORSMiddleware, **cors_config)

# Authentication setup for hackathon
security = HTTPBearer(auto_error=False)
HACKRX_API_KEY = os.getenv("HACKRX_API_KEY", "hackrx_2024_secret_key")

def verify_hackrx_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token for HackRx competition."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != HACKRX_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials

# HackRx Competition Request/Response models with exact specification
class HackRxRequest(BaseModel):
    """Request model matching exact HackRx specification."""
    documents: str = Field(
        ..., 
        description="URL of the PDF document to process",
        example="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    )
    questions: List[str] = Field(
        ..., 
        min_items=1,
        max_items=20,  # Reasonable limit for hackathon
        description="List of questions to ask about the document",
        example=[
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?"
        ]
    )
    
    @validator('documents')
    def validate_document_url(cls, v):
        """Validate document URL format."""
        if not v or not v.strip():
            raise ValueError("Document URL cannot be empty")
        if not v.startswith(('http://', 'https://')):
            raise ValueError("Document URL must be a valid HTTP/HTTPS URL")
        return v.strip()
    
    @validator('questions')
    def validate_questions(cls, v):
        """Validate questions list."""
        if not v:
            raise ValueError("At least one question is required")
        
        # Clean and validate each question
        cleaned_questions = []
        for q in v:
            if not q or not q.strip():
                continue
            cleaned_questions.append(q.strip())
        
        if not cleaned_questions:
            raise ValueError("At least one non-empty question is required")
        
        return cleaned_questions

class HackRxResponse(BaseModel):
    """Response model matching exact HackRx specification."""
    answers: List[str] = Field(
        ..., 
        description="List of answers corresponding to the input questions"
    )
    questions: List[str]  # List of questions to answer

# Global storage for chatbot instances (optimized for hackathon)
chatbots = {}
TEMP_DIR = "temp_pdfs"
os.makedirs(TEMP_DIR, exist_ok=True)

# Hackathon performance tracking
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_response_time": 0
}

def download_pdf_from_url(url: str) -> str:
    """
    Download PDF from URL and save to temporary file.
    Optimized for hackathon with better error handling and logging.
    """
    try:
        logger.info(f"📥 Downloading PDF from URL: {url[:100]}...")
        start_time = time.time()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_DIR) as temp_file:
            temp_path = temp_file.name
            
            # Download the file with optimized settings for hackathon
            headers = {
                'User-Agent': 'HackRx-PDFBot/2.0',
                'Accept': 'application/pdf,*/*'
            }
            
            response = requests.get(
                url, 
                stream=True, 
                timeout=60,  # Increased timeout for large competition files
                headers=headers,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Verify content type
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                logger.warning(f"⚠️  Content type might not be PDF: {content_type}")
            
            # Write content to temp file with progress tracking
            total_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
                total_size += len(chunk)
            
            download_time = time.time() - start_time
            logger.info(f"✅ PDF downloaded successfully: {total_size/1024/1024:.2f}MB in {download_time:.2f}s")
            
            return temp_path
            
    except requests.exceptions.Timeout:
        logger.error("⏰ Download timeout - file might be too large or server slow")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Document download timed out. Please try with a smaller file or check the URL."
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Download error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download document: {str(e)}"
        )
    except Exception as e:
        logger.error(f"❌ Unexpected download error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error downloading document: {str(e)}"
        )

def process_document_and_get_chatbot(document_url: str) -> PDFChatbot:
    """
    Process a document from URL and return a chatbot instance.
    Optimized for hackathon with caching and performance monitoring.
    """
    # Generate cache key from URL
    cache_key = document_url
    
    # Check if we already have a chatbot for this URL (hackathon optimization)
    if cache_key in chatbots:
        logger.info(f"🚀 Using cached chatbot for document")
        return chatbots[cache_key]
    
    try:
        start_time = time.time()
        logger.info("🤖 Creating new chatbot instance...")
        
        # Download PDF
        pdf_path = download_pdf_from_url(document_url)
        
        # Create and setup chatbot with hackathon optimizations
        chatbot = PDFChatbot(pdf_path, force_reprocess=False)
        chatbot.setup_index()
        
        # Process document with timing
        process_start = time.time()
        chatbot.process_document()
        process_time = time.time() - process_start
        
        # Cache the chatbot for future requests (hackathon optimization)
        chatbots[cache_key] = chatbot
        
        total_time = time.time() - start_time
        logger.info(f"✅ Chatbot ready! Processing: {process_time:.2f}s, Total: {total_time:.2f}s")
        
        return chatbot
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Chatbot creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing document: {str(e)}"
        )

# Hackathon-optimized startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application with hackathon optimizations."""
    logger.info("🚀 Starting HackRx 6.0 PDF Chatbot API")
    logger.info(f"📊 Environment: {os.getenv('ENVIRONMENT', 'hackrx')}")
    logger.info(f"🔐 Auth: {'Enabled' if HACKRX_API_KEY else 'Disabled'}")
    logger.info(f"📁 Temp directory: {TEMP_DIR}")

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint with hackathon-specific information."""
    return {
        "message": "HackRx 6.0 PDF Chatbot API is running",
        "status": "healthy",
        "version": "2.0.0",
        "competition": "HackRx 6.0",
        "endpoint": "/hackrx/run",
        "docs": "/docs",
        "stats": request_stats
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check for hackathon monitoring."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "hackrx"),
        "cached_documents": len(chatbots),
        "temp_files": len(os.listdir(TEMP_DIR)) if os.path.exists(TEMP_DIR) else 0,
        "stats": request_stats
    }

@app.post("/hackrx/run", 
          response_model=HackRxResponse,
          tags=["HackRx Competition"],
          summary="Process PDF and answer questions",
          description="Main competition endpoint - processes a PDF document and answers questions about it")
async def hackrx_endpoint(
    request: HackRxRequest,
    auth_token: str = Depends(verify_hackrx_auth)
):
    """
    HackRx 6.0 Competition Endpoint - Exact specification compliance.
    
    This endpoint processes a PDF document from a URL and answers questions about it.
    Optimized for hackathon judging with comprehensive error handling and performance tracking.
    """
    start_time = time.time()
    request_stats["total_requests"] += 1
    
    try:
        logger.info(f"🎯 Processing HackRx request: {len(request.questions)} questions")
        logger.info(f"📄 Document: {request.documents[:100]}...")
        
        # Process the document and get chatbot
        chatbot = process_document_and_get_chatbot(request.documents)
        
        # Process each question with detailed logging
        answers = []
        for i, question in enumerate(request.questions):
            question_start = time.time()
            logger.info(f"❓ Question {i+1}/{len(request.questions)}: {question[:100]}...")
            
            try:
                # Get answer from chatbot
                answer_data = chatbot.ask_question(question)
                
                # Extract and clean the answer text
                if isinstance(answer_data, dict):
                    answer_text = answer_data.get('answer', 'No answer could be generated.')
                elif isinstance(answer_data, str):
                    answer_text = answer_data
                else:
                    answer_text = str(answer_data)
                
                # Clean up the answer
                answer_text = answer_text.strip()
                if not answer_text:
                    answer_text = "Unable to generate an answer for this question."
                
                answers.append(answer_text)
                
                question_time = time.time() - question_start
                logger.info(f"✅ Answer {i+1} ready ({question_time:.2f}s): {answer_text[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Error processing question {i+1}: {e}")
                error_answer = f"Error processing question: {str(e)}"
                answers.append(error_answer)
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        request_stats["successful_requests"] += 1
        request_stats["average_response_time"] = (
            (request_stats["average_response_time"] * (request_stats["successful_requests"] - 1) + total_time) /
            request_stats["successful_requests"]
        )
        
        logger.info(f"🏁 Request completed: {len(answers)} answers in {total_time:.2f}s")
        
        # Return response in exact HackRx format
        return HackRxResponse(answers=answers)
        
    except HTTPException as e:
        request_stats["failed_requests"] += 1
        logger.error(f"❌ HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        request_stats["failed_requests"] += 1
        logger.error(f"❌ Unexpected error in hackrx_endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

# Hackathon cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown with hackathon-specific logging."""
    logger.info("🛑 Shutting down HackRx 6.0 PDF Chatbot API")
    logger.info(f"📊 Final stats: {request_stats}")
    
    # Clean up temporary files
    try:
        import glob
        temp_files = glob.glob(os.path.join(TEMP_DIR, "*.pdf"))
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                logger.info(f"🧹 Cleaned up: {temp_file}")
            except Exception as e:
                logger.warning(f"⚠️  Could not remove {temp_file}: {e}")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")

# Hackathon-optimized server configuration
if __name__ == "__main__":
    logger.info("🎮 Starting in development mode for hackathon testing")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=False,  # Disabled for hackathon stability
        log_level="info",
        workers=1  # Single worker for hackathon simplicity
    )
