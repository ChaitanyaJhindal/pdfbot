"""
Vercel-optimized FastAPI web service for the PDF Chatbot - HackRx 6.0 Competition.
Provides REST API endpoint for processing documents and answering questions.
"""

import os
import logging
import requests
import tempfile
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends, status, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from advanced_pdf_bot import PDFChatbot

# Setup Vercel-specific logging (no file output)
def setup_vercel_logging():
    """Set up logging for Vercel serverless environment (console only)."""
    import sys
    
    # Create a custom stream handler for console output only
    class VercelStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                # Replace Unicode emojis with simple text
                msg = msg.replace('🔍', '[SEARCH]')
                msg = msg.replace('✅', '[SUCCESS]')
                msg = msg.replace('🤔', '[THINKING]')
                msg = msg.replace('❌', '[ERROR]')
                msg = msg.replace('⚠️', '[WARNING]')
                
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                pass  # Silent fail for serverless
    
    # Configure logging with console-only output
    logging.basicConfig(
        level=logging.WARNING,  # Reduced verbosity for serverless
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[VercelStreamHandler(sys.stdout)],
        force=True
    )
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.ERROR)

# Setup logging for Vercel
setup_vercel_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Chatbot API - HackRx 6.0",
    description="AI-powered PDF question-answering service for HackRx competition",
    version="1.0.0"
)

# Add CORS middleware with explicit configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly include OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# Add explicit OPTIONS handler for preflight requests
@app.options("/hackrx/run")
async def hackrx_options(response: Response):
    """Handle CORS preflight requests."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Max-Age"] = "3600"
    return {"message": "CORS preflight successful"}

# Authentication setup
security = HTTPBearer()
API_KEY = os.getenv("HACKRX_API_KEY", "hackrx_2024_secret_key")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the bearer token."""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Request/Response models for HackRx competition
class HackRxRequest(BaseModel):
    documents: str  # URL of the document to process
    questions: List[str]  # List of questions to answer

class HackRxResponse(BaseModel):
    answers: List[str]  # List of answers corresponding to the questions

def download_pdf_from_url(url: str) -> str:
    """Download PDF from URL and save to temporary file."""
    try:
        logger.info(f"Downloading PDF from URL: {url}")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_path = temp_file.name
            
            # Download the file with shorter timeout for serverless
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Write content to temp file
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
                
        logger.info(f"PDF downloaded successfully to: {temp_path}")
        return temp_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading PDF: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error downloading PDF: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def process_document_and_get_chatbot(document_url: str) -> PDFChatbot:
    """Process a document from URL and return a chatbot instance."""
    try:
        # Download PDF
        pdf_path = download_pdf_from_url(document_url)
        
        # Create and setup chatbot (no caching for serverless)
        logger.info("Creating new chatbot instance")
        chatbot = PDFChatbot(pdf_path, force_reprocess=False)
        chatbot.setup_index()
        chatbot.process_document()
        
        logger.info("Chatbot created successfully")
        return chatbot
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "PDF Chatbot API for HackRx 6.0 is running on Vercel", "status": "healthy"}

@app.get("/health")
async def health():
    """Additional health check endpoint."""
    return {"status": "ok", "platform": "vercel"}

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_endpoint(
    request: HackRxRequest,
    response: Response,
    token: str = Depends(verify_token)
):
    """
    HackRx 6.0 competition endpoint.
    Process a document from URL and answer questions about it.
    """
    # Add explicit CORS headers
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    
    try:
        logger.info(f"Processing HackRx request with {len(request.questions)} questions")
        
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="Document URL is required")
        
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Limit questions for serverless performance
        if len(request.questions) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 questions allowed per request")
        
        # Process the document and get chatbot
        chatbot = process_document_and_get_chatbot(request.documents)
        
        # Process each question
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}")
            
            try:
                # Get answer from chatbot
                answer_data = chatbot.ask_question(question)
                
                # Extract the answer text
                if isinstance(answer_data, dict):
                    answer_text = answer_data.get('answer', 'No answer could be generated.')
                else:
                    answer_text = str(answer_data)
                
                answers.append(answer_text)
                logger.info(f"Answer {i+1} generated successfully")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
                answers.append(f"Error processing question: {str(e)}")
        
        logger.info(f"Successfully processed all {len(request.questions)} questions")
        
        # Clean up temp file
        try:
            if hasattr(chatbot, 'pdf_path') and os.path.exists(chatbot.pdf_path):
                os.remove(chatbot.pdf_path)
        except Exception:
            pass  # Ignore cleanup errors
        
        return HackRxResponse(answers=answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in hackrx_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Export the app for Vercel
handler = app
