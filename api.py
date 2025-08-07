"""
FastAPI web service for the PDF Chatbot - HackRx 6.0 Competition.
Provides REST API endpoint for processing documents and answering questions.
"""

import os
import shutil
import logging
import requests
import tempfile
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
import uvicorn
from advanced_pdf_bot import PDFChatbot, setup_logging

# Setup logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PDF Chatbot API - HackRx 6.0",
    description="AI-powered PDF question-answering service for HackRx competition",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Request/Response models for HackRx competition
class HackRxRequest(BaseModel):
    documents: str  # URL of the document to process
    questions: List[str]  # List of questions to answer

class HackRxResponse(BaseModel):
    answers: List[str]  # List of answers corresponding to the questions

# Global storage for chatbot instances (for caching)
chatbots = {}
TEMP_DIR = "temp_pdfs"
os.makedirs(TEMP_DIR, exist_ok=True)

def download_pdf_from_url(url: str) -> str:
    """
    Download PDF from URL and save to temporary file.
    
    Args:
        url: URL of the PDF to download
        
    Returns:
        str: Path to the downloaded PDF file
    """
    try:
        logger.info(f"Downloading PDF from URL: {url}")
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', dir=TEMP_DIR) as temp_file:
            temp_path = temp_file.name
            
            # Download the file
            response = requests.get(url, stream=True, timeout=60)
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
    """
    Process a document from URL and return a chatbot instance.
    
    Args:
        document_url: URL of the document to process
        
    Returns:
        PDFChatbot: Configured chatbot instance
    """
    # Check if we already have a chatbot for this URL
    if document_url in chatbots:
        logger.info(f"Using cached chatbot for URL: {document_url}")
        return chatbots[document_url]
    
    try:
        # Download PDF
        pdf_path = download_pdf_from_url(document_url)
        
        # Create and setup chatbot
        logger.info("Creating new chatbot instance")
        chatbot = PDFChatbot(pdf_path, force_reprocess=False)
        chatbot.setup_index()
        chatbot.process_document()
        
        # Cache the chatbot
        chatbots[document_url] = chatbot
        logger.info(f"Chatbot created and cached for URL: {document_url}")
        
        return chatbot
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Starting PDF Chatbot API for HackRx 6.0")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "PDF Chatbot API for HackRx 6.0 is running", "status": "healthy"}

@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_endpoint(request: HackRxRequest):
    """
    HackRx 6.0 competition endpoint.
    Process a document from URL and answer questions about it.
    
    Args:
        request: HackRxRequest containing document URL and questions
    
    Returns:
        HackRxResponse: Answers to the questions
    """
    try:
        logger.info(f"Processing HackRx request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="Document URL is required")
        
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        # Process the document and get chatbot
        chatbot = process_document_and_get_chatbot(request.documents)
        
        # Process each question
        answers = []
        for i, question in enumerate(request.questions):
            logger.info(f"Processing question {i+1}/{len(request.questions)}: {question}")
            
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
        
        return HackRxResponse(answers=answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in hackrx_endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down PDF Chatbot API")
    
    # Clean up temporary files
    try:
        import glob
        temp_files = glob.glob(os.path.join(TEMP_DIR, "*.pdf"))
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                logger.info(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {e}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
