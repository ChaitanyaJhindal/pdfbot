"""
Advanced PDF Chatbot using Pinecone for vector search and OpenAI LLM for generating answers.

This script functions as an intelligent PDF chatbot that:
1. Extracts text from PDF files
2. Chunks the text for processing
3. Creates embeddings and stores them in Pinecone
4. Performs semantic search to find relevant context
5. Uses OpenAI LLM to generate answers based on the context

Requirements:
- Set up environment variables in a .env file
- Install required packages: pypdf, langchain-text-splitters, openai, pinecone-client, python-dotenv
"""

import os
import re
import logging
import json
import time
from typing import List, Any, Dict, TypedDict
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Try to import sentence-transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


def setup_logging(log_level: str = "INFO"):
    """
    Set up logging configuration with Unicode support for Windows.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    import sys
    
    # Create a custom stream handler for Windows console
    class UnicodeStreamHandler(logging.StreamHandler):
        def emit(self, record):
            try:
                msg = self.format(record)
                # Replace Unicode emojis with simple text for Windows compatibility
                msg = msg.replace('🔍', '[SEARCH]')
                msg = msg.replace('✅', '[SUCCESS]')
                msg = msg.replace('🤔', '[THINKING]')
                msg = msg.replace('❌', '[ERROR]')
                msg = msg.replace('⚠️', '[WARNING]')
                
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except UnicodeEncodeError:
                # Fallback: encode to ASCII, ignoring problematic characters
                try:
                    msg_ascii = msg.encode('ascii', 'ignore').decode('ascii')
                    stream.write(msg_ascii + self.terminator)
                    self.flush()
                except Exception:
                    pass  # Silent fail to prevent logging loops
    
    # Set up logging with Unicode-safe handlers
    handlers = [
        UnicodeStreamHandler(sys.stdout),
        logging.FileHandler('pdf_chatbot.log', encoding='utf-8')
    ]
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Reduce noise from external libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("pinecone").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# Configuration - fetch API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")  # Default environment

# For embeddings, we need a real OpenAI key since Groq doesn't support embeddings
OPENAI_EMBEDDING_KEY = os.getenv("OPENAI_EMBEDDING_KEY", OPENAI_API_KEY)

# Validate that all required environment variables are set
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables")

# Initialize clients
# Groq client for LLM (chat completions)
groq_client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# OpenAI client for embeddings (since Groq doesn't support embeddings)
# If OPENAI_EMBEDDING_KEY is the same as OPENAI_API_KEY (Groq key), 
# we'll use HuggingFace embeddings instead
if OPENAI_EMBEDDING_KEY.startswith("gsk_"):
    # This is a Groq key, use HuggingFace embeddings
    USE_OPENAI_EMBEDDINGS = False
    embedding_client = None
else:
    # This is an OpenAI key, use OpenAI embeddings
    USE_OPENAI_EMBEDDINGS = True
    embedding_client = openai.OpenAI(api_key=OPENAI_EMBEDDING_KEY)

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# Global constants - Updated for Groq compatibility
EMBEDDING_MODEL = "text-embedding-3-small" if USE_OPENAI_EMBEDDINGS else "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-70b-versatile"  # Groq's powerful model
PINECONE_INDEX_NAME = "pdf-chatbot-index"
EMBEDDING_DIMENSION = 1536 if USE_OPENAI_EMBEDDINGS else 384  # Different dimensions for different models

# Initialize embedding model for local embeddings if needed
local_embedding_model = None
if not USE_OPENAI_EMBEDDINGS and SENTENCE_TRANSFORMERS_AVAILABLE:
    local_embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Get embeddings for texts using either OpenAI or local model."""
    if USE_OPENAI_EMBEDDINGS and embedding_client:
        # Use OpenAI embeddings
        response = embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    elif local_embedding_model:
        # Use local sentence transformer
        embeddings = local_embedding_model.encode(texts)
        return embeddings.tolist()
    else:
        # Fallback - use Groq (though it doesn't support embeddings)
        # This will likely fail, but we'll try anyway
        try:
            response = groq_client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            raise ValueError(f"No embedding service available. OpenAI: {USE_OPENAI_EMBEDDINGS}, Local: {SENTENCE_TRANSFORMERS_AVAILABLE}, Error: {e}")

def get_single_embedding(text: str) -> List[float]:
    """Get embedding for a single text."""
    return get_embeddings([text])[0]

# LangGraph State Definition
class ProcessingState(TypedDict):
    query: str
    pdf_path: str
    doc_id: str
    document_summary: str
    page_summaries: List[str]
    relevant_pages: List[int]
    context: str
    answer: str
    should_process: bool
    error: str


def extract_text_from_pdf(pdf_path: str) -> Dict[int, str]:
    """
    Extract text from a PDF file page by page using pypdf library.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Dict[int, str]: Dictionary with page numbers as keys and page text as values
        
    Raises:
        FileNotFoundError: If the PDF file is not found
        Exception: For other PDF processing errors
    """
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        pages_text = {}
        print(f"Extracting text from PDF: {pdf_path}")
        
        # Open and read the PDF file
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            print(f"Processing {total_pages} pages...")
            
            # Extract text from each page separately
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                pages_text[page_num] = page_text
                print(f"Processed page {page_num}/{total_pages}")
        
        print(f"Successfully extracted text from {len(pages_text)} pages")
        return pages_text
        
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        raise
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        raise


def get_text_chunks_by_page(pages_text: Dict[int, str]) -> Dict[int, List[str]]:
    """
    Split text into chunks page by page using RecursiveCharacterTextSplitter.
    
    Args:
        pages_text (Dict[int, str]): Dictionary with page numbers and their text
        
    Returns:
        Dict[int, List[str]]: Dictionary with page numbers and their chunks
    """
    print("Splitting text into chunks by page...")
    
    # Initialize the text splitter with specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    page_chunks = {}
    total_chunks = 0
    
    for page_num, text in pages_text.items():
        if text.strip():  # Only process pages with content
            chunks = text_splitter.split_text(text)
            page_chunks[page_num] = chunks
            total_chunks += len(chunks)
            print(f"Page {page_num}: {len(chunks)} chunks")
    
    print(f"Created {total_chunks} total chunks across {len(page_chunks)} pages")
    return page_chunks


def generate_summary(text: str, summary_type: str = "document") -> str:
    """
    Generate a summary of the given text using OpenAI LLM.
    
    Args:
        text (str): Text to summarize
        summary_type (str): Type of summary ("document", "page", or "chunk")
        
    Returns:
        str: Generated summary
    """
    try:
        if summary_type == "document":
            prompt = f"""Please provide a comprehensive summary of this document. Include:
1. Main topics and themes
2. Key concepts and ideas
3. Important entities (people, places, organizations)
4. Overall purpose and scope

Document text:
{text[:4000]}"""  # Limit to avoid token limits
        
        elif summary_type == "page":
            prompt = f"""Please provide a concise summary of this page content. Include:
1. Main topics discussed
2. Key points and concepts
3. Important details

Page text:
{text}"""
        
        else:  # chunk
            prompt = f"""Please provide a brief summary of this text chunk focusing on:
1. Main topic
2. Key information
3. Context

Text chunk:
{text}"""

        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that creates concise, informative summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return f"Summary generation failed: {str(e)}"


def setup_pinecone_index(index_name: str, dimension: int, force_recreate: bool = False):
    """
    Set up Pinecone index - create if it doesn't exist, or connect to existing one.
    
    Args:
        index_name (str): Name of the Pinecone index
        dimension (int): Dimension of the embeddings (1536 for text-embedding-3-small)
        force_recreate (bool): If True, delete and recreate the index even if it exists
        
    Returns:
        tuple: (pinecone.Index, bool) - The Pinecone index object and whether it was newly created
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Setting up Pinecone index: {index_name}")
    
    # List existing indexes
    existing_indexes = [index.name for index in pinecone_client.list_indexes()]
    
    # Check if index exists
    if index_name in existing_indexes:
        if force_recreate:
            logger.info(f"Force recreating index {index_name}")
            try:
                pinecone_client.delete_index(index_name)
                logger.info(f"Index {index_name} deleted successfully")
            except Exception as e:
                logger.error(f"Error deleting existing index: {e}")
                raise
        else:
            # Try to connect to existing index
            try:
                index = pinecone_client.Index(index_name)
                # Test the connection and check stats
                stats = index.describe_index_stats()
                logger.info(f"Connected to existing index {index_name} with {stats.total_vector_count} vectors")
                return index, False
            except Exception as e:
                logger.warning(f"Error connecting to existing index: {e}")
                logger.info("Will recreate the index...")
                try:
                    pinecone_client.delete_index(index_name)
                    logger.info(f"Index {index_name} deleted successfully")
                except Exception as delete_error:
                    logger.error(f"Error deleting corrupted index: {delete_error}")
                    raise
    
    # Create new index
    logger.info(f"Creating new Pinecone index: {index_name}")
    try:
        pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        logger.info(f"Index {index_name} created successfully")
        
        # Connect to the new index
        index = pinecone_client.Index(index_name)
        logger.info(f"Connected to new index: {index_name}")
        return index, True
        
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise


def check_document_exists(index, doc_id: str) -> bool:
    """
    Check if a document already exists in the Pinecone index.
    
    Args:
        index: Pinecone index object
        doc_id (str): Document ID to check
        
    Returns:
        bool: True if document exists, False otherwise
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Query for document summary to check if document exists
        results = index.query(
            vector=[0.0] * EMBEDDING_DIMENSION,  # Dummy vector
            top_k=1,
            include_metadata=True,
            filter={"type": "document_summary", "doc_id": doc_id}
        )
        
        exists = len(results.matches) > 0
        if exists:
            logger.info(f"Document {doc_id} already exists in index")
        else:
            logger.info(f"Document {doc_id} not found in index")
            
        return exists
        
    except Exception as e:
        logger.warning(f"Error checking document existence: {e}")
        return False


def embed_and_upsert_with_summaries(index, pages_text: Dict[int, str], page_chunks: Dict[int, List[str]], doc_id: str):
    """
    Generate embeddings for text chunks and summaries, then upsert them into Pinecone index.
    
    Args:
        index: Pinecone index object
        pages_text (Dict[int, str]): Dictionary with page numbers and their text
        page_chunks (Dict[int, List[str]]): Dictionary with page numbers and their chunks
        doc_id (str): Unique identifier for the document
    """
    print(f"Generating embeddings and upserting with summaries...")
    
    try:
        # Generate document summary
        full_text = "\n".join(pages_text.values())
        doc_summary = generate_summary(full_text, "document")
        print("Generated document summary")
        
        # Store document summary
        doc_summary_embedding = get_single_embedding(doc_summary)
        
        # Upsert document summary
        index.upsert(vectors=[{
            "id": f"{doc_id}-summary",
            "values": doc_summary_embedding,
            "metadata": {
                "type": "document_summary",
                "text": doc_summary,
                "doc_id": doc_id
            }
        }])
        
        # Process each page
        vectors_to_upsert = []
        page_summaries = {}
        
        for page_num in sorted(page_chunks.keys()):
            page_text = pages_text[page_num]
            chunks = page_chunks[page_num]
            
            # Generate page summary
            page_summary = generate_summary(page_text, "page")
            page_summaries[page_num] = page_summary
            print(f"Generated summary for page {page_num}")
            
            # Create page summary embedding
            page_summary_embedding = get_single_embedding(page_summary)
            
            # Add page summary vector
            vectors_to_upsert.append({
                "id": f"{doc_id}-page-{page_num}-summary",
                "values": page_summary_embedding,
                "metadata": {
                    "type": "page_summary",
                    "text": page_summary,
                    "doc_id": doc_id,
                    "page_number": page_num,
                    "full_page_text": page_text
                }
            })
            
            # Generate embeddings for chunks
            if chunks:
                chunk_embeddings = get_embeddings(chunks)
                
                # Create vectors for each chunk
                for chunk_idx, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    # Include previous page summary as context if available
                    prev_summary = page_summaries.get(page_num - 1, "") if page_num > 1 else ""
                    
                    vector = {
                        "id": f"{doc_id}-page-{page_num}-chunk-{chunk_idx}",
                        "values": embedding,
                        "metadata": {
                            "type": "chunk",
                            "text": chunk,
                            "doc_id": doc_id,
                            "page_number": page_num,
                            "chunk_index": chunk_idx,
                            "page_summary": page_summary,
                            "prev_page_summary": prev_summary,
                            "full_page_text": page_text
                        }
                    }
                    vectors_to_upsert.append(vector)
        
        # Upsert vectors in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert) + batch_size - 1)//batch_size}")
        
        print(f"Successfully upserted {len(vectors_to_upsert)} vectors with summaries to Pinecone index")
        return doc_summary, page_summaries
        
    except Exception as e:
        print(f"Error during embedding and upsert: {str(e)}")
        raise


def check_query_relevance(index, query: str, doc_id: str) -> tuple[bool, str]:
    """
    Check if the user query is relevant to the document by comparing with document summary.
    
    Args:
        index: Pinecone index object
        query (str): User's query
        doc_id (str): Document ID
        
    Returns:
        tuple[bool, str]: (is_relevant, document_summary)
    """
    try:
        # Generate embedding for the query
        query_embedding = get_single_embedding(query)
        
        # Search for document summary
        search_results = index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True,
            filter={"type": "document_summary", "doc_id": doc_id}
        )
        
        if not search_results.matches:
            return False, "No document summary found"
        
        # Get similarity score
        similarity_score = search_results.matches[0].score
        document_summary = search_results.matches[0].metadata.get('text', '')
        
        # Threshold for relevance (can be adjusted)
        relevance_threshold = 0.2  # Lowered from 0.7 to 0.2 for better sensitivity
        
        print(f"Query relevance score: {similarity_score:.3f}")
        
        if similarity_score >= relevance_threshold:
            return True, document_summary
        else:
            # Use LLM to make final decision
            relevance_prompt = f"""
            Document Summary: {document_summary}
            
            User Query: {query}
            
            Is this query relevant to the document? Consider if the query could be answered using information from this document.
            Answer only 'YES' or 'NO' followed by a brief explanation.
            """
            
            response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines query relevance."},
                    {"role": "user", "content": relevance_prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )
            
            answer = response.choices[0].message.content.strip().upper()
            is_relevant = answer.startswith('YES')
            
            return is_relevant, document_summary
            
    except Exception as e:
        print(f"Error checking query relevance: {str(e)}")
        return True, ""  # Default to processing if error occurs


def get_relevant_context_enhanced(index, query: str, doc_id: str, top_k: int = 5) -> str:
    """
    Perform enhanced semantic search to get relevant context for the query.
    
    Args:
        index: Pinecone index object
        query (str): User's query
        doc_id (str): Document ID
        top_k (int): Number of top results to retrieve
        
    Returns:
        str: Combined context from retrieved chunks
    """
    try:
        # Generate embedding for the query
        query_embedding = get_single_embedding(query)
        
        # First, search for relevant page summaries
        page_summary_results = index.query(
            vector=query_embedding,
            top_k=3,
            include_metadata=True,
            filter={"type": "page_summary", "doc_id": doc_id}
        )
        
        relevant_pages = []
        if page_summary_results.matches:
            for match in page_summary_results.matches:
                if match.score > 0.2:  # Lowered threshold for page summaries
                    page_num = match.metadata.get('page_number')
                    if page_num:
                        relevant_pages.append(page_num)
        
        # If we have relevant pages, search for chunks within those pages
        if relevant_pages:
            print(f"Found relevant pages: {relevant_pages}")
            context_chunks = []
            
            for page_num in relevant_pages:
                # Get chunks from this specific page
                chunk_results = index.query(
                    vector=query_embedding,
                    top_k=5,
                    include_metadata=True,
                    filter={"type": "chunk", "doc_id": doc_id, "page_number": page_num}
                )
                
                for match in chunk_results.matches:
                    if match.score > 0.2:  # Lowered relevance threshold for chunks
                        chunk_text = match.metadata.get('text', '')
                        page_summary = match.metadata.get('page_summary', '')
                        prev_summary = match.metadata.get('prev_page_summary', '')
                        
                        # Create enhanced context with summaries
                        enhanced_chunk = f"""
                        [Page {page_num} Context]
                        Previous Page Summary: {prev_summary}
                        Current Page Summary: {page_summary}
                        
                        Content: {chunk_text}
                        """
                        context_chunks.append(enhanced_chunk)
            
            if context_chunks:
                return "\n\n".join(context_chunks[:top_k])
        
        # Fallback: general search across all chunks
        print("Performing general search across all chunks...")
        search_results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"type": "chunk", "doc_id": doc_id}
        )
        
        context_chunks = []
        for match in search_results.matches:
            if 'text' in match.metadata and match.score > 0.2:  # Lowered threshold
                chunk_text = match.metadata['text']
                page_num = match.metadata.get('page_number', 'Unknown')
                context_chunks.append(f"[Page {page_num}] {chunk_text}")
        
        return "\n\n".join(context_chunks)
        
    except Exception as e:
        print(f"Error during enhanced semantic search: {str(e)}")
        return ""


def get_llm_answer(context: str, query: str) -> dict:
    """
    Generate answer using OpenAI LLM based on the provided context with citations.
    
    Args:
        context (str): Retrieved context from semantic search
        query (str): User's original query
        
    Returns:
        dict: Dictionary containing 'answer', 'citations', and 'confidence'
    """
    import logging
    import json
    logger = logging.getLogger(__name__)
    
    # Construct the prompt with citation requirements
    prompt = f"""You are a helpful AI assistant. Answer the following question based ONLY on the provided context. 

IMPORTANT INSTRUCTIONS:
1. If the answer is found in the context, provide a clear answer
2. ALWAYS cite the specific page numbers where you found the information
3. Format your response as a JSON object with the following structure:
{{
    "answer": "Your detailed answer here",
    "citations": ["Page X", "Page Y"],
    "confidence": "high/medium/low",
    "explanation": "Brief explanation of why you're confident in this answer"
}}
4. If the answer is not found in the context, return:
{{
    "answer": "I could not find the answer in the provided document.",
    "citations": [],
    "confidence": "low",
    "explanation": "The required information is not present in the available context"
}}

Context:
---
{context}
---

Question: {query}

Remember to respond ONLY in valid JSON format."""

    try:
        # Make API call to OpenAI with retry mechanism
        response = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = groq_client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant that answers questions based only on the provided context and ALWAYS responds in valid JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                break
            except Exception as api_error:
                logger.warning(f"API call attempt {attempt + 1} failed: {api_error}")
                if attempt == max_retries - 1:
                    raise api_error
                import time
                time.sleep(2 ** attempt)  # Exponential backoff
        
        if not response:
            raise Exception("Failed to get response from OpenAI API")
            
        # Parse the JSON response
        response_text = response.choices[0].message.content.strip()
        
        try:
            # Try to parse as JSON
            result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ['answer', 'citations', 'confidence', 'explanation']
            for field in required_fields:
                if field not in result:
                    result[field] = "N/A" if field != 'citations' else []
                    
            logger.info(f"Generated answer with {len(result.get('citations', []))} citations")
            return result
            
        except json.JSONDecodeError:
            logger.warning("LLM response was not valid JSON, falling back to plain text")
            # Fallback to plain text response
            return {
                "answer": response_text,
                "citations": [],
                "confidence": "medium",
                "explanation": "Response generated without structured format"
            }
        
    except Exception as e:
        logger.error(f"Error generating LLM response: {str(e)}")
        return {
            "answer": "Sorry, I encountered an error while generating the response.",
            "citations": [],
            "confidence": "low",
            "explanation": f"Error occurred: {str(e)}"
        }


def check_relevance_node(state: ProcessingState) -> ProcessingState:
    """Check if the query is relevant to the document."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Checking query relevance...")
    
    try:
        index = pinecone_client.Index(PINECONE_INDEX_NAME)
        is_relevant, doc_summary = check_query_relevance(index, state["query"], state["doc_id"])
        
        state["should_process"] = is_relevant
        state["document_summary"] = doc_summary
        
        if is_relevant:
            logger.info("✅ Query is relevant to the document")
        else:
            logger.info("❌ Query is not relevant to the document")
            
    except Exception as e:
        logger.error(f"Error in relevance checking: {e}")
        state["error"] = str(e)
        state["should_process"] = False
        
    return state


def search_context_node(state: ProcessingState) -> ProcessingState:
    """Search for relevant context if query is relevant."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not state["should_process"]:
        return state
        
    logger.info("🔍 Searching for relevant context...")
    
    try:
        index = pinecone_client.Index(PINECONE_INDEX_NAME)
        context = get_relevant_context_enhanced(index, state["query"], state["doc_id"], top_k=5)
        state["context"] = context
        
        if context:
            logger.info("✅ Found relevant context")
        else:
            logger.warning("❌ No relevant context found")
            
    except Exception as e:
        logger.error(f"Error in context search: {e}")
        state["error"] = str(e)
        
    return state


def generate_answer_node(state: ProcessingState) -> ProcessingState:
    """Generate answer using LLM."""
    import logging
    logger = logging.getLogger(__name__)
    
    if not state["should_process"] or not state["context"]:
        if not state["should_process"]:
            answer_data = {
                "answer": "I'm sorry, but your question doesn't seem to be related to the content of this document. Please ask questions about the topics covered in the PDF.",
                "citations": [],
                "confidence": "high",
                "explanation": "Query determined to be irrelevant to document content"
            }
        else:
            answer_data = {
                "answer": "I couldn't find relevant information in the document to answer your question.",
                "citations": [],
                "confidence": "medium",
                "explanation": "No relevant context found in document"
            }
        state["answer"] = answer_data
        return state
        
    logger.info("🤔 Generating answer...")
    
    try:
        answer_data = get_llm_answer(state["context"], state["query"])
        state["answer"] = answer_data
        logger.info("✅ Answer generated successfully")
        
    except Exception as e:
        logger.error(f"Error in generate_answer_node: {e}")
        state["error"] = str(e)
        state["answer"] = {
            "answer": "Sorry, I encountered an error while generating the response.",
            "citations": [],
            "confidence": "low",
            "explanation": f"System error: {str(e)}"
        }
        
    return state


def should_process_query(state: ProcessingState) -> str:
    """Conditional edge function to determine if query should be processed."""
    if state["should_process"]:
        return "search_context"
    else:
        return "generate_answer"


# Create LangGraph workflow
def create_workflow():
    """Create the LangGraph workflow for processing queries."""
    workflow = StateGraph(ProcessingState)
    
    # Add nodes
    workflow.add_node("check_relevance", check_relevance_node)
    workflow.add_node("search_context", search_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    
    # Add edges
    workflow.set_entry_point("check_relevance")
    workflow.add_conditional_edges(
        "check_relevance",
        should_process_query,
        {
            "search_context": "search_context",
            "generate_answer": "generate_answer"
        }
    )
    workflow.add_edge("search_context", "generate_answer")
    workflow.add_edge("generate_answer", END)
    
    # Compile workflow
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


class PDFChatbot:
    """Enhanced PDF Chatbot with LangGraph workflow."""
    
    def __init__(self, pdf_path: str, force_reprocess: bool = False):
        self.pdf_path = pdf_path
        self.doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        self.index = None
        self.workflow = create_workflow()
        self.document_summary = ""
        self.force_reprocess = force_reprocess
        self.logger = logging.getLogger(__name__)
        self.is_newly_created = False
        
    def setup_index(self):
        """Set up Pinecone index."""
        self.logger.info("📊 Initializing Pinecone index...")
        self.index, self.is_newly_created = setup_pinecone_index(
            PINECONE_INDEX_NAME, 
            EMBEDDING_DIMENSION, 
            force_recreate=self.force_reprocess
        )
        
    def process_document(self):
        """Process the PDF document and store in vector database."""
        self.logger.info("📄 Processing PDF document...")
        
        # Check if document already exists (unless force reprocessing)
        if not self.force_reprocess and not self.is_newly_created:
            if check_document_exists(self.index, self.doc_id):
                self.logger.info(f"✅ Document {self.doc_id} already exists in index, skipping processing")
                # Get existing document summary
                try:
                    results = self.index.query(
                        vector=[0.0] * EMBEDDING_DIMENSION,
                        top_k=1,
                        include_metadata=True,
                        filter={"type": "document_summary", "doc_id": self.doc_id}
                    )
                    if results.matches:
                        self.document_summary = results.matches[0].metadata.get('text', '')
                        self.logger.info("Retrieved existing document summary")
                except Exception as e:
                    self.logger.warning(f"Could not retrieve existing summary: {e}")
                return
        
        # Extract text page by page
        pages_text = extract_text_from_pdf(self.pdf_path)
        
        if not any(text.strip() for text in pages_text.values()):
            raise ValueError("No text was extracted from the PDF. Please check if the PDF contains readable text.")
        
        # Create chunks by page
        page_chunks = get_text_chunks_by_page(pages_text)
        
        # Generate embeddings and upsert with summaries
        self.logger.info(f"🔄 Processing document with ID: {self.doc_id}")
        doc_summary, page_summaries = embed_and_upsert_with_summaries(
            self.index, pages_text, page_chunks, self.doc_id
        )
        
        self.document_summary = doc_summary
        self.logger.info("✅ Document processing completed!")
        
    def ask_question(self, query: str) -> dict:
        """Ask a question using the LangGraph workflow."""
        initial_state = ProcessingState(
            query=query,
            pdf_path=self.pdf_path,
            doc_id=self.doc_id,
            document_summary="",
            page_summaries=[],
            relevant_pages=[],
            context="",
            answer="",
            should_process=False,
            error=""
        )
        
        # Run the workflow
        config = {"configurable": {"thread_id": "pdf_chat_session"}}
        result = self.workflow.invoke(initial_state, config)
        
        return result["answer"]


if __name__ == "__main__":
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    print("🤖 Advanced PDF Chatbot with LangGraph")
    print("=" * 50)
    
    # User Input - PDF file path
    pdf_path = r"C:\Users\Chait\Downloads\Chaitanya_jhindal_PS2_final_report (1).pdf"
    
    print(f"Processing PDF: {pdf_path}")
    logger.info(f"Starting PDF chatbot with file: {pdf_path}")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        error_msg = f"❌ Error: PDF file not found at {pdf_path}"
        print(error_msg)
        logger.error(error_msg)
        print("Please check the file path and try again.")
        exit(1)
    
    try:
        # Initialize the chatbot (set force_reprocess=True for development)
        chatbot = PDFChatbot(pdf_path, force_reprocess=False)
        
        # Setup index and process document
        chatbot.setup_index()
        chatbot.process_document()
        
        # Interactive Q&A loop
        print("\n💬 PDF Chatbot is ready! Ask questions about your document.")
        print("Features:")
        print("- ✅ Smart relevance checking (saves API costs)")
        print("- ✅ Page-aware context retrieval")
        print("- ✅ Summary-enhanced responses with citations")
        print("- ✅ Persistent vector storage")
        print("- ✅ Enhanced error handling and logging")
        print("Type 'quit' or 'exit' to stop the chatbot.")
        print("-" * 50)
        
        while True:
            # Get user query
            user_query = input("\nYour question: ").strip()
            
            # Check for exit commands
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thank you for using the PDF Chatbot!")
                logger.info("User ended session")
                break
            
            if not user_query:
                print("Please enter a valid question.")
                continue
            
            print("\n🔄 Processing your question...")
            logger.info(f"Processing query: {user_query}")
            
            # Use LangGraph workflow to process the query
            answer_data = chatbot.ask_question(user_query)
            
            # Display the structured answer
            print("\n📝 Answer:")
            print("-" * 30)
            
            if isinstance(answer_data, dict):
                print(f"Answer: {answer_data.get('answer', 'No answer provided')}")
                
                citations = answer_data.get('citations', [])
                if citations:
                    print(f"\n📖 Sources: {', '.join(citations)}")
                
                confidence = answer_data.get('confidence', 'unknown')
                print(f"🎯 Confidence: {confidence}")
                
                explanation = answer_data.get('explanation', '')
                if explanation and explanation != 'N/A':
                    print(f"💡 Explanation: {explanation}")
            else:
                # Fallback for string responses
                print(answer_data)
                
            print("-" * 30)
    
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot stopped by user.")
        logger.info("Chatbot stopped by user interrupt")
    except Exception as e:
        error_msg = f"❌ An error occurred: {str(e)}"
        print(f"\n{error_msg}")
        logger.error(error_msg, exc_info=True)
        print("Please check your configuration and try again.")
