from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import os
import shutil
from typing import List
import uvicorn
import json
import threading
from collections import deque
from datetime import datetime

# Import our custom modules
from backend.ingestion import ingest_documents
from backend.embeddings import chunk_text, generate_embeddings
from backend.retrieval import initialize_vector_db, add_embeddings_to_db, query_vector_db
from backend.synthesis import generate_answer, _is_summarization_query
import pathlib
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Log streaming system
class LogStream:
    def __init__(self, max_logs=1000):
        self.logs = deque(maxlen=max_logs)
        self.lock = threading.Lock()

    def add_log(self, level, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message
        }
        with self.lock:
            self.logs.append(log_entry)

    def get_logs(self, limit=100):
        with self.lock:
            return list(self.logs)[-limit:]

    def clear_logs(self):
        with self.lock:
            self.logs.clear()

# Global log stream instance
log_stream = LogStream()

# Custom log handler to capture logs
class StreamlitLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        level = record.levelname
        log_stream.add_log(level, log_entry)

# Set OpenAI API key from environment variable only
# Do not hardcode API keys - rely solely on environment variables
# Users should set OPENAI_API_KEY in their environment before running the application

# Initialize FastAPI app
app = FastAPI(
    title="DocuMind API",
    description="Knowledge-Base Search Engine API",
    version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    include_sources: bool = False
    llm_provider: str = "local"  # openai, ollama, huggingface, local
    model: str = "gpt-3.5-turbo"  # model name for the selected provider (not used for local)

# Global database object
vector_db = None

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    global vector_db
    try:
        # Add custom log handler for streaming
        streamlit_handler = StreamlitLogHandler()
        streamlit_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        streamlit_handler.setFormatter(formatter)
        logging.getLogger().addHandler(streamlit_handler)

        # Check if OpenAI API key is set
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            logger.info("OpenAI API key is configured successfully")
        else:
            logger.warning("OpenAI API key is not set")

        vector_db = initialize_vector_db("faiss", "data/vector_store/")
        logger.info("Vector database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector database: {str(e)}")
        raise


@app.get("/status")
async def get_status():
    """
    Health check endpoint.

    Returns:
        dict: Status information about the API and database.

    Example:
        GET /status
        Response: {"status": "healthy", "database_initialized": true}
    """
    global vector_db

    db_status = vector_db is not None
    if vector_db and vector_db["type"] == "faiss":
        db_status = db_status and (vector_db["index"] is not None)

    return {
        "status": "healthy",
        "database_initialized": db_status,
        "database_type": vector_db["type"] if vector_db else None
    }


@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple documents or pasted text.

    This endpoint accepts multiple files (PDF, DOCX, TXT) or pasted text sent as pseudo-files,
    extracts text, generates embeddings, and stores them in the vector database.

    Args:
        files: List of uploaded files or pseudo-files (for pasted text).

    Returns:
        dict: Upload results with success/failure counts.

    Raises:
        HTTPException: If no files are provided or processing fails.

    Example:
        POST /upload
        Content-Type: multipart/form-data
        Body: files=file1.pdf&files=pasted_text.txt

        Response: {
            "successful_uploads": 2,
            "total_chunks": 15
        }
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    global vector_db
    if not vector_db:
        raise HTTPException(status_code=500, detail="Vector database not initialized")

    # Ensure upload directory exists
    upload_dir = pathlib.Path("data/uploaded_docs")
    upload_dir.mkdir(parents=True, exist_ok=True)

    successful_uploads = 0
    total_chunks = 0

    try:
        all_chunks = []

        # Process each file
        for file in files:
            if not file.filename:
                continue

            try:
                # Read file content
                content = await file.read()

                # Detect pasted text (pseudo-file)
                if file.filename == "pasted_text.txt":
                    text = content.decode("utf-8")
                    logger.info("Processing pasted text")
                else:
                    # Sanitize filename to prevent path traversal
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", file.filename)
                    file_path = upload_dir / safe_name
                    with open(file_path, "wb") as buffer:
                        buffer.write(content)

                    # Process single document
                    documents = ingest_documents([file_path])
                    if not documents:
                        logger.warning(f"Could not process file: {file.filename}")
                        continue

                    text = documents[0]["content"]
                    logger.info(f"Processed uploaded file: {file.filename}")

                # Split into chunks
                chunks = chunk_text(text, chunk_size=500, overlap=50)
                all_chunks.extend(chunks)
                successful_uploads += 1
                logger.info(f"Generated {len(chunks)} chunks from {file.filename}")

            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No text chunks could be generated from any files")

        # Generate embeddings for all chunks
        embeddings = generate_embeddings(all_chunks, method="sentence-transformers")

        # If vector_db isn't initialized for some reason, try to initialize it now
        if not vector_db:
            try:
                vector_db = initialize_vector_db("faiss", "data/vector_store/")
            except Exception as e:
                logger.error(f"Failed to initialize vector DB during upload: {e}")
                raise HTTPException(status_code=500, detail=f"Vector DB initialization failed: {e}")

        # Add to vector database
        add_embeddings_to_db(all_chunks, embeddings, vector_db)

        logger.info(f"Successfully processed {successful_uploads} files with {len(all_chunks)} total chunks")

        return {
            "successful_uploads": successful_uploads,
            "total_chunks": len(all_chunks)
        }

    except Exception as e:
        logger.error(f"Error processing uploads: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process files: {str(e)}")


@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query the knowledge base and generate an answer.

    This endpoint accepts a natural language query, retrieves relevant document chunks,
    and generates a synthesized answer using the language model.

    Args:
        request: QueryRequest object with query text and include_sources flag.

    Returns:
        dict: Query results with answer and optional sources.

    Raises:
        HTTPException: If query is empty or processing fails.

    Example:
        POST /query
        Content-Type: application/json
        Body: {"query": "What is machine learning?", "include_sources": true}

        Response: {
            "query": "What is machine learning?",
            "answer": "Machine learning is a subset of AI...",
            "sources_included": true
        }
    """
    query = request.query
    include_sources = request.include_sources

    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    global vector_db
    if not vector_db:
        raise HTTPException(status_code=500, detail="Vector database not initialized")

    # Generate embedding for query
    try:
        query_chunks = [query]  # Treat query as a single chunk
        query_embedding = generate_embeddings(query_chunks, method="sentence-transformers")[0]
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return {
            "query": query,
            "answer": f"Error generating query embeddings: {str(e)}. Please try again later.",
            "error": True
        }

    # Retrieve relevant chunks - get more chunks for implementation/architecture questions
    try:
        # More sophisticated query type detection for better retrieval
        query_lower = query.lower()
        query_type = "general"
        top_k = 5
        
        # Define categories of questions with related keywords
        query_categories = {
            "implementation": ["implement", "architecture", "design", "structure", "component", "stack", "technical", "build", "develop", "construct"],
            "comprehensive": ["explain", "elaborate", "detail", "comprehensive", "thorough", "complete", "full", "in-depth", "outline"],
            "technical": ["technology", "framework", "language", "platform", "protocol", "system", "infrastructure"],
            "process": ["step", "process", "procedure", "workflow", "pipeline", "approach", "method"]
        }
        
        # Check which category this query falls into
        for category, keywords in query_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                query_type = category
                # Assign appropriate chunk count based on query type
                if category == "implementation" or category == "comprehensive":
                    top_k = 12  # Get lots of chunks for implementation questions
                elif category == "technical":
                    top_k = 8   # Get more chunks for technical questions
                elif category == "process":
                    top_k = 8   # Get more chunks for process questions
                break
                
        logger.info(f"Query type: {query_type}, retrieving {top_k} chunks")
        
        retrieved_chunks = query_vector_db(query_embedding, vector_db, top_k=top_k)
    except Exception as e:
        logger.error(f"Error retrieving chunks: {str(e)}")
        return {
            "query": query,
            "answer": f"Error retrieving relevant information: {str(e)}. Please try again later.",
            "error": True
        }

    if not retrieved_chunks:
        return {
            "query": query,
            "answer": "No relevant information found in the documents.",
            "sources_included": False
        }

    # Check if it's a summarization request
    is_summarization = _is_summarization_query(query)
    
    # Generate answer - always use local provider to avoid API quota issues
    try:
        # Force local provider to avoid API quota issues
        llm_provider = "gpt-4o"  # Use local
        logger.info(f"Using local text generation provider (overriding {request.llm_provider})")
        
        answer = generate_answer(
            query=query, 
            retrieved_chunks=retrieved_chunks, 
            model=request.model,
            include_sources=include_sources,
            llm_provider=llm_provider  # Use local provider
        )
        
        # If the answer seems too short, try to enrich it
        if len(answer) < 100 and len(retrieved_chunks) > 1:
            logger.warning(f"Answer too short ({len(answer)} chars). Attempting to generate a more comprehensive response.")
            # Try again with more detailed processing
            enriched_answer = "Based on the document information:\n\n"
            for i, chunk in enumerate(retrieved_chunks[:5]):
                enriched_answer += f"Point {i+1}: {chunk}\n\n"
            
            # Use the enriched answer if it's significantly longer
            if len(enriched_answer) > len(answer) * 2:
                answer = enriched_answer
                logger.info(f"Using enriched answer: {len(answer)} characters")
        
        success = True
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        # Create a simple answer from the retrieved chunks
        answer = f"I encountered an error while generating a proper answer ({str(e)}), but here are the most relevant excerpts from your documents:\n\n"
        for i, chunk in enumerate(retrieved_chunks[:3]):
            answer += f"Excerpt {i+1}: {chunk[:150]}...\n\n"
        success = False

    logger.info(f"Query processed: '{query[:50]}...' -> Answer length: {len(answer)}")

    return {
        "query": query,
        "answer": answer,
        "sources_included": include_sources,
        "chunks_retrieved": len(retrieved_chunks),
        "is_summarization": is_summarization,
        "llm_provider": request.llm_provider,
        "model": request.model,
        "success": success
    }


@app.get("/logs")
async def get_logs(limit: int = 100):
    """
    Get recent backend logs for display in the UI.

    Args:
        limit: Maximum number of logs to return (default: 100)

    Returns:
        dict: Recent logs with timestamps and levels

    Example:
        GET /logs?limit=50
        Response: {
            "logs": [
                {"timestamp": "2025-10-15 21:35:40", "level": "INFO", "message": "Server started"},
                {"timestamp": "2025-10-15 21:35:58", "level": "INFO", "message": "Document uploaded"}
            ]
        }
    """
    logs = log_stream.get_logs(limit)
    return {"logs": logs}


@app.post("/logs/clear")
async def clear_logs():
    """
    Clear all backend logs.

    Returns:
        dict: Confirmation message

    Example:
        POST /logs/clear
        Response: {"message": "Logs cleared successfully"}
    """
    log_stream.clear_logs()
    return {"message": "Logs cleared successfully"}


# Example usage in comments
"""
Example API Usage:

1. Health Check:
   GET http://localhost:8000/status

2. Upload Files (using curl):
   curl -X POST "http://localhost:8000/upload" \
        -F "files=@document1.pdf" \
        -F "files=@document2.docx"

3. Query Documents (using curl):
   curl -X POST "http://localhost:8000/query" \
        -d "query=What is machine learning?" \
        -d "include_sources=true"

4. Using Python requests:
   import requests

   # Upload
   files = {'files': open('document.pdf', 'rb')}
   response = requests.post('http://localhost:8000/upload', files=files)

   # Query
   data = {'query': 'What is AI?', 'include_sources': 'true'}
   response = requests.post('http://localhost:8000/query', data=data)
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)