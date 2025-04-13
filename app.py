"""
API server for the Literature Review System.

This module provides a FastAPI server that exposes the Literature Review System
functionality through REST API endpoints.
"""

import os
import json
import logging
import uuid
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import LiteratureReviewSystem
from config import DEFAULT_MODEL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Literature Review System API",
    description="API for processing academic papers and generating literature reviews",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Background tasks mapping
background_tasks = {}

# Initialize Literature Review System
system = LiteratureReviewSystem()
system.initialize()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Create results directory if it doesn't exist
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Define models for API requests and responses
class PaperUploadResponse(BaseModel):
    task_id: str
    message: str

class PaperProcessingResponse(BaseModel):
    paper_ids: List[str]
    message: str

class EntityExtractionRequest(BaseModel):
    paper_ids: List[str]
    model: Optional[str] = DEFAULT_MODEL

class ReviewGenerationRequest(BaseModel):
    topic: str
    paper_ids: List[str]
    model: Optional[str] = DEFAULT_MODEL

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class GraphBuildRequest(BaseModel):
    paper_ids: List[str]
    model: Optional[str] = DEFAULT_MODEL

class PaperSearchRequest(BaseModel):
    query: str

def process_papers_task(task_id: str, folder_path: str, model: str):
    """Background task for processing papers."""
    try:
        custom_system = LiteratureReviewSystem(model=model)
        paper_ids = custom_system.process_papers(folder_path)
        background_tasks[task_id] = {
            "status": "completed",
            "result": {"paper_ids": paper_ids}
        }
    except Exception as e:
        logger.error(f"Error processing papers: {str(e)}", exc_info=True)
        background_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }

def extract_entities_task(task_id: str, paper_ids: List[str], model: str):
    """Background task for extracting entities."""
    try:
        custom_system = LiteratureReviewSystem(model=model)
        entities = custom_system.extract_entities(paper_ids)
        background_tasks[task_id] = {
            "status": "completed",
            "result": {"processed_papers": len(entities)}
        }
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
        background_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }

def build_graph_task(task_id: str, paper_ids: List[str], model: str):
    """Background task for building knowledge graph."""
    try:
        custom_system = LiteratureReviewSystem(model=model)
        stats = custom_system.build_knowledge_graph(paper_ids)
        background_tasks[task_id] = {
            "status": "completed",
            "result": stats
        }
    except Exception as e:
        logger.error(f"Error building knowledge graph: {str(e)}", exc_info=True)
        background_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }

def generate_review_task(task_id: str, topic: str, paper_ids: List[str], model: str):
    """Background task for generating literature review."""
    try:
        custom_system = LiteratureReviewSystem(model=model)
        review = custom_system.generate_review(topic, paper_ids)
        
        # Save review to file
        filename = f"literature_review_{topic.replace(' ', '_').lower()}_{task_id}.md"
        filepath = RESULTS_DIR / filename
        with open(filepath, "w") as f:
            f.write(review)
        
        background_tasks[task_id] = {
            "status": "completed",
            "result": {
                "filename": filename,
                "length": len(review)
            }
        }
    except Exception as e:
        logger.error(f"Error generating review: {str(e)}", exc_info=True)
        background_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }

@app.get("/", include_in_schema=False)
def read_root():
    """Root endpoint."""
    return {"message": "Literature Review System API"}

@app.post("/papers/upload", response_model=PaperUploadResponse)
async def upload_papers(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """
    Upload PDF papers for processing.
    
    Args:
        files: List of PDF files to upload
        
    Returns:
        Task ID for tracking the processing job
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create a temporary directory for this batch
    batch_dir = UPLOAD_DIR / f"batch_{uuid.uuid4().hex}"
    batch_dir.mkdir(exist_ok=True)
    
    try:
        # Save uploaded files
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue
                
            file_path = batch_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(await file.read())
                
        # Generate a task ID
        task_id = uuid.uuid4().hex
        
        # Store task in background tasks
        background_tasks[task_id] = {"status": "processing"}
        
        # Start background processing
        background_tasks.add_task(
            process_papers_task,
            task_id=task_id,
            folder_path=str(batch_dir),
            model=DEFAULT_MODEL
        )
        
        return PaperUploadResponse(
            task_id=task_id,
            message=f"Processing {len(files)} papers in the background"
        )
    except Exception as e:
        logger.error(f"Error uploading papers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/papers/process_url", response_model=PaperUploadResponse)
async def process_url(
        background_tasks: BackgroundTasks,
        url: str = Form(...)
):
    """
    Process a paper from a URL.

    Args:
        url: URL to a PDF file

    Returns:
        Task ID for tracking the processing job
    """
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")

    # Generate a task ID
    task_id = uuid.uuid4().hex

    # Store task in background tasks
    background_tasks[task_id] = {"status": "processing"}

    # Define the background task function
    async def process_url_task():
        try:
            system = LiteratureReviewSystem()
            paper_id = system.process_paper_from_url(url)
            background_tasks[task_id] = {
                "status": "completed",
                "result": {"paper_id": paper_id}
            }
        except Exception as e:
            logger.error(f"Error processing URL: {str(e)}", exc_info=True)
            background_tasks[task_id] = {
                "status": "failed",
                "error": str(e)
            }

    # Start background processing
    background_tasks.add_task(process_url_task)

    return PaperUploadResponse(
        task_id=task_id,
        message=f"Processing paper from URL in the background"
    )


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task_status(task_id: str):
    """
    Get the status of a background task.
    
    Args:
        task_id: ID of the task to check
        
    Returns:
        Current status of the task
    """
    if task_id not in background_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = background_tasks[task_id]
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info["status"],
        result=task_info.get("result"),
        error=task_info.get("error")
    )

@app.post("/papers/entities", response_model=PaperUploadResponse)
def extract_entities(
    request: EntityExtractionRequest,
    background_tasks: BackgroundTasks
):
    """
    Extract entities from papers.
    
    Args:
        request: Paper IDs and model to use
        
    Returns:
        Task ID for tracking the extraction job
    """
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="No paper IDs provided")
    
    # Generate a task ID
    task_id = uuid.uuid4().hex
    
    # Store task in background tasks
    background_tasks[task_id] = {"status": "processing"}
    
    # Start background processing
    background_tasks.add_task(
        extract_entities_task,
        task_id=task_id,
        paper_ids=request.paper_ids,
        model=request.model
    )
    
    return PaperUploadResponse(
        task_id=task_id,
        message=f"Extracting entities from {len(request.paper_ids)} papers in the background"
    )

@app.post("/graph/build", response_model=PaperUploadResponse)
def build_knowledge_graph(
    request: GraphBuildRequest,
    background_tasks: BackgroundTasks
):
    """
    Build knowledge graph from papers.
    
    Args:
        request: Paper IDs and model to use
        
    Returns:
        Task ID for tracking the graph building job
    """
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="No paper IDs provided")
    
    # Generate a task ID
    task_id = uuid.uuid4().hex
    
    # Store task in background tasks
    background_tasks[task_id] = {"status": "processing"}
    
    # Start background processing
    background_tasks.add_task(
        build_graph_task,
        task_id=task_id,
        paper_ids=request.paper_ids,
        model=request.model
    )
    
    return PaperUploadResponse(
        task_id=task_id,
        message=f"Building knowledge graph for {len(request.paper_ids)} papers in the background"
    )

@app.post("/reviews/generate", response_model=PaperUploadResponse)
def generate_review(
    request: ReviewGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate a literature review.
    
    Args:
        request: Review topic, paper IDs, and model to use
        
    Returns:
        Task ID for tracking the review generation job
    """
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="No paper IDs provided")
    
    if not request.topic:
        raise HTTPException(status_code=400, detail="No topic provided")
    
    # Generate a task ID
    task_id = uuid.uuid4().hex
    
    # Store task in background tasks
    background_tasks[task_id] = {"status": "processing"}
    
    # Start background processing
    background_tasks.add_task(
        generate_review_task,
        task_id=task_id,
        topic=request.topic,
        paper_ids=request.paper_ids,
        model=request.model
    )
    
    return PaperUploadResponse(
        task_id=task_id,
        message=f"Generating literature review on '{request.topic}' with {len(request.paper_ids)} papers in the background"
    )

@app.get("/reviews/download/{task_id}")
def download_review(task_id: str):
    """
    Download a generated literature review.
    
    Args:
        task_id: ID of the review generation task
        
    Returns:
        Literature review file
    """
    if task_id not in background_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = background_tasks[task_id]
    
    if task_info["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Review generation not completed yet (status: {task_info['status']})")
    
    if "result" not in task_info or "filename" not in task_info["result"]:
        raise HTTPException(status_code=404, detail="Review file not found")
    
    filename = task_info["result"]["filename"]
    filepath = RESULTS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Review file not found")
    
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="text/markdown"
    )

@app.post("/papers/search")
def search_papers(request: PaperSearchRequest):
    """
    Search for papers by title or abstract.
    
    Args:
        request: Search query
        
    Returns:
        List of matching papers
    """
    if not request.query:
        raise HTTPException(status_code=400, detail="No search query provided")
    
    try:
        results = system.supabase.search_papers(request.query)
        return {"papers": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Error searching papers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers")
def list_papers(limit: int = Query(100, ge=1, le=1000)):
    """
    List all papers in the database.
    
    Args:
        limit: Maximum number of papers to return
        
    Returns:
        List of papers
    """
    try:
        # This is a simplified approach, in a real system you'd want pagination
        papers = system.supabase.client.table("papers").select("*").limit(limit).execute()
        return {"papers": papers.data, "count": len(papers.data)}
    except Exception as e:
        logger.error(f"Error listing papers: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/papers/{paper_id}")
def get_paper(paper_id: str):
    """
    Get details of a specific paper.
    
    Args:
        paper_id: ID of the paper
        
    Returns:
        Paper details
    """
    try:
        paper = system.supabase.get_paper(paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="Paper not found")
        
        entities = system.supabase.get_paper_entities(paper_id)
        
        return {
            "paper": paper,
            "entities": entities
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting paper: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
def shutdown_event():
    """Clean up on shutdown."""
    system.close()
    logger.info("API server shutdown")

def start():
    """Start the API server."""
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()