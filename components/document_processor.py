"""
Document Processor component for the Literature Review System.

This module handles the extraction of text and metadata from academic papers,
primarily in PDF format, and stores them in the database.
"""

import os
import uuid
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import PyPDF2

from config import DEFAULT_MODEL
from database.supabase_manager import SupabaseManager
from models import call_model, parse_json_response
from prompts import METADATA_EXTRACTION_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes academic papers and extracts metadata and content."""
    
    def __init__(self, supabase_manager: SupabaseManager, model: str = DEFAULT_MODEL):
        """
        Initialize the document processor.
        
        Args:
            supabase_manager: SupabaseManager instance for database operations
            model: LLM model to use for metadata extraction
        """
        self.supabase = supabase_manager
        self.model = model
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        logger.info(f"Extracting text from {pdf_path}")
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    
                    # Log progress for large PDFs
                    if (page_num + 1) % 10 == 0:
                        logger.debug(f"Processed {page_num + 1} pages from {pdf_path}")
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            raise
        
        logger.info(f"Extracted {len(text)} characters from {pdf_path}")
        return text
    
    def extract_metadata_with_llm(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from paper text using LLM.
        
        Args:
            text: Paper text content
            
        Returns:
            Dictionary containing extracted metadata
        """
        logger.info("Extracting metadata using LLM")
        
        # Use the first 15K characters which should contain all metadata
        prompt = METADATA_EXTRACTION_PROMPT.format(text=text[:15000])
        
        try:
            response = call_model(
                prompt=prompt,
                model=self.model,
                system_prompt="You are a scientific paper metadata extractor.",
                response_format={"type": "json_object"}
            )
            
            metadata = parse_json_response(response)
            logger.info(f"Extracted metadata: {json.dumps(metadata)[:100]}...")
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {str(e)}")
            # Return fallback metadata
            return {
                "title": "Unknown Title",
                "authors": [],
                "year": None,
                "venue": "Unknown Venue",
                "doi": None,
                "abstract": ""
            }
    
    def process_and_store_paper(self, pdf_path: str) -> str:
        """
        Process a paper and store it in the database.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ID of the processed paper
        """
        logger.info(f"Processing paper: {pdf_path}")
        
        # Extract text
        full_text = self.extract_text_from_pdf(pdf_path)
        
        # Extract metadata
        metadata = self.extract_metadata_with_llm(full_text)
        
        # Generate a unique ID
        paper_id = str(uuid.uuid4())
        
        # Prepare paper data
        paper_data = {
            "id": paper_id,
            "title": metadata.get("title"),
            "authors": metadata.get("authors", []),
            "year": metadata.get("year"),
            "venue": metadata.get("venue"),
            "doi": metadata.get("doi"),
            "abstract": metadata.get("abstract"),
            "full_text": full_text,
            "processed": False,
            "source_file": os.path.basename(pdf_path)
        }
        
        # Store in database
        self.supabase.store_paper(paper_data)
        
        logger.info(f"Processed paper '{metadata.get('title')}' → ID: {paper_id}")
        return paper_id
    
    def batch_process_papers(self, pdf_folder_path: str) -> List[str]:
        """
        Process all PDFs in a folder.
        
        Args:
            pdf_folder_path: Path to folder containing PDF files
            
        Returns:
            List of processed paper IDs
        """
        logger.info(f"Batch processing papers from {pdf_folder_path}")
        paper_ids = []
        
        # Check if folder exists
        if not os.path.isdir(pdf_folder_path):
            logger.error(f"Folder not found: {pdf_folder_path}")
            raise FileNotFoundError(f"Folder not found: {pdf_folder_path}")
        
        # Process each PDF
        pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for filename in pdf_files:
            try:
                file_path = os.path.join(pdf_folder_path, filename)
                paper_id = self.process_and_store_paper(file_path)
                paper_ids.append(paper_id)
                logger.info(f"Successfully processed {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                # Continue with next file
        
        logger.info(f"Batch processing complete. Processed {len(paper_ids)} papers.")
        return paper_ids