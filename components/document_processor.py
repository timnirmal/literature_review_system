"""
Document Processor component for the Literature Review System.

This module handles the extraction of text and metadata from academic papers,
primarily in PDF format, and stores them in the database.
"""

import os
import uuid
import json
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import io

from gemini_parser import DocumentProcessor as GeminiParser

from config import DEFAULT_MODEL, GEMINI_API_KEY, USE_GEMINI_PARSER
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
        
        # Initialize Gemini Parser if API key is available
        if GEMINI_API_KEY:
            self.parser = GeminiParser(api_key=GEMINI_API_KEY)
            logger.info("Gemini Parser initialized")
        else:
            self.parser = None
            logger.info("Gemini API key not provided. Using PyPDF2 for text extraction.")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file using PyPDF2.
        For Gemini Parser, we use process_and_store_paper directly.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        logger.info(f"Extracting text from {pdf_path}")
        
        # If Gemini is enabled, we'll use it at a higher level in process_and_store_paper
        # This method is now primarily for PyPDF2 extraction
        try:
            return self._extract_text_with_pypdf2(pdf_path)
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def _extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """
        Extract text using PyPDF2.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        import PyPDF2

        basename = os.path.basename(pdf_path)
        if basename.startswith("._"):
            logger.warning(f"Skipping metadata file: {pdf_path}")
            return ""
        
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                input_buffer = io.BytesIO(file.read())
                pdf_reader = PyPDF2.PdfReader(input_buffer)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    
                    # Log progress for large PDFs
                    if (page_num + 1) % 10 == 0:
                        logger.debug(f"Processed {page_num + 1} pages from {pdf_path}")
            
            logger.info(f"Extracted {len(text)} characters from {pdf_path} using PyPDF2")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2 from {pdf_path}: {str(e)}")
            return ""
    
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
        
        # Extract text - use appropriate method based on configuration
        full_text = ""
        
        if self.parser and USE_GEMINI_PARSER:
            try:
                logger.info(f"Using Gemini Parser for {pdf_path}")
                
                # Create a temporary output directory for saving the processed file
                output_dir = Path(tempfile.mkdtemp(prefix="gemini_output_"))
                file_path = Path(pdf_path)

                logger.info(f"Processing {file_path} with Gemini Parser")
                
                # Process with Gemini and save to output directory
                self.parser.process_folder(
                    folder_path=file_path.parent,
                    output_dir=output_dir,
                    out_ext="md"
                )

                logger.info(f"Gemini Parser output saved to {output_dir}")
                
                # Look for the corresponding output file
                expected_output = output_dir / f"{file_path.stem}.md"
                if expected_output.exists():
                    full_text = expected_output.read_text(encoding="utf-8")
                    logger.info(f"Extracted {len(full_text)} characters using Gemini Parser output file")
                else:
                    # Try direct processing as fallback
                    result = self.parser.process_file(file_path)
                    if isinstance(result, dict) and result.get('text'):
                        full_text = result.get('text')
                        logger.info(f"Extracted {len(full_text)} characters using Gemini Parser direct processing")
                    else:
                        logger.warning(f"Gemini Parser didn't produce expected output for {pdf_path}")
                        # Fall back to PyPDF2
                        full_text = self._extract_text_with_pypdf2(pdf_path)
                
                # Clean up temporary directory
                for f in output_dir.glob("*"):
                    try:
                        f.unlink()
                    except:
                        pass
                try:
                    output_dir.rmdir()
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Error using Gemini Parser: {str(e)}")
                # Fall back to PyPDF2
                full_text = self._extract_text_with_pypdf2(pdf_path)
        else:
            # Use PyPDF2
            full_text = self._extract_text_with_pypdf2(pdf_path)
        
        if not full_text:
            logger.error(f"Failed to extract text from {pdf_path}")
            raise ValueError(f"Could not extract text from {pdf_path}")
        
        # Extract metadata
        metadata = self.extract_metadata_with_llm(full_text)

        logger.info(f"Extracted metadata: {json.dumps(metadata)[:100]}... xxx")
        
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
    
    def process_from_url(self, url: str) -> str:
        """
        Process a paper from a URL.
        
        Args:
            url: URL to the PDF file
            
        Returns:
            ID of the processed paper
        """
        logger.info(f"Processing paper from URL: {url}")
        
        if not self.parser:
            raise ValueError("Gemini Parser not initialized. Cannot process from URL.")
        
        try:
            # Create a temporary directory for output
            output_dir = Path(tempfile.mkdtemp(prefix="gemini_url_output_"))
            
            # Use Gemini Parser to process from URL
            result = self.parser.process_from_url(url)
            
            full_text = ""
            
            # Check if we got a direct result
            if isinstance(result, dict) and result.get('text'):
                full_text = result.get('text')
                logger.info(f"Extracted {len(full_text)} characters from URL using direct result")
            else:
                # Fallback approach: look for output files in the standard location
                # This is based on how Gemini parser might save output
                # In real usage, you might need to adjust this based on actual behavior
                
                # Look for any markdown files created
                md_files = list(output_dir.glob("*.md"))
                if md_files:
                    full_text = md_files[0].read_text(encoding="utf-8")
                    logger.info(f"Extracted {len(full_text)} characters from URL using output file")
            
            # Clean up
            for f in output_dir.glob("*"):
                try:
                    f.unlink()
                except:
                    pass
            try:
                output_dir.rmdir()
            except:
                pass
                
            if not full_text:
                logger.error(f"Gemini Parser couldn't extract text from URL {url}")
                raise ValueError(f"Could not extract text from URL {url}")
            
            # Extract metadata
            metadata = self.extract_metadata_with_llm(full_text)

            logger.info(f"Extracted metadata from URL: {json.dumps(metadata)[:100]}... yyy")
            
            # Generate a unique ID
            paper_id = str(uuid.uuid4())
            
            # Extract filename from URL
            source_file = url.split('/')[-1] if '/' in url else url
            
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
                "source_file": source_file,
                "source_url": url
            }
            
            # Store in database
            self.supabase.store_paper(paper_data)
            
            logger.info(f"Processed paper from URL '{metadata.get('title')}' → ID: {paper_id}")
            return paper_id
            
        except Exception as e:
            logger.error(f"Error processing paper from URL {url}: {str(e)}")
            raise
    
    def batch_process_papers(self, pdf_folder_path: str) -> List[str]:
        """
        Process all PDFs in a folder.
        
        Args:
            pdf_folder_path: Path to folder containing PDF files
            
        Returns:
            List of processed paper IDs
        """
        logger.info(f"Batch processing papers from {pdf_folder_path}")
        
        # Check if folder exists
        if not os.path.isdir(pdf_folder_path):
            logger.error(f"Folder not found: {pdf_folder_path}")
            raise FileNotFoundError(f"Folder not found: {pdf_folder_path}")
        
        # Get list of PDF files in the folder
        pdf_files = [f for f in os.listdir(pdf_folder_path) if f.lower().endswith('.pdf')]
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_folder_path}")
            return []
        
        paper_ids = []
        
        # Process each file individually regardless of whether Gemini is used
        # This is more reliable than using process_folder_with_gemini
        for filename in pdf_files:
            try:
                file_path = os.path.join(pdf_folder_path, filename)
                basename = os.path.basename(file_path)
                if basename.startswith("._"):
                    logger.warning(f"Skipping metadata file: {file_path}")
                    continue
                paper_id = self.process_and_store_paper(file_path)
                paper_ids.append(paper_id)
                logger.info(f"Successfully processed {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                # Continue with next file
        
        logger.info(f"Batch processing complete. Processed {len(paper_ids)} papers.")
        return paper_ids
    
    def list_parser_caches(self) -> List[Dict[str, Any]]:
        """
        List all document caches from Gemini Parser.
        
        Returns:
            List of cache information
        """
        if not self.parser:
            raise ValueError("Gemini Parser not initialized.")
        
        try:
            caches = self.parser.list_caches()
            return caches
        except Exception as e:
            logger.error(f"Error listing caches: {str(e)}")
            return []
    
    def delete_parser_cache(self, cache_id: str) -> bool:
        """
        Delete a document cache from Gemini Parser.
        
        Args:
            cache_id: ID of the cache to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.parser:
            raise ValueError("Gemini Parser not initialized.")
        
        try:
            self.parser.delete_cache(cache_id)
            logger.info(f"Deleted cache {cache_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting cache {cache_id}: {str(e)}")
            return False