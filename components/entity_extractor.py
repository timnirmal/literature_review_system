"""
Entity Extractor component for the Literature Review System.

This module extracts key entities (concepts, methods, findings, technologies, datasets)
from academic papers using LLMs.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional

from config import DEFAULT_MODEL, CHUNK_SIZE
from database.supabase_manager import SupabaseManager
from models import call_model, parse_json_response
from prompts import ENTITY_EXTRACTION_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extracts key entities from academic papers."""
    
    def __init__(self, supabase_manager: SupabaseManager, model: str = DEFAULT_MODEL):
        """
        Initialize the entity extractor.
        
        Args:
            supabase_manager: SupabaseManager instance for database operations
            model: LLM model to use for entity extraction
        """
        self.supabase = supabase_manager
        self.model = model
    
    def extract_entities_from_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        Extract key entities from a paper using LLM.
        
        Args:
            paper_id: ID of the paper to analyze
            
        Returns:
            Dictionary of extracted entities by category
        """
        logger.info(f"Extracting entities from paper {paper_id}")
        
        # Retrieve paper from database
        paper = self.supabase.get_paper(paper_id)
        
        if not paper:
            err_msg = f"Paper with ID {paper_id} not found"
            logger.error(err_msg)
            raise ValueError(err_msg)
        
        # Extract entities using LLM
        all_entities = {
            "concepts": [],
            "methods": [],
            "findings": [],
            "technologies": [],
            "datasets": []
        }
        
        # Split text into chunks if too long
        text_chunks = self._chunk_text(paper["full_text"], CHUNK_SIZE)
        logger.info(f"Processing paper '{paper['title']}' in {len(text_chunks)} chunks")
        
        # Process each chunk
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{len(text_chunks)} for paper {paper_id}")
            
            # Format prompt
            prompt = ENTITY_EXTRACTION_PROMPT.format(
                title=paper["title"],
                abstract=paper["abstract"],
                text=chunk
            )
            
            try:
                # Call LLM
                response = call_model(
                    prompt=prompt,
                    model=self.model,
                    system_prompt="You are a scientific entity extractor. Extract key entities from academic papers accurately.",
                    response_format={"type": "json_object"}
                )
                
                # Parse response
                chunk_entities = parse_json_response(response)
                
                # Merge entities from this chunk
                for category in all_entities:
                    if category in chunk_entities and isinstance(chunk_entities[category], list):
                        all_entities[category].extend(chunk_entities[category])
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error extracting entities from chunk {i+1}: {str(e)}")
                # Continue with next chunk
        
        # Deduplicate entities by name within each category
        for category in all_entities:
            unique_entities = {}
            for entity in all_entities[category]:
                # Use name as key for deduplication
                name = entity.get("name", "")
                if name and name not in unique_entities:
                    unique_entities[name] = entity
            
            # Replace with deduplicated list
            all_entities[category] = list(unique_entities.values())
            
            logger.info(f"Extracted {len(all_entities[category])} unique {category}")
        
        # Store entities in database
        self.supabase.store_paper_entities(paper_id, all_entities)
        
        # Mark paper as processed
        self.supabase.update_paper(paper_id, {"processed": True})
        
        logger.info(f"Completed entity extraction for paper {paper_id}")
        return all_entities
    
    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """
        Split text into chunks of approximately equal size.
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs to preserve context
        for paragraph in text.split("\n"):
            # If adding this paragraph would exceed chunk size, start a new chunk
            if len(current_chunk) + len(paragraph) > chunk_size:
                chunks.append(current_chunk)
                current_chunk = paragraph + "\n"
            else:
                current_chunk += paragraph + "\n"
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def batch_extract_entities(self, paper_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities from multiple papers.
        
        Args:
            paper_ids: List of paper IDs to process
            
        Returns:
            Dictionary mapping paper IDs to their extracted entities
        """
        logger.info(f"Batch extracting entities from {len(paper_ids)} papers")
        results = {}
        
        for paper_id in paper_ids:
            try:
                entities = self.extract_entities_from_paper(paper_id)
                results[paper_id] = entities
                logger.info(f"Successfully extracted entities from paper {paper_id}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error extracting entities from paper {paper_id}: {str(e)}")
                # Continue with next paper
        
        logger.info(f"Batch entity extraction complete. Processed {len(results)}/{len(paper_ids)} papers.")
        return results
    
    def process_unprocessed_papers(self, limit: int = 10) -> List[str]:
        """
        Process papers that haven't had entities extracted yet.
        
        Args:
            limit: Maximum number of papers to process
            
        Returns:
            List of processed paper IDs
        """
        logger.info(f"Processing unprocessed papers (limit: {limit})")
        
        # Get unprocessed papers
        papers = self.supabase.get_unprocessed_papers(limit)
        paper_ids = [paper["id"] for paper in papers]
        
        if not paper_ids:
            logger.info("No unprocessed papers found.")
            return []
        
        logger.info(f"Found {len(paper_ids)} unprocessed papers. Extracting entities...")
        
        # Extract entities
        results = self.batch_extract_entities(paper_ids)
        
        logger.info(f"Processed {len(results)} unprocessed papers")
        return list(results.keys())