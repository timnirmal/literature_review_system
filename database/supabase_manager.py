"""
Supabase database manager for the Literature Review System.

This module handles all Supabase database operations, providing a unified
interface for storing and retrieving documents, entities, and generated reviews.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_KEY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupabaseManager:
    """Supabase database manager for storing papers, entities, and reviews."""
    
    def __init__(self, url: str = SUPABASE_URL, key: str = SUPABASE_KEY):
        """
        Initialize Supabase client.
        
        Args:
            url: Supabase project URL
            key: Supabase API key
        """
        self.url = url
        self.key = key
        self.client = None
        self.connect()
    
    def connect(self) -> None:
        """Establish connection to Supabase."""
        try:
            self.client = create_client(self.url, self.key)
            logger.info("Connected to Supabase")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {str(e)}")
            raise
    
    def initialize_tables(self) -> None:
        """
        Initialize Supabase tables if they don't exist.
        
        Note: In a production environment, this would typically be
        handled by Supabase migrations rather than programmatically.
        """
        # This is a placeholder since Supabase schema is typically
        # defined through the Supabase dashboard or migrations
        logger.info("Supabase tables initialization would happen here")
        pass
    
    def store_paper(self, paper_data: Dict[str, Any]) -> str:
        """
        Store a paper in the database.
        
        Args:
            paper_data: Dictionary containing paper metadata and content
            
        Returns:
            ID of the stored paper
        """
        # Add timestamp
        if "created_at" not in paper_data:
            paper_data["created_at"] = datetime.now().isoformat()
        
        if "processed" not in paper_data:
            paper_data["processed"] = False
        
        # check if paper already exists
        existing_paper = self.client.table("papers").select("*").eq("title", paper_data["title"]).execute()
        if existing_paper.data:
            logger.warning(f"Paper '{paper_data['title']}' already exists in the database")
            return existing_paper.data[0]["id"]

        result = self.client.table("papers").insert(paper_data).execute()
        
        if not result.data:
            raise ValueError("Failed to store paper in Supabase")
        
        paper_id = result.data[0]["id"]
        logger.info(f"Stored paper '{paper_data.get('title', 'Unknown')}' with ID {paper_id}")
        
        return paper_id
    
    def update_paper(self, paper_id: str, updates: Dict[str, Any]) -> None:
        """
        Update a paper's metadata.
        
        Args:
            paper_id: ID of the paper to update
            updates: Dictionary of fields to update
        """
        self.client.table("papers").update(updates).eq("id", paper_id).execute()
        logger.info(f"Updated paper {paper_id}")
    
    def store_paper_entities(self, paper_id: str, entities: Dict[str, Any]) -> None:
        """
        Store entities extracted from a paper.
        
        Args:
            paper_id: ID of the source paper
            entities: Dictionary of entity categories and their entities
        """
        entity_data = {
            "paper_id": paper_id,
            "entities": entities,
            "created_at": datetime.now().isoformat()
        }
        
        result = self.client.table("paper_entities").insert(entity_data).execute()
        
        if not result.data:
            raise ValueError(f"Failed to store entities for paper {paper_id}")
        
        # Mark paper as processed
        self.update_paper(paper_id, {"processed": True})
        
        logger.info(f"Stored entities for paper {paper_id}")
    
    def store_review_structure(self, topic: str, structure: Dict[str, Any], paper_ids: List[str]) -> None:
        """
        Store a literature review structure.
        
        Args:
            topic: Review topic
            structure: Review structure dictionary
            paper_ids: List of paper IDs included in the review
        """
        structure_data = {
            "topic": topic,
            "structure": structure,
            "paper_ids": paper_ids,
            "created_at": datetime.now().isoformat()
        }
        
        result = self.client.table("review_structures").insert(structure_data).execute()
        
        if not result.data:
            raise ValueError(f"Failed to store review structure for topic '{topic}'")
        
        logger.info(f"Stored review structure for topic '{topic}'")
    
    def store_research_gaps(self, topic: str, gaps: List[Dict[str, Any]], paper_ids: List[str]) -> None:
        """
        Store research gaps identified in the literature.
        
        Args:
            topic: Research topic
            gaps: List of gap dictionaries
            paper_ids: List of paper IDs analyzed
        """
        gaps_data = {
            "topic": topic,
            "gaps": gaps,
            "paper_ids": paper_ids,
            "created_at": datetime.now().isoformat()
        }
        
        result = self.client.table("research_gaps").insert(gaps_data).execute()
        
        if not result.data:
            raise ValueError(f"Failed to store research gaps for topic '{topic}'")
        
        logger.info(f"Stored {len(gaps)} research gaps for topic '{topic}'")
    
    def store_research_trends(self, trends: Dict[str, Any], paper_ids: List[str]) -> None:
        """
        Store research trends identified in the literature.
        
        Args:
            trends: Trends dictionary
            paper_ids: List of paper IDs analyzed
        """
        trends_data = {
            "trends": trends,
            "paper_ids": paper_ids,
            "created_at": datetime.now().isoformat()
        }
        
        result = self.client.table("research_trends").insert(trends_data).execute()
        
        if not result.data:
            raise ValueError("Failed to store research trends")
        
        logger.info(f"Stored research trends for {len(paper_ids)} papers")
    
    def store_literature_review(self, topic: str, content: str, paper_ids: List[str]) -> str:
        """
        Store a generated literature review.
        
        Args:
            topic: Review topic
            content: Review content text
            paper_ids: List of paper IDs included in the review
            
        Returns:
            ID of the stored review
        """
        review_data = {
            "topic": topic,
            "content": content,
            "paper_ids": paper_ids,
            "created_at": datetime.now().isoformat()
        }
        
        result = self.client.table("literature_reviews").insert(review_data).execute()
        
        if not result.data:
            raise ValueError(f"Failed to store literature review for topic '{topic}'")
        
        review_id = result.data[0]["id"]
        logger.info(f"Stored literature review for topic '{topic}' with ID {review_id}")
        
        return review_id
    
    def get_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        Get a paper by ID.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            Paper data dictionary or empty dictionary if not found
        """
        result = self.client.table("papers").select("*").eq("id", paper_id).execute()
        
        if not result.data:
            logger.warning(f"Paper with ID {paper_id} not found")
            return {}
        
        return result.data[0]
    
    def get_paper_entities(self, paper_id: str) -> Dict[str, Any]:
        """
        Get entities for a specific paper.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            Entities dictionary or empty dictionary if not found
        """
        result = self.client.table("paper_entities").select("*").eq("paper_id", paper_id).execute()
        
        if not result.data:
            logger.warning(f"No entities found for paper {paper_id}")
            return {}
        
        return result.data[0]["entities"]
    
    def get_papers_by_ids(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get multiple papers by their IDs.
        
        Args:
            paper_ids: List of paper IDs
            
        Returns:
            List of paper data dictionaries
        """
        # Supabase doesn't have a direct "IN" query method, so we need to fetch all and filter
        papers = []
        for paper_id in paper_ids:
            paper = self.get_paper(paper_id)
            if paper:
                papers.append(paper)
        
        return papers
    
    def get_latest_review_structure(self, topic: str, paper_ids: List[str]) -> Dict[str, Any]:
        """
        Get the latest review structure for a topic and set of papers.
        
        Args:
            topic: Review topic
            paper_ids: List of paper IDs
            
        Returns:
            Review structure dictionary or empty dictionary if not found
        """
        # Sort paper_ids to ensure consistent matching regardless of order
        sorted_paper_ids = sorted(paper_ids)
        
        # We need to query and filter manually since Supabase doesn't have advanced filtering
        result = self.client.table("review_structures").select("*").eq("topic", topic).execute()
        
        if not result.data:
            return {}
        
        # Find matching structure with the same papers (regardless of order)
        matching_structures = []
        for structure in result.data:
            stored_paper_ids = structure.get("paper_ids", [])
            if sorted(stored_paper_ids) == sorted_paper_ids:
                matching_structures.append(structure)
        
        if not matching_structures:
            return {}
        
        # Return the most recent one
        matching_structures.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return matching_structures[0].get("structure", {})
    
    def get_latest_research_gaps(self, topic: str, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get the latest research gaps for a topic and set of papers.
        
        Args:
            topic: Research topic
            paper_ids: List of paper IDs
            
        Returns:
            List of research gap dictionaries or empty list if not found
        """
        # Sort paper_ids to ensure consistent matching regardless of order
        sorted_paper_ids = sorted(paper_ids)
        
        result = self.client.table("research_gaps").select("*").eq("topic", topic).execute()
        
        if not result.data:
            return []
        
        # Find matching gaps with the same papers
        matching_gaps = []
        for gap_record in result.data:
            stored_paper_ids = gap_record.get("paper_ids", [])
            if sorted(stored_paper_ids) == sorted_paper_ids:
                matching_gaps.append(gap_record)
        
        if not matching_gaps:
            return []
        
        # Return the most recent one
        matching_gaps.sort(key=lambda g: g.get("created_at", ""), reverse=True)
        return matching_gaps[0].get("gaps", [])
    
    def get_latest_research_trends(self, paper_ids: List[str]) -> Dict[str, Any]:
        """
        Get the latest research trends for a set of papers.
        
        Args:
            paper_ids: List of paper IDs
            
        Returns:
            Trends dictionary or empty dictionary if not found
        """
        # Sort paper_ids to ensure consistent matching
        sorted_paper_ids = sorted(paper_ids)
        
        result = self.client.table("research_trends").select("*").execute()
        
        if not result.data:
            return {}
        
        # Find matching trends with the same papers
        matching_trends = []
        for trend_record in result.data:
            stored_paper_ids = trend_record.get("paper_ids", [])
            if sorted(stored_paper_ids) == sorted_paper_ids:
                matching_trends.append(trend_record)
        
        if not matching_trends:
            return {}
        
        # Return the most recent one
        matching_trends.sort(key=lambda t: t.get("created_at", ""), reverse=True)
        return matching_trends[0].get("trends", {})
    
    def get_paper_count(self) -> int:
        """
        Get the total number of papers in the database.
        
        Returns:
            Number of papers
        """
        result = self.client.table("papers").select("id", count="exact").execute()
        return result.count if hasattr(result, "count") else 0
    
    def get_unprocessed_papers(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get papers that haven't been processed for entity extraction.
        
        Args:
            limit: Maximum number of papers to return
            
        Returns:
            List of paper dictionaries
        """
        result = self.client.table("papers").select("*").eq("processed", False).limit(limit).execute()
        return result.data
    
    def search_papers(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for papers by title or abstract.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching paper dictionaries
        """
        # Supabase doesn't have great full-text search in the client library
        # This is a simplified approach
        results = self.client.table("papers").select("*").execute()
        
        # Filter papers that contain the query in title or abstract
        matching_papers = []
        query_lower = query.lower()
        for paper in results.data:
            title = paper.get("title", "").lower()
            abstract = paper.get("abstract", "").lower()
            
            if query_lower in title or query_lower in abstract:
                matching_papers.append(paper)
        
        return matching_papers
    
    def get_all_papers_ids(self) -> List[str]:
        """
        Get all paper IDs in the database.
        
        Returns:
            List of paper IDs
        """
        result = self.client.table("papers").select("id").execute()
        if not result.data:
            return []
        return [paper["id"] for paper in result.data]
