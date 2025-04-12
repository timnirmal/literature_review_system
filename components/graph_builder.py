"""
Knowledge Graph Builder component for the Literature Review System.

This module constructs a knowledge graph representing papers, authors,
entities, and their relationships in Neo4j.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Tuple

from config import DEFAULT_MODEL
from database.neo4j_manager import Neo4jManager
from database.supabase_manager import SupabaseManager
from models import call_model, parse_json_response
from prompts import PAPER_RELATIONSHIP_PROMPT, CITATION_ANALYSIS_PROMPT, TIME_PERIOD_SUMMARY_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KnowledgeGraphBuilder:
    """Constructs and analyzes the knowledge graph of papers and their relationships."""
    
    def __init__(
        self,
        neo4j_manager: Neo4jManager,
        supabase_manager: SupabaseManager,
        model: str = DEFAULT_MODEL
    ):
        """
        Initialize the knowledge graph builder.
        
        Args:
            neo4j_manager: Neo4jManager instance for graph database operations
            supabase_manager: SupabaseManager instance for document database operations
            model: LLM model to use for relationship analysis
        """
        self.neo4j = neo4j_manager
        self.supabase = supabase_manager
        self.model = model
    
    def initialize_graph_schema(self):
        """Initialize Neo4j graph schema with constraints and indexes."""
        logger.info("Initializing Neo4j graph schema")
        self.neo4j.initialize_schema()
    
    def add_paper_to_graph(self, paper_id: str) -> bool:
        """
        Add a paper and its entities to the knowledge graph.
        
        Args:
            paper_id: ID of the paper to add
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Adding paper {paper_id} to knowledge graph")
        
        try:
            # Fetch paper data
            paper = self.supabase.get_paper(paper_id)
            if not paper:
                logger.error(f"Paper {paper_id} not found")
                return False
            
            # Fetch entities
            entities = self.supabase.get_paper_entities(paper_id)
            if not entities:
                logger.warning(f"No entities found for paper {paper_id}")
                # We can still add the paper without entities
            
            # Add paper node
            self.neo4j.add_paper({
                "id": paper_id,
                "title": paper.get("title", "Unknown Title"),
                "year": paper.get("year"),
                "venue": paper.get("venue", "Unknown Venue"),
                "abstract": paper.get("abstract", ""),
                "doi": paper.get("doi")
            })
            
            # Add authors
            authors = paper.get("authors", [])
            if authors:
                self.neo4j.add_author_relationships(paper_id, authors)
            
            # Add entities and relationships
            if entities:
                if "concepts" in entities:
                    self.neo4j.add_entities(paper_id, "Concept", entities["concepts"])
                
                if "methods" in entities:
                    self.neo4j.add_entities(paper_id, "Method", entities["methods"])
                
                if "technologies" in entities:
                    self.neo4j.add_entities(paper_id, "Technology", entities["technologies"])
                
                if "datasets" in entities:
                    self.neo4j.add_entities(paper_id, "Dataset", entities["datasets"])
                
                # Add findings as properties of the paper
                if "findings" in entities and entities["findings"]:
                    findings_text = "\n".join([
                        f"{f['name']}: {f['description']}" 
                        for f in entities["findings"]
                    ])
                    self.neo4j.add_paper({
                        "id": paper_id,
                        "findings": findings_text[:1000]  # Limit size
                    })
            
            logger.info(f"Successfully added paper {paper_id} to knowledge graph")
            return True
            
        except Exception as e:
            logger.error(f"Error adding paper {paper_id} to knowledge graph: {str(e)}")
            return False
    
    def analyze_paper_relationships(self, paper_ids: List[str]) -> int:
        """
        Analyze relationships between papers using LLM.
        
        Args:
            paper_ids: List of paper IDs to analyze
            
        Returns:
            Number of relationships identified
        """
        logger.info(f"Analyzing relationships between {len(paper_ids)} papers")
        
        paper_data = {}
        relationship_count = 0
        
        # Fetch all paper data
        for paper_id in paper_ids:
            paper = self.supabase.get_paper(paper_id)
            if paper:
                paper_data[paper_id] = paper
        
        if len(paper_data) < 2:
            logger.warning("Not enough papers found for relationship analysis")
            return 0
        
        # Compare papers pairwise
        for i, paper_id1 in enumerate(paper_ids):
            for paper_id2 in paper_ids[i+1:]:
                if paper_id1 not in paper_data or paper_id2 not in paper_data:
                    continue
                
                paper1 = paper_data[paper_id1]
                paper2 = paper_data[paper_id2]
                
                logger.info(f"Analyzing relationship between papers {paper_id1} and {paper_id2}")
                
                # Analyze relationship
                relationship = self._analyze_paper_pair(paper1, paper2)
                
                if relationship["relationship_type"] != "UNRELATED":
                    # Add relationship to graph
                    self.neo4j.add_paper_relationship(
                        paper_id1=paper_id1,
                        paper_id2=paper_id2,
                        relationship_type=relationship["relationship_type"],
                        description=relationship["description"],
                        confidence=relationship["confidence"]
                    )
                    relationship_count += 1
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
        
        logger.info(f"Identified {relationship_count} relationships between papers")
        return relationship_count
    
    def _analyze_paper_pair(self, paper1: Dict[str, Any], paper2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the relationship between two papers using LLM.
        
        Args:
            paper1: First paper data
            paper2: Second paper data
            
        Returns:
            Dictionary describing the relationship
        """
        # Format prompt
        prompt = PAPER_RELATIONSHIP_PROMPT.format(
            title1=paper1["title"],
            abstract1=paper1["abstract"],
            title2=paper2["title"],
            abstract2=paper2["abstract"]
        )
        
        try:
            # Call LLM
            response = call_model(
                prompt=prompt,
                model=self.model,
                system_prompt="You are a scientific paper relationship analyzer.",
                response_format={"type": "json_object"}
            )
            
            # Parse result
            result = parse_json_response(response)
            
            # Ensure required fields are present
            if "relationship_type" not in result:
                result["relationship_type"] = "UNRELATED"
            if "description" not in result:
                result["description"] = ""
            if "confidence" not in result:
                result["confidence"] = 0
                
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing paper relationship: {str(e)}")
            return {
                "relationship_type": "UNRELATED",
                "description": "Analysis failed",
                "confidence": 0
            }
    
    def create_chronological_organization(self, paper_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organize papers chronologically and identify trends.
        
        Args:
            paper_ids: List of paper IDs to organize
            
        Returns:
            Dictionary mapping time periods to lists of papers
        """
        logger.info(f"Creating chronological organization for {len(paper_ids)} papers")
        
        # Get papers with years
        papers_with_years = []
        for paper_id in paper_ids:
            paper = self.supabase.get_paper(paper_id)
            if paper and paper.get("year"):
                papers_with_years.append(paper)
        
        if not papers_with_years:
            logger.warning("No papers with year information found")
            return {}
        
        # Sort by year
        papers_with_years.sort(key=lambda p: p["year"])
        
        # Group by time periods (2-year intervals)
        periods = {}
        for paper in papers_with_years:
            year = int(paper["year"])
            period = f"{year}-{year+1}"
            
            if period not in periods:
                periods[period] = []
            
            periods[period].append(paper)
        
        # Store chronological data in graph
        for period, papers in periods.items():
            # Create period node
            self.neo4j.create_time_period(period)
            
            # Connect papers to period
            for paper in papers:
                self.neo4j.connect_paper_to_period(paper["id"], period)
        
        logger.info(f"Created chronological organization with {len(periods)} time periods")
        return periods
    
    def analyze_citation_network(self, paper_ids: List[str]) -> int:
        """
        Analyze citation patterns between papers.
        
        Args:
            paper_ids: List of paper IDs to analyze
            
        Returns:
            Number of citations identified
        """
        logger.info(f"Analyzing citation network for {len(paper_ids)} papers")
        
        citation_count = 0
        paper_data = {}
        
        # Fetch paper data
        for paper_id in paper_ids:
            paper = self.supabase.get_paper(paper_id)
            if paper:
                paper_data[paper_id] = paper
        
        # Sort papers by year
        sorted_papers = sorted(
            [p for p in paper_data.values() if p.get("year")],
            key=lambda p: p["year"]
        )
        
        if len(sorted_papers) < 2:
            logger.warning("Not enough papers with year information for citation analysis")
            return 0
        
        # For each paper, analyze which older papers it might cite
        for i, paper in enumerate(sorted_papers):
            # Only look at potential citations to older papers
            potential_citations = sorted_papers[:i]
            
            if not potential_citations:
                continue
            
            # Prepare data for citation analysis
            citing_paper = paper
            
            # Format potential cited papers
            potential_cited_papers = "\n".join([
                f"Paper {idx+1}: {p['title']} ({p['year']})" 
                for idx, p in enumerate(potential_citations)
            ])
            
            # Analyze citations
            logger.info(f"Analyzing potential citations from paper '{citing_paper['title']}'")
            
            prompt = CITATION_ANALYSIS_PROMPT.format(
                citing_year=citing_paper['year'],
                citing_title=citing_paper['title'],
                citing_abstract=citing_paper['abstract'],
                potential_cited_papers=potential_cited_papers
            )
            
            try:
                # Call LLM
                response = call_model(
                    prompt=prompt,
                    model=self.model,
                    system_prompt="You are a citation analyzer for academic papers.",
                    response_format={"type": "json_object"}
                )
                
                # Parse result
                citation_analysis = parse_json_response(response)
                
                # Process citations
                for citation in citation_analysis.get("citations", []):
                    if citation.get("cited") and citation.get("confidence", 0) > 0.7:
                        # Get cited paper
                        cited_idx = citation.get("paper_idx", 0) - 1
                        if 0 <= cited_idx < len(potential_citations):
                            cited_paper = potential_citations[cited_idx]
                            
                            # Add citation to graph
                            self.neo4j.add_citation(
                                citing_paper_id=citing_paper["id"],
                                cited_paper_id=cited_paper["id"],
                                confidence=citation.get("confidence", 0.7),
                                reason=citation.get("reason", "")
                            )
                            citation_count += 1
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error analyzing citations for paper {citing_paper['id']}: {str(e)}")
        
        logger.info(f"Identified {citation_count} citations between papers")
        return citation_count
    
    def build_knowledge_graph(self, paper_ids: List[str]) -> Dict[str, int]:
        """
        Build the complete knowledge graph for a set of papers.
        
        Args:
            paper_ids: List of paper IDs to include
            
        Returns:
            Dictionary with statistics about the built graph
        """
        logger.info(f"Building knowledge graph for {len(paper_ids)} papers")
        
        # Initialize schema
        self.initialize_graph_schema()
        
        # Add papers to graph
        papers_added = 0
        for paper_id in paper_ids:
            success = self.add_paper_to_graph(paper_id)
            if success:
                papers_added += 1
        
        # Analyze relationships
        relationships = self.analyze_paper_relationships(paper_ids)
        
        # Create chronological organization
        periods = self.create_chronological_organization(paper_ids)
        
        # Analyze citation network
        citations = self.analyze_citation_network(paper_ids)
        
        # Return statistics
        stats = {
            "papers_added": papers_added,
            "relationships_identified": relationships,
            "time_periods": len(periods),
            "citations_identified": citations
        }
        
        logger.info(f"Knowledge graph built successfully: {stats}")
        return stats