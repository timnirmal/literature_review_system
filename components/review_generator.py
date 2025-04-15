"""
Literature Review Generator component for the Literature Review System.

This module generates comprehensive literature reviews based on the
knowledge graph and extracted entities from academic papers.
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple

from config import DEFAULT_MODEL
from database.neo4j_manager import Neo4jManager
from database.supabase_manager import SupabaseManager
from models import call_model, parse_json_response
from prompts import (
    TOPIC_CLUSTERING_PROMPT,
    NARRATIVE_STRUCTURE_PROMPT,
    RESEARCH_GAPS_PROMPT,
    RESEARCH_TRENDS_PROMPT,
    SECTION_GENERATION_PROMPT
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReviewGenerator:
    """Generates comprehensive literature reviews based on analyzed papers."""
    
    def __init__(
        self,
        neo4j_manager: Neo4jManager,
        supabase_manager: SupabaseManager,
        model: str = DEFAULT_MODEL
    ):
        """
        Initialize the review generator.
        
        Args:
            neo4j_manager: Neo4jManager instance for graph database operations
            supabase_manager: SupabaseManager instance for document database operations
            model: LLM model to use for review generation
        """
        self.neo4j = neo4j_manager
        self.supabase = supabase_manager
        self.model = model
    
    def cluster_papers_by_topic(self, paper_ids: List[str], support_doc: str) -> Dict[str, Dict[str, Any]]:
        """
        Cluster papers into topics using LLM.
        
        Args:
            paper_ids: List of paper IDs to cluster
            
        Returns:
            Dictionary mapping topic names to topic data
        """
        logger.info(f"Clustering {len(paper_ids)} papers by topic")
        
        # Get paper data
        papers_data = []
        for paper_id in paper_ids:
            paper = self.supabase.get_paper(paper_id)
            if paper:
                papers_data.append(paper)
        
        if not papers_data:
            logger.warning("No papers found to cluster")
            return {}
        
        # Format papers for LLM
        papers_text = "\n\n".join([
            f"Paper {i+1}:\nTitle: {p['title']}\nAbstract: {p['abstract']}\nID: {p['id']}"
            for i, p in enumerate(papers_data)
        ])

        if support_doc:
            papers_text += f"\n\nSupporting Document:\n{support_doc}"
        
        # Ask LLM to cluster
        prompt = TOPIC_CLUSTERING_PROMPT.format(papers_text=papers_text)
        
        try:
            # Call LLM
            response = call_model(
                prompt=prompt,
                model=self.model,
                system_prompt="You are a research paper topic clustering specialist.",
                response_format={"type": "json_object"}
            )
            
            # Parse clusters
            clusters = parse_json_response(response)
            
            # Store clusters in graph
            for topic_name, topic_data in clusters.items():
                # Create topic node
                self.neo4j.create_topic(
                    topic_name=topic_name,
                    description=topic_data.get("description", "")
                )
                
                # Connect papers to topic
                for paper_id in topic_data.get("paper_ids", []):
                    self.neo4j.connect_paper_to_topic(paper_id, topic_name)
            
            logger.info(f"Clustered papers into {len(clusters)} topics")
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering papers: {str(e)}")
            return {}
    
    def create_narrative_structure(self, topic: str, paper_ids: List[str], support_doc: str) -> Dict[str, Any]:
        """
        Create a narrative structure for the literature review.
        
        Args:
            topic: Review topic
            paper_ids: List of paper IDs to include
            
        Returns:
            Dictionary with the review structure
        """
        logger.info(f"Creating narrative structure for topic '{topic}'")
        
        # Check if we already have a structure
        existing_structure = self.supabase.get_latest_review_structure(topic, paper_ids)
        if existing_structure:
            logger.info(f"Using existing narrative structure for topic '{topic}'")
            return existing_structure
        
        # Get paper data
        papers_data = []
        for paper_id in paper_ids:
            paper = self.supabase.get_paper(paper_id)
            if paper:
                papers_data.append(paper)
        
        if not papers_data:
            logger.warning("No papers found for creating narrative structure")
            return {}

        # Ensure all papers have a year
        for paper in papers_data:
            if not paper.get("year"):
                paper["year"] = 9999
        
        # Sort papers by year
        papers_data.sort(key=lambda p: p.get("year", 9999))
        
        # Format papers for LLM
        papers_text = "\n\n".join([
            f"Paper {i+1}:\nTitle: {p['title']}\nAuthors: {', '.join(p.get('authors', []))}\n"
            f"Year: {p.get('year', 'Unknown')}\nAbstract: {p['abstract']}\nID: {p['id']}"
            for i, p in enumerate(papers_data)
        ])

        if support_doc:
            papers_text += f"\n\nSupporting Document:\n{support_doc}"
        
        # Ask LLM to create structure
        prompt = NARRATIVE_STRUCTURE_PROMPT.format(
            topic=topic,
            papers_text=papers_text
        )
        
        try:
            # Call LLM
            response = call_model(
                prompt=prompt,
                model=self.model,
                system_prompt="You are a literature review structure specialist.",
                response_format={"type": "json_object"}
            )
            
            # Parse structure
            structure = parse_json_response(response)
            
            # Store structure in database
            self.supabase.store_review_structure(topic, structure, paper_ids)
            
            logger.info(f"Created narrative structure with {len(structure)} sections")
            return structure
            
        except Exception as e:
            logger.error(f"Error creating narrative structure: {str(e)}")
            return {}
    
    def identify_research_gaps(self, topic: str, paper_ids: List[str], support_doc: str) -> List[Dict[str, Any]]:
        """
        Identify research gaps in the literature.
        
        Args:
            topic: Research topic
            paper_ids: List of paper IDs to analyze
            
        Returns:
            List of research gap dictionaries
        """
        logger.info(f"Identifying research gaps for topic '{topic}'")
        
        # Check if we already have gaps
        existing_gaps = self.supabase.get_latest_research_gaps(topic, paper_ids)
        if existing_gaps and not support_doc:
            logger.info(f"Using existing research gaps for topic '{topic}'")
            return existing_gaps
        
        # Get methods used across papers
        methods = self.neo4j.get_methods_for_papers(paper_ids)
        methods_text = "\n".join([
            f"- {m['name']} (used in {m['count']} papers)" 
            for m in methods
        ])
        
        # Get concepts explored
        concepts = self.neo4j.get_concepts_for_papers(paper_ids)
        concepts_text = "\n".join([
            f"- {c['name']} (appears in {c['count']} papers)" 
            for c in concepts
        ])
        
        # Get papers summary
        papers_data = []
        for paper_id in paper_ids:
            paper = self.supabase.get_paper(paper_id)
            if paper:
                papers_data.append(paper)
        
        papers_text = "\n\n".join([
            f"Paper: {p['title']} ({p.get('year', 'Unknown')})\nAbstract: {p['abstract']}"
            for p in papers_data
        ])

        if support_doc:
            papers_text += f"\n\nSupporting Document:\n{support_doc}"
        
        # Ask LLM to identify gaps
        prompt = RESEARCH_GAPS_PROMPT.format(
            topic=topic,
            methods_text=methods_text,
            concepts_text=concepts_text,
            papers_text=papers_text
        )
        
        try:
            # Call LLM
            response = call_model(
                prompt=prompt,
                model=self.model,
                system_prompt="You are a research gap identification specialist.",
                response_format={"type": "json_object"}
            )
            
            # Parse gaps
            result = parse_json_response(response)
            gaps = result.get("gaps", [])
            
            # Store gaps in database
            self.supabase.store_research_gaps(topic, gaps, paper_ids)
            
            logger.info(f"Identified {len(gaps)} research gaps")
            return gaps
            
        except Exception as e:
            logger.error(f"Error identifying research gaps: {str(e)}")
            return []
    
    def identify_research_trends(self, paper_ids: List[str], support_doc: str) -> Dict[str, Any]:
        """
        Identify research trends over time.
        
        Args:
            paper_ids: List of paper IDs to analyze
            
        Returns:
            Dictionary with trend information
        """
        logger.info(f"Identifying research trends for {len(paper_ids)} papers")
        
        # Check if we already have trends
        existing_trends = self.supabase.get_latest_research_trends(paper_ids)
        if existing_trends and not support_doc:
            logger.info("Using existing research trends")
            return existing_trends
        
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
        
        # Group by year
        papers_by_year = {}
        for paper in papers_with_years:
            year = str(paper["year"])
            if year not in papers_by_year:
                papers_by_year[year] = []
            papers_by_year[year].append(paper)
        
        # Format data for LLM
        years_text = ""
        for year, papers in papers_by_year.items():
            papers_text = "\n".join([f"- {p['title']}" for p in papers])
            years_text += f"\nYear {year} ({len(papers)} papers):\n{papers_text}\n"

        if support_doc:
            years_text += f"\nSupporting Document:\n{support_doc}"
        
        # Ask LLM to identify trends
        prompt = RESEARCH_TRENDS_PROMPT.format(years_text=years_text)
        
        try:
            # Call LLM
            response = call_model(
                prompt=prompt,
                model=self.model,
                system_prompt="You are a research trend analysis specialist.",
                response_format={"type": "json_object"}
            )
            
            # Parse trends
            trends = parse_json_response(response)
            
            # Store trends in database
            self.supabase.store_research_trends(trends, paper_ids)
            
            logger.info("Successfully identified research trends")
            return trends
            
        except Exception as e:
            logger.error(f"Error identifying research trends: {str(e)}")
            return {}
    
    def generate_section(self, topic: str, section_name: str, section_data: Dict[str, Any], papers_data: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a section of the literature review.
        
        Args:
            topic: Review topic
            section_name: Name of the section
            section_data: Section data from the narrative structure
            papers_data: Dictionary mapping paper IDs to their data
            
        Returns:
            Generated section content
        """
        logger.info(f"Generating section '{section_name}' for topic '{topic}'")
        
        # Get the papers for this section
        section_papers = []
        for paper_id in section_data.get("paper_ids", []):
            if paper_id in papers_data:
                section_papers.append(papers_data[paper_id])
        
        if not section_papers:
            logger.warning(f"No papers found for section '{section_name}'")
            return f"## {section_name}\n\nNo papers were available for this section."
        
        # Format papers for this section
        papers_text = []
        for paper in section_papers:
            metadata = paper["metadata"]
            entities = paper.get("entities", {})
            
            paper_text = f"Title: {metadata['title']}\n"
            paper_text += f"Authors: {', '.join(metadata.get('authors', []))}\n"
            paper_text += f"Year: {metadata.get('year', 'Unknown')}\n"
            paper_text += f"Abstract: {metadata.get('abstract', '')}\n\n"
            
            if entities:
                if "concepts" in entities and entities["concepts"]:
                    concepts_text = ", ".join([c["name"] for c in entities["concepts"][:5]])
                    paper_text += f"Key Concepts: {concepts_text}\n"
                
                if "methods" in entities and entities["methods"]:
                    methods_text = ", ".join([m["name"] for m in entities["methods"][:5]])
                    paper_text += f"Methods: {methods_text}\n"
                
                if "findings" in entities and entities["findings"]:
                    findings_text = "; ".join([f["name"] for f in entities["findings"][:5]])
                    paper_text += f"Key Findings: {findings_text}\n"
            
            papers_text.append(paper_text)
        
        all_papers_text = "\n\n".join(papers_text)
        
        # Generate section content
        prompt = SECTION_GENERATION_PROMPT.format(
            topic=topic,
            section_name=section_name,
            section_purpose=section_data.get("section_purpose", ""),
            focus_points=section_data.get("focus_points", ""),
            all_papers_text=all_papers_text
        )
        
        try:
            # Call LLM
            response = call_model(
                prompt=prompt,
                model=self.model,
                system_prompt="You are an academic literature review writer with expertise in synthesizing research papers."
            )
            
            # Format the section
            section_content = f"## {section_name}\n\n{response}"
            logger.info(f"Generated section '{section_name}' with {len(response)} characters")
            
            return section_content
            
        except Exception as e:
            logger.error(f"Error generating section '{section_name}': {str(e)}")
            return f"## {section_name}\n\nError generating this section."
    
    def generate_literature_review(self, topic: str, paper_ids: List[str], support_doc: str) -> str:
        """
        Generate a complete literature review.
        
        Args:
            topic: Review topic
            paper_ids: List of paper IDs to include
            
        Returns:
            Complete literature review text
        """
        logger.info(f"Generating literature review on topic '{topic}' with {len(paper_ids)} papers")
        
        # Step 1: Get or create narrative structure
        structure = self.supabase.get_latest_review_structure(topic, paper_ids)
        if not structure and not support_doc:
            structure = self.create_narrative_structure(topic, paper_ids, support_doc)
        
        if not structure:
            err_msg = f"Failed to create narrative structure for topic '{topic}'"
            logger.error(err_msg)
            return f"# Error Generating Literature Review\n\n{err_msg}"
        
        # Step 2: Get paper data with entities
        papers_data = {}
        for paper_id in paper_ids:
            paper = self.supabase.get_paper(paper_id)
            entities = self.supabase.get_paper_entities(paper_id)
            
            if paper:
                papers_data[paper_id] = {
                    "metadata": paper,
                    "entities": entities
                }
        
        if not papers_data:
            err_msg = "No papers found for literature review"
            logger.error(err_msg)
            return f"# Error Generating Literature Review\n\n{err_msg}"
        
        # Step 3: Get research gaps
        gaps = self.supabase.get_latest_research_gaps(topic, paper_ids)
        if not gaps or support_doc:
            gaps = self.identify_research_gaps(topic, paper_ids, support_doc)
        
        # Step 4: Get research trends
        trends = self.supabase.get_latest_research_trends(paper_ids)
        if not trends or support_doc:
            trends = self.identify_research_trends(paper_ids, support_doc)
        
        # Step 5: Generate each section of the review
        review_sections = {}
        
        for section_name, section_data in structure.items():
            section_content = self.generate_section(topic, section_name, section_data, papers_data)
            review_sections[section_name] = section_content
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Step 6: Generate gaps and future research section if gaps found
        if gaps:
            gaps_text = "\n\n".join([
                f"**{gap.get('gap_type', 'Research')} Gap**: {gap.get('description', '')}\n"
                f"*Significance*: {gap.get('significance', '')}\n"
                f"*Future Research Direction*: {gap.get('future_research', '')}"
                for gap in gaps[:5]  # Use top 5 gaps
            ])
            
            gaps_section = f"""
            ## Research Gaps and Future Directions
            
            The analysis of the literature on {topic} revealed several important gaps that merit attention in future research:
            
            {gaps_text}
            
            These gaps represent opportunities for researchers to make significant contributions to the field.
            """
            
            review_sections["Research Gaps and Future Directions"] = gaps_section
        
        # Step 7: Generate trends section if trends found
        if trends:
            emerging_text = "\n".join([
                f"- {theme}" for theme in trends.get("emerging_themes", [])
            ])
            
            trends_section = f"""
            ## Research Trends and Trajectory
            
            {trends.get("overall_trajectory", "Analysis of research trends shows evolving interests in this field.")}
            
            **Emerging Research Themes:**
            {emerging_text}
            
            **Methodological Evolution:**
            {trends.get("methodological_evolution", "")}
            
            **Conceptual Shifts:**
            {trends.get("conceptual_shifts", "")}
            """
            
            review_sections["Research Trends and Trajectory"] = trends_section
        
        # Step 8: Combine all sections into complete review
        full_review = f"""
        # Literature Review: {topic}
        
        """
        
        # Follow the structure order
        for section_name in structure.keys():
            if section_name in review_sections:
                full_review += f"{review_sections[section_name]}\n\n"
        
        # Add remaining sections (gaps, trends) if not already included
        for section_name, content in review_sections.items():
            if section_name not in structure:
                full_review += f"{content}\n\n"
        
        # Step 9: Store the review in database
        self.supabase.store_literature_review(topic, full_review, paper_ids)
        
        logger.info(f"Generated complete literature review on '{topic}' with {len(full_review)} characters")
        return full_review