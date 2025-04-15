"""
Main module for the Literature Review System.

This module initializes the system components and provides the main workflow
for processing papers, extracting entities, building knowledge graphs,
and generating literature reviews.
"""

import os
import argparse
import logging
from typing import List, Dict, Any

from config import (
    DEFAULT_MODEL, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    SUPABASE_URL, SUPABASE_KEY,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
)
from models import initialize_anthropic, initialize_genai
from database.neo4j_manager import Neo4jManager
from database.supabase_manager import SupabaseManager
from components.document_processor import DocumentProcessor
from components.entity_extractor import EntityExtractor
from components.graph_builder import KnowledgeGraphBuilder
from components.review_generator import ReviewGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("literature_review.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiteratureReviewSystem:
    """Literature Review System orchestrating all components."""
    
    def __init__(
        self,
        supabase_url: str = SUPABASE_URL,
        supabase_key: str = SUPABASE_KEY,
        neo4j_uri: str = NEO4J_URI,
        neo4j_user: str = NEO4J_USER,
        neo4j_password: str = NEO4J_PASSWORD,
        model: str = DEFAULT_MODEL
    ):
        """
        Initialize the Literature Review System.
        
        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            model: Default LLM model to use
        """
        # Initialize optional API clients
        if ANTHROPIC_API_KEY:
            initialize_anthropic(ANTHROPIC_API_KEY)
            logger.info("Initialized Anthropic client")
        
        if GOOGLE_API_KEY:
            initialize_genai(GOOGLE_API_KEY)
            logger.info("Initialized Google Generative AI client")
        
        # Initialize database managers
        self.supabase = SupabaseManager(url=supabase_url, key=supabase_key)
        self.neo4j = Neo4jManager(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
        
        # Initialize components
        self.document_processor = DocumentProcessor(self.supabase, model=model)
        self.entity_extractor = EntityExtractor(self.supabase, model=model)
        self.graph_builder = KnowledgeGraphBuilder(self.neo4j, self.supabase, model=model)
        self.review_generator = ReviewGenerator(self.neo4j, self.supabase, model=model)
        
        self.model = model
        logger.info(f"Literature Review System initialized with model: {model}")
    
    def initialize(self):
        """Initialize the system databases."""
        logger.info("Initializing system databases")
        
        # Initialize Supabase tables
        self.supabase.initialize_tables()
        
        # Initialize Neo4j schema
        self.neo4j.initialize_schema()
        
        logger.info("System initialization complete")
    
    def process_papers(self, pdf_folder_path: str) -> List[str]:
        """
        Process all papers in a folder.
        
        Args:
            pdf_folder_path: Path to folder with PDF files
            
        Returns:
            List of processed paper IDs
        """
        logger.info(f"Processing papers from {pdf_folder_path}")
        return self.document_processor.batch_process_papers(pdf_folder_path)
    
    def process_paper_from_url(self, url: str) -> str:
        """
        Process a paper from a URL.

        Args:
            url: URL to the PDF file

        Returns:
            ID of the processed paper
        """
        logger.info(f"Processing paper from URL: {url}")
        return self.document_processor.process_from_url(url)

    def extract_entities(self, paper_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Extract entities from papers.
        
        Args:
            paper_ids: List of paper IDs to process
            
        Returns:
            Dictionary mapping paper IDs to extracted entities
        """
        logger.info(f"Extracting entities from {len(paper_ids)} papers")
        return self.entity_extractor.batch_extract_entities(paper_ids)
    
    def build_knowledge_graph(self, paper_ids: List[str]) -> Dict[str, int]:
        """
        Build knowledge graph from papers.
        
        Args:
            paper_ids: List of paper IDs to include
            
        Returns:
            Dictionary with graph statistics
        """
        logger.info(f"Building knowledge graph for {len(paper_ids)} papers")
        return self.graph_builder.build_knowledge_graph(paper_ids)
    
    def generate_review(self, topic: str, paper_ids: List[str], support_doc: str) -> str:
        """
        Generate a literature review.
        
        Args:
            topic: Review topic
            paper_ids: List of paper IDs to include
            
        Returns:
            Generated literature review text
        """
        logger.info(f"Generating literature review on '{topic}' with {len(paper_ids)} papers")
        
        # Step 1: Cluster papers
        clusters = self.review_generator.cluster_papers_by_topic(paper_ids, support_doc)
        logger.info(f"Clustered papers into {len(clusters)} topics")
        
        # Step 2: Create narrative structure
        structure = self.review_generator.create_narrative_structure(topic, paper_ids, support_doc)
        logger.info(f"Created narrative structure with {len(structure)} sections")
        
        # Step 3: Identify research gaps
        gaps = self.review_generator.identify_research_gaps(topic, paper_ids, support_doc)
        logger.info(f"Identified {len(gaps)} research gaps")
        
        # Step 4: Identify research trends
        trends = self.review_generator.identify_research_trends(paper_ids, support_doc)
        logger.info("Identified research trends")
        
        # Step 5: Generate the review
        review = self.review_generator.generate_literature_review(topic, paper_ids, support_doc)
        logger.info(f"Generated literature review with {len(review)} characters")
        
        return review
    
    def run_full_pipeline(self, pdf_folder_path: str, topic: str) -> str:
        """
        Run the full pipeline from paper processing to review generation.
        
        Args:
            pdf_folder_path: Path to folder with PDF files
            topic: Review topic
            
        Returns:
            Generated literature review text
        """
        logger.info(f"Running full pipeline for topic '{topic}'")
        
        # Step 1: Process papers
        paper_ids = self.process_papers(pdf_folder_path)
        logger.info(f"Processed {len(paper_ids)} papers")
        
        if not paper_ids:
            err_msg = "No papers were processed. Check the PDF folder path."
            logger.error(err_msg)
            return f"# Error\n\n{err_msg}"
        
        # Step 2: Extract entities
        self.extract_entities(paper_ids)
        logger.info("Extracted entities from papers")
        
        # Step 3: Build knowledge graph
        self.build_knowledge_graph(paper_ids)
        logger.info("Built knowledge graph")
        
        # Step 4: Generate review
        review = self.generate_review(topic, paper_ids)
        logger.info("Generated literature review")
        
        # Save to file
        output_file = f"literature_review_{topic.replace(' ', '_').lower()}.md"
        with open(output_file, "w") as file:
            file.write(review)
        
        logger.info(f"Literature review saved to {output_file}")
        
        return review
    
    def run_on_old_papers(self, topic: str, support_doc: str) -> str:
        """
        Run the review generation on previously processed papers.
        
        Args:
            topic: Review topic
            support_doc: Path to a support document for additional context
            
        Returns:
            Generated literature review text
        """
        logger.info(f"Running review generation on old papers for topic '{topic}'")

        if not support_doc:
            err_msg = "Support document is required for old papers."
            logger.error(err_msg)
            return f"# Error\n\n{err_msg}"
        
        # Get paper IDs from Supabase
        paper_ids = self.supabase.get_all_papers_ids()
        
        if not paper_ids:
            err_msg = "No previously processed papers found."
            logger.error(err_msg)
            return f"# Error\n\n{err_msg}"
        
        # Generate review
        review = self.generate_review(topic, paper_ids, support_doc)
        logger.info("Generated literature review from old papers")
        
        return review
    
    def close(self):
        """Close database connections."""
        logger.info("Closing database connections")
        self.neo4j.close()
        logger.info("Literature Review System shutdown complete")


def main():
    """Main function for command-line execution."""
    parser = argparse.ArgumentParser(description="Literature Review System")
    parser.add_argument(
        "--pdf_folder", 
        type=str, 
        help="Path to folder containing PDF files"
    )
    parser.add_argument(
        "--topic", 
        type=str, 
        help="Topic for the literature review"
    )
    parser.add_argument(
        "--paper_ids", 
        type=str, 
        help="Comma-separated list of paper IDs (if papers already processed)"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="URL to a PDF file to process"
    )
    parser.add_argument(
        "--support_doc",
        type=str,
        help="Path to a support document for additional context"
    )

    args = parser.parse_args()
    
    # Validate arguments
    if not args.topic:
        print("Error: Review topic is required")
        parser.print_help()
        return
    
    if not args.pdf_folder and not args.paper_ids and not args.url and not args.support_doc:
        print("Error: Either PDF folder, URL, or paper IDs must be provided")
        parser.print_help()
        return
    
    try:
        # Initialize system
        system = LiteratureReviewSystem()
        system.initialize()
        
        # Process from URL if provided
        if args.url:
            paper_id = system.process_paper_from_url(args.url)
            print(f"Processed paper from URL. Paper ID: {paper_id}")
            paper_ids = [paper_id]

            # Extract entities
            system.extract_entities(paper_ids)

            # Build knowledge graph
            system.build_knowledge_graph(paper_ids)

            # Generate review
            review = system.generate_review(args.topic, paper_ids)

            # Save to file
            output_file = f"literature_review_{args.topic.replace(' ', '_').lower()}.md"
            with open(output_file, "w") as file:
                file.write(review)

            print(f"Literature review saved to {output_file}")

        # Process from folder if provided
        elif args.pdf_folder:
            # Full pipeline with PDF processing
            review = system.run_full_pipeline(args.pdf_folder, args.topic)
        elif args.support_doc:
            if not os.path.exists(args.support_doc):
                err_msg = f"Support document '{args.support_doc}' does not exist."
                logger.error(err_msg)
                return f"# Error\n\n{err_msg}"

            # check if support_doc is md file 
            if not args.support_doc.endswith(".md"):
                print("Error: Support document must be a markdown file (.md)")
                return
            
            # read support document
            with open(args.support_doc, "r") as file:
                support_doc_content = file.read()

            # Run on old papers with support document
            review = system.run_on_old_papers(args.topic, support_doc_content)

            # Save to file
            output_file = f"literature_review_{args.topic.replace(' ', '_').lower()}.md"
            with open(output_file, "w") as file:
                file.write(review)

            print(f"Literature review saved to {output_file}")
        else:
            # Use existing paper IDs
            paper_ids = [id.strip() for id in args.paper_ids.split(",")]
            review = system.generate_review(args.topic, paper_ids)
            
            # Save to file
            output_file = f"literature_review_{args.topic.replace(' ', '_').lower()}.md"
            with open(output_file, "w") as file:
                file.write(review)
            
            print(f"Literature review saved to {output_file}")
        
        # Clean up
        system.close()
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
