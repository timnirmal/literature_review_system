"""
Neo4j database manager for the Literature Review System.

This module handles all Neo4j database operations, providing a unified
interface for creating, updating, and querying the knowledge graph.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from neo4j import GraphDatabase, Transaction, Driver, Session

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jManager:
    """Neo4j database manager for the knowledge graph."""
    
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        """
        Initialize Neo4j database connection.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.connect()
    
    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connection
            with self.driver.session() as session:
                result = session.run("RETURN 'Connected' AS status")
                for record in result:
                    logger.info(f"Neo4j connection status: {record['status']}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def initialize_schema(self) -> None:
        """Initialize Neo4j graph schema with constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT technology_name IF NOT EXISTS FOR (t:Technology) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT dataset_name IF NOT EXISTS FOR (d:Dataset) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT time_period_name IF NOT EXISTS FOR (t:TimePeriod) REQUIRE t.name IS UNIQUE"
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logger.warning(f"Error creating constraint: {str(e)}")
        
        logger.info("Neo4j schema initialized with constraints")
    
    def add_paper(self, paper_data: Dict[str, Any]) -> None:
        """
        Add a paper node to the graph.
        
        Args:
            paper_data: Dictionary containing paper metadata
        """
        cypher = """
        MERGE (p:Paper {id: $id})
        SET p.title = $title,
            p.year = $year,
            p.venue = $venue,
            p.abstract = $abstract,
            p.doi = $doi
        """
        
        with self.driver.session() as session:
            session.run(cypher, paper_data)
        
        logger.info(f"Added paper '{paper_data['title']}' to Neo4j")
    
    def add_author_relationships(self, paper_id: str, authors: List[str]) -> None:
        """
        Add author nodes and connect them to a paper.
        
        Args:
            paper_id: Unique ID of the paper
            authors: List of author names
        """
        cypher = """
        MATCH (p:Paper {id: $paper_id})
        MERGE (a:Author {name: $author_name})
        MERGE (a)-[:AUTHORED]->(p)
        """
        
        with self.driver.session() as session:
            for author in authors:
                session.run(cypher, {"paper_id": paper_id, "author_name": author})
        
        logger.info(f"Added {len(authors)} authors for paper {paper_id}")
    
    def add_entities(self, paper_id: str, entity_type: str, entities: List[Dict[str, Any]]) -> None:
        """
        Add entity nodes of a specific type and connect to paper.
        
        Args:
            paper_id: Unique ID of the paper
            entity_type: Type of entity (Concept, Method, Technology, Dataset)
            entities: List of entity objects
        """
        cypher = f"""
        MATCH (p:Paper {{id: $paper_id}})
        MERGE (e:{entity_type} {{name: $name}})
        SET e.description = $description
        MERGE (p)-[r:CONTAINS_{entity_type.upper()}]->(e)
        SET r.context = $context
        """
        
        with self.driver.session() as session:
            for entity in entities:
                session.run(cypher, {
                    "paper_id": paper_id,
                    "name": entity["name"],
                    "description": entity.get("description", ""),
                    "context": entity.get("context", "")
                })
        
        logger.info(f"Added {len(entities)} {entity_type} entities for paper {paper_id}")
    
    def add_paper_relationship(
        self,
        paper_id1: str,
        paper_id2: str,
        relationship_type: str,
        description: str,
        confidence: float
    ) -> None:
        """
        Add a relationship between two papers.
        
        Args:
            paper_id1: ID of the first paper
            paper_id2: ID of the second paper
            relationship_type: Type of relationship (e.g., "BUILDS_UPON")
            description: Description of the relationship
            confidence: Confidence score (0-1)
        """
        cypher = """
        MATCH (p1:Paper {id: $paper_id1})
        MATCH (p2:Paper {id: $paper_id2})
        MERGE (p1)-[r:RELATED_TO]->(p2)
        SET r.type = $rel_type,
            r.description = $description,
            r.confidence = $confidence
        """
        
        with self.driver.session() as session:
            session.run(cypher, {
                "paper_id1": paper_id1,
                "paper_id2": paper_id2,
                "rel_type": relationship_type,
                "description": description,
                "confidence": confidence
            })
        
        logger.info(f"Added '{relationship_type}' relationship between papers {paper_id1} and {paper_id2}")
    
    def add_citation(
        self,
        citing_paper_id: str,
        cited_paper_id: str,
        confidence: float,
        reason: str
    ) -> None:
        """
        Add a citation relationship between papers.
        
        Args:
            citing_paper_id: ID of the citing paper
            cited_paper_id: ID of the cited paper
            confidence: Confidence score (0-1)
            reason: Reason for inferring this citation
        """
        cypher = """
        MATCH (citing:Paper {id: $citing_id})
        MATCH (cited:Paper {id: $cited_id})
        MERGE (citing)-[r:CITES]->(cited)
        SET r.confidence = $confidence,
            r.reason = $reason
        """
        
        with self.driver.session() as session:
            session.run(cypher, {
                "citing_id": citing_paper_id,
                "cited_id": cited_paper_id,
                "confidence": confidence,
                "reason": reason
            })
        
        logger.info(f"Added citation from paper {citing_paper_id} to {cited_paper_id}")
    
    def create_time_period(self, period_name: str) -> None:
        """
        Create a time period node.
        
        Args:
            period_name: Name of the time period (e.g., "2020-2021")
        """
        cypher = """
        MERGE (p:TimePeriod {name: $period})
        """
        
        with self.driver.session() as session:
            session.run(cypher, {"period": period_name})
        
        logger.info(f"Created time period node: {period_name}")
    
    def connect_paper_to_period(self, paper_id: str, period_name: str) -> None:
        """
        Connect a paper to a time period.
        
        Args:
            paper_id: ID of the paper
            period_name: Name of the time period
        """
        cypher = """
        MATCH (paper:Paper {id: $paper_id})
        MATCH (period:TimePeriod {name: $period_name})
        MERGE (paper)-[:PUBLISHED_IN]->(period)
        """
        
        with self.driver.session() as session:
            session.run(cypher, {
                "paper_id": paper_id,
                "period_name": period_name
            })
        
        logger.info(f"Connected paper {paper_id} to time period {period_name}")
    
    def create_topic(self, topic_name: str, description: str) -> None:
        """
        Create a topic node.
        
        Args:
            topic_name: Name of the research topic
            description: Description of the topic
        """
        cypher = """
        MERGE (t:Topic {name: $name})
        SET t.description = $description
        """
        
        with self.driver.session() as session:
            session.run(cypher, {
                "name": topic_name,
                "description": description
            })
        
        logger.info(f"Created topic node: {topic_name}")
    
    def connect_paper_to_topic(self, paper_id: str, topic_name: str) -> None:
        """
        Connect a paper to a topic.
        
        Args:
            paper_id: ID of the paper
            topic_name: Name of the topic
        """
        cypher = """
        MATCH (p:Paper {id: $paper_id})
        MATCH (t:Topic {name: $topic_name})
        MERGE (p)-[:BELONGS_TO]->(t)
        """
        
        with self.driver.session() as session:
            session.run(cypher, {
                "paper_id": paper_id,
                "topic_name": topic_name
            })
        
        logger.info(f"Connected paper {paper_id} to topic {topic_name}")
    
    def get_methods_for_papers(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get all methods used across papers.
        
        Args:
            paper_ids: List of paper IDs
            
        Returns:
            List of method objects with name and usage count
        """
        cypher = """
        MATCH (p:Paper)-[:CONTAINS_METHOD]->(m:Method)
        WHERE p.id IN $paper_ids
        RETURN m.name, count(p) as usage_count
        ORDER BY usage_count DESC
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, {"paper_ids": paper_ids})
            methods = [{"name": record["m.name"], "count": record["usage_count"]} 
                      for record in result]
        
        return methods
    
    def get_concepts_for_papers(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get all concepts explored across papers.
        
        Args:
            paper_ids: List of paper IDs
            
        Returns:
            List of concept objects with name and usage count
        """
        cypher = """
        MATCH (p:Paper)-[:CONTAINS_CONCEPT]->(c:Concept)
        WHERE p.id IN $paper_ids
        RETURN c.name, count(p) as usage_count
        ORDER BY usage_count DESC
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, {"paper_ids": paper_ids})
            concepts = [{"name": record["c.name"], "count": record["usage_count"]} 
                       for record in result]
        
        return concepts
    
    def get_paper_relationships(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Get all relationships for a specific paper.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            List of relationship objects
        """
        cypher = """
        MATCH (p1:Paper {id: $paper_id})-[r:RELATED_TO]->(p2:Paper)
        RETURN p2.id AS related_paper_id, p2.title AS related_paper_title, 
               r.type AS relationship_type, r.description, r.confidence
        UNION
        MATCH (p1:Paper)-[r:RELATED_TO]->(p2:Paper {id: $paper_id})
        RETURN p1.id AS related_paper_id, p1.title AS related_paper_title,
               r.type AS relationship_type, r.description, r.confidence
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, {"paper_id": paper_id})
            relationships = [dict(record) for record in result]
        
        return relationships
    
    def get_citation_network(self, paper_ids: List[str]) -> Dict[str, List[str]]:
        """
        Get the citation network for a set of papers.
        
        Args:
            paper_ids: List of paper IDs
            
        Returns:
            Dictionary mapping paper IDs to lists of cited paper IDs
        """
        cypher = """
        MATCH (citing:Paper)-[r:CITES]->(cited:Paper)
        WHERE citing.id IN $paper_ids AND cited.id IN $paper_ids
        RETURN citing.id AS citing_id, cited.id AS cited_id
        """
        
        network = {paper_id: [] for paper_id in paper_ids}
        
        with self.driver.session() as session:
            result = session.run(cypher, {"paper_ids": paper_ids})
            for record in result:
                network[record["citing_id"]].append(record["cited_id"])
        
        return network
    
    def get_paper_metadata(self, paper_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific paper.
        
        Args:
            paper_id: ID of the paper
            
        Returns:
            Dictionary with paper metadata
        """
        cypher = """
        MATCH (p:Paper {id: $paper_id})
        OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
        RETURN p.id, p.title, p.year, p.venue, p.abstract, p.doi,
               collect(a.name) AS authors
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, {"paper_id": paper_id})
            record = result.single()
            if not record:
                return {}
            
            return {
                "id": record["p.id"],
                "title": record["p.title"],
                "year": record["p.year"],
                "venue": record["p.venue"],
                "abstract": record["p.abstract"],
                "doi": record["p.doi"],
                "authors": record["authors"]
            }
    
    def get_papers_by_topic(self, topic_name: str) -> List[Dict[str, Any]]:
        """
        Get all papers belonging to a specific topic.
        
        Args:
            topic_name: Name of the topic
            
        Returns:
            List of paper metadata dictionaries
        """
        cypher = """
        MATCH (p:Paper)-[:BELONGS_TO]->(t:Topic {name: $topic_name})
        OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
        RETURN p.id, p.title, p.year, p.abstract,
               collect(a.name) AS authors
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, {"topic_name": topic_name})
            papers = []
            for record in result:
                papers.append({
                    "id": record["p.id"],
                    "title": record["p.title"],
                    "year": record["p.year"],
                    "abstract": record["p.abstract"],
                    "authors": record["authors"]
                })
        
        return papers
    
    def get_papers_by_time_period(self, period_name: str) -> List[Dict[str, Any]]:
        """
        Get all papers published in a specific time period.
        
        Args:
            period_name: Name of the time period
            
        Returns:
            List of paper metadata dictionaries
        """
        cypher = """
        MATCH (p:Paper)-[:PUBLISHED_IN]->(t:TimePeriod {name: $period_name})
        RETURN p.id, p.title, p.year, p.abstract
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, {"period_name": period_name})
            papers = [dict(record) for record in result]
        
        return papers
    
    def get_entities_for_paper(self, paper_id: str, entity_type: str) -> List[Dict[str, Any]]:
        """
        Get all entities of a specific type for a paper.
        
        Args:
            paper_id: ID of the paper
            entity_type: Type of entity (Concept, Method, Technology, Dataset)
            
        Returns:
            List of entity dictionaries
        """
        cypher = f"""
        MATCH (p:Paper {{id: $paper_id}})-[r:CONTAINS_{entity_type.upper()}]->(e:{entity_type})
        RETURN e.name, e.description, r.context
        """
        
        with self.driver.session() as session:
            result = session.run(cypher, {"paper_id": paper_id})
            entities = []
            for record in result:
                entities.append({
                    "name": record["e.name"],
                    "description": record["e.description"],
                    "context": record["r.context"]
                })
        
        return entities