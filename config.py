"""
Configuration settings for the Literature Review System.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_ORG_ID = os.environ.get("OPENAI_ORG_ID", "")  # Optional

# Supabase Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Neo4j Configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

# Model Configuration
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gpt-4o")  # Default LLM
DEFAULT_EMBEDDING_MODEL = os.environ.get("DEFAULT_EMBEDDING_MODEL", "text-embedding-ada-002")

# Processing Configuration
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "30000"))  # Maximum chunk size for LLM processing
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))  # Maximum retries for API calls
RETRY_DELAY = int(os.environ.get("RETRY_DELAY", "1"))  # Seconds to wait between retries