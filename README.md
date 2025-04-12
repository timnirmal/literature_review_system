https://claude.ai/share/a5ed5324-7aa5-473c-a77b-35f796c32e03

# Literature Review System - Modular Implementation

I've reorganized the Literature Review System into a modular structure that makes it more maintainable, flexible, and easier to extend. This implementation separates concerns, makes each component model-agnostic, and provides both a command-line interface and a REST API.

## Project Structure
```
literature_review_system/
├── config.py                 # Configuration settings
├── prompts.py                # LLM prompts
├── models.py                 # Model definitions and interface
├── database/
│   ├── __init__.py
│   ├── neo4j_manager.py      # Neo4j functions
│   ├── supabase_manager.py   # Supabase functions
├── components/
│   ├── __init__.py
│   ├── document_processor.py # Document processing
│   ├── entity_extractor.py   # Entity extraction
│   ├── graph_builder.py      # Knowledge graph construction
│   ├── review_generator.py   # Literature review generation
├── main.py                   # System initialization and workflow
├── app.py                    # API endpoints
└── requirements.txt          # Dependencies
```

## Key Features

1. Model Agnostic: Support for different LLM providers (OpenAI, Anthropic Claude, Google Gemini) with a unified interface.
2. Modular Components:
    - DocumentProcessor: Extracts text and metadata from PDFs
    - EntityExtractor: Identifies key entities from papers
    - KnowledgeGraphBuilder: Constructs and analyzes the knowledge graph
    - ReviewGenerator: Creates comprehensive literature reviews

3. Database Abstraction:
    - SupabaseManager: Handles document storage and retrieval
    - Neo4jManager: Manages the knowledge graph operations

4. Prompt Management: All prompts are centralized in a dedicated file for easy updates.

5. Multiple Interfaces:
    - Command-Line: For direct usage via main.py
    - REST API: For integration with web applications



## Usage
### Running from Command Line
```bash
# Process PDFs and generate a review
python main.py --pdf_folder ./papers --topic "Machine Learning in Healthcare"

# Use existing papers (already processed)
python main.py --topic "Machine Learning in Healthcare" --paper_ids "id1,id2,id3"

# Specify a different model
python main.py --pdf_folder ./papers --topic "AI Ethics" --model "claude-3-opus-20240229"
```
## Using the API
The system exposes a FastAPI-based REST API:

1. Upload papers:
```
POST /papers/upload
```

2. Extract entities:
```
POST /papers/entities
```

3. Build knowledge graph:
```
POST /graph/build
```

4. Generate literature review:
```
POST /reviews/generate
```

5. Download a generated review:
```
GET /reviews/download/{task_id}
```

## Benefits of this Implementation

1. Separation of Concerns: Each component has a specific responsibility, making the code easier to understand and maintain.
2. Flexibility: Easy to swap out components or add new ones.
3. Model Agnosticism: Support for various LLM providers through a unified interface.
4. API-First Design: The system can be integrated with other applications or front-ends.
5. Background Processing: Long-running tasks are handled in the background, allowing for asynchronous operation.
6. Reusability: Components can be used independently or together as needed.

## Getting Started

1. Install dependencies:
```
bash
pip install -r requirements.txt
```

2. Set up environment variables in a .env file:
```
OPENAI_API_KEY=your_openai_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

3. Run the API server:

```bash
python app.py
```

4. Or use the command-line interface:
```bash
python main.py --pdf_folder ./papers --topic "Your Research Topic"
```








Supabase Table Schema
Here's the table schema you'll need to set up in Supabase for the Literature Review System:
1. papers
```sql
create table public.papers (
  id uuid primary key,
  title text not null,
  authors jsonb not null default '[]',
  year integer,
  venue text,
  doi text,
  abstract text,
  full_text text,
  processed boolean default false,
  source_file text,
  created_at timestamp with time zone default now()
);

-- Add index for faster search
create index papers_title_abstract_idx on public.papers using gin (to_tsvector('english', title || ' ' || abstract));
create index papers_processed_idx on public.papers (processed);
```

2. paper_entities
```sql
create table public.paper_entities (
  id serial primary key,
  paper_id uuid references public.papers(id) on delete cascade,
  entities jsonb not null,
  created_at timestamp with time zone default now()
);

-- Add index for faster lookups
create index paper_entities_paper_id_idx on public.paper_entities (paper_id);
```

3. review_structures
```sql
create table public.review_structures (
  id serial primary key,
  topic text not null,
  structure jsonb not null,
  paper_ids jsonb not null,
  created_at timestamp with time zone default now()
);

-- Add index for faster lookups
create index review_structures_topic_idx on public.review_structures (topic);
```

4. research_gaps
```sql
create table public.research_gaps (
  id serial primary key,
  topic text not null,
  gaps jsonb not null,
  paper_ids jsonb not null,
  created_at timestamp with time zone default now()
);

-- Add index for faster lookups
create index research_gaps_topic_idx on public.research_gaps (topic);
```

5. research_trends
```sql
create table public.research_trends (
  id serial primary key,
  trends jsonb not null,
  paper_ids jsonb not null,
  created_at timestamp with time zone default now()
);
```

6. literature_reviews
```sql
create table public.literature_reviews (
  id serial primary key,
  topic text not null,
  content text not null,
  paper_ids jsonb not null,
  created_at timestamp with time zone default now()
);

-- Add index for faster lookups
create index literature_reviews_topic_idx on public.literature_reviews (topic);
```


6. Understanding the Schema
The Literature Review System will automatically set up the following schema in Neo4j:

```
Nodes:

Paper: Academic papers
Author: Paper authors
Concept: Theoretical frameworks and ideas
Method: Research methodologies
Technology: Tools and technologies
Dataset: Data sources
Topic: Research topics
TimePeriod: Chronological organization


Relationships:

AUTHORED: Connects authors to papers
CONTAINS_CONCEPT: Connects papers to concepts
CONTAINS_METHOD: Connects papers to methods
CONTAINS_TECHNOLOGY: Connects papers to technologies
CONTAINS_DATASET: Connects papers to datasets
RELATED_TO: Connects papers to related papers
CITES: Represents citation relationships
BELONGS_TO: Connects papers to topics
PUBLISHED_IN: Connects papers to time periods
```



7. Basic Neo4j Queries
Once your system is running and populated, you can explore the knowledge graph with queries like:

```sql
// View all papers
MATCH (p:Paper) RETURN p LIMIT 10;

// Find related papers
MATCH (p:Paper {title: "Your Paper Title"})-[r:RELATED_TO]->(related:Paper)
RETURN related.title, r.type, r.confidence;

// See topics and papers
MATCH (t:Topic)<-[:BELONGS_TO]-(p:Paper)
RETURN t.name, count(p) as paper_count
ORDER BY paper_count DESC;

// View citation network
MATCH (p1:Paper)-[r:CITES]->(p2:Paper)
RETURN p1.title, p2.title, r.confidence
ORDER BY r.confidence DESC;
```










