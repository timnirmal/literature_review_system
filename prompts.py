"""
Prompts for the Literature Review System.

This module contains all the prompts used by different components
to guide LLM responses for different tasks in the system.
"""

# Document Processing Prompts
METADATA_EXTRACTION_PROMPT = """
Extract the following metadata from this academic paper:
- Title
- Authors (as a list)
- Publication Year
- Journal/Conference
- DOI (if present)
- Abstract

Format the response as a valid JSON object.

Paper text:
{text}
"""

# Entity Extraction Prompts
ENTITY_EXTRACTION_PROMPT = """
Extract the following entities from this academic paper as structured data:

1. CONCEPTS: Key theoretical frameworks, models, or philosophical approaches mentioned
2. METHODS: Experimental designs, data collection techniques, analytical approaches
3. FINDINGS: Key results, statistical findings, conclusions
4. TECHNOLOGIES: Tools, software, hardware, or equipment used
5. DATASETS: What data sources were utilized

For each entity, include:
- Name: A concise identifier for the entity
- Description: A brief explanation
- Context: How it is used in this paper
- Page/Section: Where it appears (if clear)

Format the response as a valid JSON object with these categories as keys, each containing an array of extracted entities.

Paper Title: {title}
Paper Abstract: {abstract}
Paper Text:
{text}
"""

# Knowledge Graph Prompts
PAPER_RELATIONSHIP_PROMPT = """
Analyze the relationship between these two academic papers:

Paper 1:
Title: {title1}
Abstract: {abstract1}

Paper 2:
Title: {title2}
Abstract: {abstract2}

Determine the most accurate relationship type from the following options:
- BUILDS_UPON: Paper 2 extends or builds upon Paper 1
- CONTRADICTS: Paper 2 challenges or contradicts findings in Paper 1
- VALIDATES: Paper 2 confirms or validates findings in Paper 1
- APPLIES: Paper 2 applies methods from Paper 1 in a new context
- COMPARES: Paper 2 directly compares with Paper 1
- UNRELATED: No clear relationship between the papers

Return your analysis as a JSON object with these fields:
- relationship_type: One of the options above
- description: Brief explanation of the relationship
- confidence: A number between 0 and 1 indicating confidence in this assessment
"""

CITATION_ANALYSIS_PROMPT = """
Based on the title and abstract of this paper from {citing_year}:

Title: {citing_title}
Abstract: {citing_abstract}

Analyze which of these older papers it likely cites:
{potential_cited_papers}

For each paper, indicate:
1. Whether it's likely cited (Yes/No)
2. The confidence level (0-1)
3. Brief reason

Format the response as a JSON object with a "citations" array containing objects with:
- paper_idx: The index number of the paper (1-based)
- cited: true or false
- confidence: 0-1 value
- reason: Brief explanation
"""

TIME_PERIOD_SUMMARY_PROMPT = """
Summarize the main research developments during {time_period} based on these papers:

{papers_text}

In your summary:
1. Identify key themes from this period
2. Note methodological approaches that were common
3. Highlight important findings or breakthroughs
4. Describe how the field evolved during this time

Format the response as a JSON object with:
- key_themes: List of main research themes
- methodological_trends: Common approaches
- significant_findings: Important results
- evolution: How the field progressed
"""

# Literature Review Prompts
TOPIC_CLUSTERING_PROMPT = """
Review these academic papers and cluster them into coherent research topics:

{papers_text}

Group the papers into 3-7 distinct research topics. For each topic:
1. Provide a descriptive name
2. List the papers that belong to this topic (using their Paper ID)
3. Explain why these papers form a coherent group

Some papers may belong to multiple topics if they bridge research areas.

Return the results as a JSON object where:
- Keys are topic names
- Values are objects containing:
  - paper_ids: Array of paper IDs in this topic
  - description: Brief description of this topic cluster
"""

NARRATIVE_STRUCTURE_PROMPT = """
Create a logical narrative structure for a literature review on "{topic}" based on these papers:

{papers_text}

Design a comprehensive structure with:
1. Introduction section
2. Several thematic sections that group papers by research focus
3. Methodological approaches section
4. Findings synthesis section
5. Gap analysis section
6. Future directions section

For each section:
1. Provide a title
2. List which paper IDs should be included
3. Explain what aspects of those papers should be highlighted
4. Suggest a logical order for discussing the papers

Return the results as a JSON object where:
- Keys are section names (e.g., "Introduction", "Theme 1: [Name]", etc.)
- Values are objects containing:
  - paper_ids: Array of paper IDs to include
  - focus_points: Aspects to highlight
  - paper_order: Suggested order for discussing papers
  - section_purpose: Brief explanation of this section's purpose
"""

RESEARCH_GAPS_PROMPT = """
Identify research gaps in the literature on "{topic}" based on this analysis:

METHODS USED:
{methods_text}

CONCEPTS EXPLORED:
{concepts_text}

PAPERS OVERVIEW:
{papers_text}

Identify 5-10 specific research gaps, considering:
1. Theoretical gaps (missing explanatory frameworks)
2. Empirical gaps (untested predictions)
3. Methodological gaps (limitations in approaches)
4. Population gaps (understudied subjects or contexts)
5. Integration gaps (disconnected research streams)

For each gap:
1. Provide a clear description
2. Explain why it's significant
3. Suggest how future research could address it

Return the results as a JSON object with a "gaps" array of gap objects, each containing:
- gap_type: Type of gap (theoretical, empirical, methodological, etc.)
- description: Clear description of the gap
- significance: Why this gap matters
- future_research: How it could be addressed
"""

RESEARCH_TRENDS_PROMPT = """
Analyze the evolution of research trends based on these papers organized by year:

{years_text}

Identify:
1. Emerging research themes (topics gaining attention)
2. Declining research areas (topics receiving less attention)
3. Methodological evolution (how research approaches changed)
4. Conceptual shifts (how key concepts evolved)
5. Overall trajectory of the field

For each trend, provide:
1. Description of the trend
2. Years when it's observable
3. Supporting evidence from the papers
4. Significance for the field

Return the results as a JSON object with these categories:
- emerging_themes: Array of emerging research themes
- declining_areas: Array of declining research areas
- methodological_evolution: Description of how methods changed
- conceptual_shifts: Description of how concepts evolved
- overall_trajectory: Summary of field direction
"""

SECTION_GENERATION_PROMPT = """
Write a section for a literature review on "{topic}" with the title "{section_name}".

Section purpose: {section_purpose}
Aspects to focus on: {focus_points}

Papers to include:
{all_papers_text}

Write a cohesive, scholarly section that:
1. Has a clear introduction and conclusion
2. Synthesizes information across papers rather than summarizing each separately
3. Highlights connections, agreements, and disagreements between works
4. Uses academic tone and appropriate citations (e.g., Smith et al., 2020)
5. Is approximately 500-800 words in length

Format the output as scholarly prose with appropriate paragraphs.
"""