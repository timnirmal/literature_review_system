"""
Model definitions and interface for the Literature Review System.

This module provides a unified interface for different LLM providers
(OpenAI, Claude, Gemini) using the OpenAI client format.
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable

import openai
import google.generativeai as genai
from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from config import OPENAI_API_KEY, MAX_RETRIES, RETRY_DELAY

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize clients
openai.api_key = OPENAI_API_KEY
anthropic_client = None  # Initialized on demand
genai_client = None  # Initialized on demand

# Model configurations
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4o": {"provider": "openai", "max_tokens": 128000, "supports_json": True},
    "gpt-4-turbo": {"provider": "openai", "max_tokens": 128000, "supports_json": True},
    "gpt-4": {"provider": "openai", "max_tokens": 8192, "supports_json": True},
    "gpt-3.5-turbo": {"provider": "openai", "max_tokens": 16384, "supports_json": True},
    
    # Anthropic models
    "claude-3-opus-20240229": {"provider": "anthropic", "max_tokens": 200000, "supports_json": True},
    "claude-3-sonnet-20240229": {"provider": "anthropic", "max_tokens": 200000, "supports_json": True},
    "claude-3-haiku-20240307": {"provider": "anthropic", "max_tokens": 200000, "supports_json": True},
    
    # Google models
    "gemini-1.5-pro": {"provider": "google", "max_tokens": 1000000, "supports_json": True},
    "gemini-1.0-pro": {"provider": "google", "max_tokens": 30720, "supports_json": True},
}

def initialize_anthropic(api_key: str) -> None:
    """Initialize the Anthropic client."""
    global anthropic_client
    anthropic_client = Anthropic(api_key=api_key)

def initialize_genai(api_key: str) -> None:
    """Initialize the Google Generative AI client."""
    global genai_client
    genai.configure(api_key=api_key)
    genai_client = genai

@retry(stop=stop_after_attempt(MAX_RETRIES), wait=wait_exponential(multiplier=1, min=RETRY_DELAY, max=10))
def call_model(
    prompt: str,
    model: str = "gpt-4o",
    system_prompt: str = "You are a helpful assistant that specializes in analyzing academic papers.",
    temperature: float = 0.1,
    response_format: Optional[Dict[str, str]] = None,
    functions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Call an LLM with unified interface across providers.
    
    Args:
        prompt: User prompt text
        model: Model identifier
        system_prompt: System instruction
        temperature: Randomness parameter (0-1)
        response_format: Format specification (e.g., {"type": "json_object"})
        functions: Function calling definitions
        
    Returns:
        Model response as string
    """
    if model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model}. Available models: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model]
    provider = config["provider"]
    
    # Request JSON format if supported
    if response_format and response_format.get("type") == "json_object" and not config["supports_json"]:
        logger.warning(f"Model {model} does not support JSON response format. Continuing without it.")
        response_format = None
    
    try:
        if provider == "openai":
            return _call_openai(prompt, model, system_prompt, temperature, response_format, functions)
        elif provider == "anthropic":
            return _call_anthropic(prompt, model, system_prompt, temperature, response_format)
        elif provider == "google":
            return _call_google(prompt, model, system_prompt, temperature, response_format)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        logger.error(f"Error calling {model}: {str(e)}")
        raise

def _call_openai(
    prompt: str,
    model: str,
    system_prompt: str,
    temperature: float,
    response_format: Optional[Dict[str, str]],
    functions: Optional[List[Dict[str, Any]]],
) -> str:
    """Call OpenAI API."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    if response_format:
        kwargs["response_format"] = response_format
    
    if functions:
        kwargs["functions"] = functions
    
    response = openai.chat.completions.create(**kwargs)
    return response.choices[0].message.content

def _call_anthropic(
    prompt: str,
    model: str,
    system_prompt: str,
    temperature: float,
    response_format: Optional[Dict[str, str]],
) -> str:
    """Call Anthropic API."""
    global anthropic_client
    if not anthropic_client:
        raise ValueError("Anthropic client not initialized. Call initialize_anthropic() first.")
    
    system = system_prompt
    
    kwargs = {
        "model": model,
        "max_tokens": 4096,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    
    if response_format and response_format.get("type") == "json_object":
        # Claude has a different way of requesting JSON
        system += " Always return your response in valid JSON format."
    
    response = anthropic_client.messages.create(**kwargs)
    return response.content[0].text

def _call_google(
    prompt: str,
    model: str,
    system_prompt: str,
    temperature: float,
    response_format: Optional[Dict[str, str]],
) -> str:
    """Call Google Generative AI API."""
    global genai_client
    if not genai_client:
        raise ValueError("Google Genai client not initialized. Call initialize_genai() first.")
    
    # Combine system prompt and user prompt for Gemini
    combined_prompt = f"{system_prompt}\n\n{prompt}"
    
    generation_config = {
        "temperature": temperature,
    }
    
    if response_format and response_format.get("type") == "json_object":
        # Add instruction for JSON format
        combined_prompt += "\n\nRespond only with a valid JSON object."
    
    model = genai_client.GenerativeModel(model_name=model, generation_config=generation_config)
    response = model.generate_content(combined_prompt)
    
    return response.text

def parse_json_response(response_text: str) -> Dict[str, Any]:
    """
    Parse a JSON response from an LLM, handling potential formatting issues.
    
    Args:
        response_text: Text response from the model
        
    Returns:
        Parsed JSON as a Python dictionary
    """
    # Try to extract JSON if it's wrapped in markdown code blocks
    if "```json" in response_text or "```" in response_text:
        try:
            # Extract content between code blocks
            json_text = response_text.split("```")[1]
            if json_text.startswith("json"):
                json_text = json_text[4:].strip()
            return json.loads(json_text)
        except (IndexError, json.JSONDecodeError):
            pass
    
    # Try direct parsing
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON response: {str(e)}")
        logger.debug(f"Response text: {response_text}")
        
        # Fall back to a simple object with error info
        return {
            "error": "Failed to parse JSON response",
            "text": response_text
        }

def get_embeddings(
    texts: List[str],
    model: str = "text-embedding-ada-002",
) -> List[List[float]]:
    """
    Get embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model: Embedding model to use
        
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    try:
        response = openai.embeddings.create(
            model=model,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"Error getting embeddings: {str(e)}")
        raise