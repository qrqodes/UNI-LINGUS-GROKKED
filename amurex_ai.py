"""
Amurex AI integration for the Enhanced Language Learning Bot.
Provides a simple interface to use Anthropic's Claude models via amurex.
"""

import logging
import os
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

try:
    from amurex import get_client, ClientType, ResponseType, ModelFamily
    AMUREX_AVAILABLE = True
except ImportError:
    logger.warning("Amurex library not found. Claude integration via amurex will not be available.")
    AMUREX_AVAILABLE = False

def is_available() -> bool:
    """Check if amurex is available and properly configured."""
    return AMUREX_AVAILABLE and os.environ.get("ANTHROPIC_API_KEY") is not None

async def chat_with_claude(messages: List[Dict[str, str]], 
                    temperature: float = 0.7,
                    max_tokens: int = 1000) -> Optional[str]:
    """
    Chat with Claude using amurex library.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        temperature: Temperature for response generation (0.0-1.0)
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        str: Claude's response or None if the query fails
    """
    if not is_available():
        logger.error("Amurex integration not available. Cannot chat with Claude.")
        return None
    
    try:
        # Get client using the API key from environment variables
        client = get_client(ClientType.Anthropic)
        
        # Convert messages to the format expected by amurex
        amurex_messages = []
        system_prompt = None
        
        # Extract system prompt if present
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content")
            elif msg.get("role") == "user":
                amurex_messages.append({"role": "user", "content": msg.get("content", "")})
            elif msg.get("role") == "assistant":
                amurex_messages.append({"role": "assistant", "content": msg.get("content", "")})
        
        # Send the chat request
        response = client.chat(
            model=ModelFamily.Claude35Sonnet.value,  # Use the latest Claude model
            system=system_prompt,  # Pass system prompt separately
            messages=amurex_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=ResponseType.TEXT
        )
        
        # Return the text response
        return response.content[0].text
    
    except Exception as e:
        logger.error(f"Error using amurex to chat with Claude: {e}")
        return None

def translate_with_claude(text: str, target_language: str) -> Optional[str]:
    """
    Translate text using Claude via amurex.
    
    Args:
        text: Text to translate
        target_language: Target language name or code
        
    Returns:
        str: Translated text or None if translation fails
    """
    if not is_available():
        logger.error("Amurex integration not available. Cannot translate with Claude.")
        return None
    
    try:
        # Get client using the API key from environment variables
        client = get_client(ClientType.Anthropic)
        
        # Create translation prompt
        prompt = (
            f"Translate the following text into {target_language}. "
            f"Provide ONLY the translated text with no explanations or notes.\n\n"
            f"Text to translate: {text}"
        )
        
        # Send the translation request
        response = client.generate(
            model=ModelFamily.Claude35Sonnet.value,  # Use the latest Claude model
            prompt=prompt,
            temperature=0.3,  # Lower temperature for more deterministic translation
            max_tokens=1000,
            response_format=ResponseType.TEXT
        )
        
        # Return the translation
        return response.content[0].text.strip()
    
    except Exception as e:
        logger.error(f"Error using amurex to translate with Claude: {e}")
        return None

def detect_language_with_claude(text: str) -> Optional[str]:
    """
    Detect language of text using Claude via amurex.
    
    Args:
        text: Text to analyze
        
    Returns:
        str: Detected language code (ISO 639-1) or None if detection fails
    """
    if not is_available():
        logger.error("Amurex integration not available. Cannot detect language with Claude.")
        return None
    
    try:
        # Get client using the API key from environment variables
        client = get_client(ClientType.Anthropic)
        
        # Create language detection prompt
        prompt = (
            f"Identify the language of the following text. "
            f"Respond with ONLY the ISO 639-1 two-letter language code (e.g., 'en', 'fr', 'zh').\n\n"
            f"Text: {text}"
        )
        
        # Send the language detection request
        response = client.generate(
            model=ModelFamily.Claude35Sonnet.value,  # Use the latest Claude model
            prompt=prompt,
            temperature=0.3,
            max_tokens=100,
            response_format=ResponseType.TEXT
        )
        
        # Return the detected language code
        return response.content[0].text.strip().lower()
    
    except Exception as e:
        logger.error(f"Error using amurex to detect language with Claude: {e}")
        return None

def extract_vocabulary(text: str) -> Optional[str]:
    """
    Extract vocabulary and language information from text using Claude via amurex.
    
    Args:
        text: Text to analyze
        
    Returns:
        str: Structured vocabulary information or None if extraction fails
    """
    if not is_available():
        logger.error("Amurex integration not available. Cannot extract vocabulary with Claude.")
        return None
    
    try:
        # Get client using the API key from environment variables
        client = get_client(ClientType.Anthropic)
        
        # Create vocabulary extraction prompt
        prompt = (
            "Please extract one interesting or useful word from the following text. "
            "Then provide a thorough definition, synonyms, antonyms (if any), and an explanatory example sentence using the word. "
            "Format your response like this:\n\n"
            "**Word:** [selected word]\n\n"
            "**Definition:** [thorough definition]\n\n"
            "**Synonyms:** [list of synonyms]\n\n"
            "**Antonyms:** [list of antonyms, or 'None' if none exist]\n\n"
            "**Example:** [clear example sentence using the word]\n\n"
            "**Usage Notes:** [any special notes about usage, context, register, etc.]\n\n"
            f"Text to analyze:\n{text}"
        )
        
        # Send the vocabulary extraction request
        response = client.generate(
            model=ModelFamily.Claude35Sonnet.value,  # Use the latest Claude model
            prompt=prompt,
            temperature=0.7,
            max_tokens=1000,
            response_format=ResponseType.TEXT
        )
        
        # Return the vocabulary information
        return response.content[0].text.strip()
    
    except Exception as e:
        logger.error(f"Error using amurex to extract vocabulary with Claude: {e}")
        return None