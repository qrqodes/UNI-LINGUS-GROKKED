"""
XAI (Grok) integration for the Enhanced Language Learning Bot.
Provides access to Grok models for AI assistance.
"""

import os
import logging
from typing import Optional, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

def get_xai_client():
    """Get XAI client with proper configuration."""
    try:
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            logger.warning("XAI_API_KEY not found in environment variables")
            return None
            
        client = OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=api_key
        )
        return client
    except Exception as e:
        logger.error(f"Failed to create XAI client: {e}")
        return None

def query_grok(prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> Optional[str]:
    """
    Query Grok AI with a prompt.
    
    Args:
        prompt: The prompt to send to Grok
        max_tokens: Maximum tokens in response
        temperature: Temperature for response generation
        
    Returns:
        Grok's response or None if failed
    """
    try:
        client = get_xai_client()
        if not client:
            return None
            
        response = client.chat.completions.create(
            model="grok-2-1212",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Grok query failed: {e}")
        return None

def translate_with_grok(text: str, target_language: str) -> Optional[str]:
    """
    Translate text using Grok.
    
    Args:
        text: Text to translate
        target_language: Target language name or code
        
    Returns:
        Translated text or None if failed
    """
    try:
        prompt = f"Translate the following text to {target_language}. Only return the translation, nothing else:\n\n{text}"
        return query_grok(prompt, max_tokens=500, temperature=0.3)
    except Exception as e:
        logger.error(f"Grok translation failed: {e}")
        return None

def chat_with_grok(messages: list, max_tokens: int = 1000, temperature: float = 0.7) -> Optional[str]:
    """
    Chat with Grok using message history.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content'
        max_tokens: Maximum tokens in response
        temperature: Temperature for response generation
        
    Returns:
        Grok's response or None if failed
    """
    try:
        client = get_xai_client()
        if not client:
            return None
            
        response = client.chat.completions.create(
            model="grok-2-1212",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Grok chat failed: {e}")
        return None

def is_grok_available() -> bool:
    """Check if Grok is available."""
    try:
        client = get_xai_client()
        return client is not None
    except:
        return False