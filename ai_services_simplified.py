"""
AI services module for translation and language-related tasks.
Uses Claude by default for all operations.
"""

import os
import json
import logging
import re
from typing import Dict, List, Optional, Any

# Anthropic imports
import anthropic
from anthropic import Anthropic

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Anthropic client
anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
if anthropic_api_key:
    # The newest Anthropic model is "claude-3-5-sonnet-20241022"
    anthropic_client = Anthropic(api_key=anthropic_api_key)
else:
    anthropic_client = None
    logger.warning("Anthropic API key not found. Anthropic services will not be available.")

def translate_with_anthropic(text: str, target_language: str) -> Optional[str]:
    """
    Translate text using Anthropic (Claude).
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code or name
        
    Returns:
        str: Translated text or None if translation fails
    """
    if not anthropic_client:
        logger.warning("Anthropic client not available for translation.")
        return None
        
    try:
        prompt = (
            f"Translate the following text to {target_language}. "
            f"Provide only the translation, no comments or explanations:\n\n{text}"
        )
        
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    except Exception as e:
        logger.error(f"Error translating with Anthropic: {e}")
        return None

def detect_language_with_ai(text: str) -> Optional[str]:
    """
    Detect language using Claude.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Detected language code or None if detection fails
    """
    # First try standard language detection libraries
    try:
        from langdetect import detect
        lang_code = detect(text)
        if lang_code:
            logger.info(f"Used langdetect for language detection: {lang_code}")
            return lang_code
    except Exception as e:
        logger.error(f"Error detecting language with langdetect: {e}")
    
    # If standard library fails, try Anthropic
    if anthropic_client:
        try:
            prompt = (
                "Detect the language of the following text. "
                "Return only the ISO 639-1 language code (e.g., 'en', 'es', 'fr', etc.):\n\n"
                f"{text}"
            )
            
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                temperature=0.3,
                max_tokens=10,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            lang_code = response.content[0].text.strip().lower()
            # Extract only the language code if there's additional text
            if len(lang_code) > 5:
                # Try to find a 2-letter code
                match = re.search(r'\b[a-z]{2}\b', lang_code)
                if match:
                    lang_code = match.group(0)
                else:
                    # Just take the first two characters
                    lang_code = lang_code[:2]
                    
            logger.info(f"Used Anthropic for language detection: {lang_code}")
            return lang_code
        except Exception as e:
            logger.error(f"Error detecting language with Anthropic: {e}")
    
    return None

def translate_text_with_fallback(text: str, target_languages: List[str]) -> Dict[str, str]:
    """
    Translate text to multiple languages using Claude.
    
    Args:
        text (str): Text to translate
        target_languages (list): List of target language codes
        
    Returns:
        dict: Dictionary mapping language codes to translations
    """
    translations = {}
    
    for lang in target_languages:
        # First try standard translator
        try:
            from translator import translate_text
            translation = translate_text(text, None, lang)
            if translation and translation != text:
                translations[lang] = translation
                logger.info(f"Used standard translator for translating to {lang}")
                continue
        except Exception as e:
            logger.error(f"Error with standard translator service: {e}")
        
        # If standard translator fails, try Anthropic
        translation = translate_with_anthropic(text, lang)
        if translation:
            translations[lang] = translation
            logger.info(f"Used Anthropic for translating to {lang}")
            continue
            
        # If all services fail, use original text as fallback
        logger.warning(f"All translation services failed for {lang}. Using original text.")
        translations[lang] = text
    
    return translations

def generate_learning_example(word: str, lang_code: str, level: str = 'intermediate') -> Dict[str, Any]:
    """
    Generate learning examples for a word using Claude.
    
    Args:
        word (str): Word to generate examples for
        lang_code (str): Language code
        level (str): Learning level (beginner, intermediate, advanced)
        
    Returns:
        dict: Dictionary with example sentence, translation, and explanation
    """
    # Default response
    default_response = {
        'word': word,
        'example': '',
        'translation': '',
        'explanation': ''
    }
    
    # Use Anthropic
    if anthropic_client:
        try:
            prompt = (
                f"Create a learning example for the word '{word}' in {lang_code} language at {level} level.\n"
                f"Include an example sentence using the word, an English translation, "
                f"and a brief explanation of the word usage. Format as JSON with these keys: "
                f"'example', 'translation', 'explanation'"
            )
            
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                temperature=0.7,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the JSON from the response
            json_match = re.search(r'\{.*\}', response.content[0].text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                result['word'] = word
                logger.info(f"Used Anthropic for generating learning example for '{word}'")
                return result
        except Exception as e:
            logger.error(f"Error generating learning example with Anthropic: {e}")
    
    return default_response

def generate_thematic_vocabulary(theme: str, lang_code: str, level: str = 'intermediate', count: int = 10) -> List[Dict[str, Any]]:
    """
    Generate thematic vocabulary list using Claude.
    
    Args:
        theme (str): Theme for vocabulary (e.g., "travel", "food", "business")
        lang_code (str): Language code
        level (str): Learning level (beginner, intermediate, advanced)
        count (int): Number of words to generate
        
    Returns:
        list: List of vocabulary items with word, translation, and example
    """
    # Default empty list
    default_response = []
    
    # Use Anthropic
    if anthropic_client:
        try:
            prompt = (
                f"Create a list of {count} {level}-level vocabulary words related to '{theme}' in {lang_code} language.\n"
                f"For each word, include the word itself, English translation, and a simple example sentence.\n"
                f"Format as a JSON array with objects containing 'word', 'translation', and 'example' keys."
            )
            
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                temperature=0.7,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the JSON from the response
            json_match = re.search(r'\[.*\]', response.content[0].text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                logger.info(f"Used Anthropic for generating thematic vocabulary for '{theme}'")
                return result
        except Exception as e:
            logger.error(f"Error generating thematic vocabulary with Anthropic: {e}")
    
    return default_response

def query_claude(prompt: str, max_tokens: int = 4000, temperature: float = 0.5) -> Optional[str]:
    """
    Send a general query to Claude and get the response.
    Falls back to Grok if Claude is overloaded or unavailable.
    
    Args:
        prompt (str): The prompt to send to Claude
        max_tokens (int): Maximum number of tokens in the response
        temperature (float): Temperature for response generation
        
    Returns:
        str: Claude's or Grok's response or None if all queries fail
    """
    # Try Claude first
    if anthropic_client:
        try:
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            logger.info("Successfully got response from Claude")
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error querying Claude: {e}")
            
            # If Claude returns overload error (HTTP 529), try Grok as fallback
            if "529" in str(e) or "overloaded" in str(e).lower():
                logger.info("Claude is overloaded, trying Grok as fallback")
                try:
                    from xai import chat_with_grok
                    grok_response = chat_with_grok(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    if grok_response:
                        logger.info("Successfully got response from Grok (fallback)")
                        return grok_response
                    else:
                        logger.error("Grok fallback also failed")
                except Exception as grok_err:
                    logger.error(f"Error querying Grok as fallback: {grok_err}")
    else:
        logger.warning("Anthropic client not available for querying.")
        
        # Try Grok directly if Claude is not configured
        try:
            from xai import chat_with_grok, is_available
            if is_available():
                grok_response = chat_with_grok(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                if grok_response:
                    logger.info("Successfully got response from Grok (primary)")
                    return grok_response
        except Exception as grok_err:
            logger.error(f"Error querying Grok: {grok_err}")
    
    return None

def is_ai_service_available() -> bool:
    """Check if any AI service is available (Anthropic or Grok)."""
    # First check if Anthropic is available
    anthropic_available = anthropic_client is not None
    
    # Then check if Grok is available as a fallback
    try:
        from xai import is_available as is_grok_available
        grok_available = is_grok_available()
    except ImportError:
        grok_available = False
    
    return anthropic_available or grok_available