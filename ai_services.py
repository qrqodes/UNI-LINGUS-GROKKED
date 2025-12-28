"""
AI services module for translation and language-related tasks.
Provides fallback mechanisms using multiple AI providers.
"""

import os
import json
import logging
import sys
from typing import Dict, List, Optional, Any, Tuple

# OpenAI imports
import openai
from openai import OpenAI

# Anthropic imports
import anthropic
from anthropic import Anthropic

# Try to import open source models
try:
    from open_source_ai import (
        translate_with_deepseek, translate_with_llama, translate_with_mixtral,
        detect_language_with_open_source, is_open_source_model_available
    )
    OPEN_SOURCE_AVAILABLE = True
except ImportError:
    OPEN_SOURCE_AVAILABLE = False
    logging.warning("Open source AI models not available.")

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
openai_api_key = os.environ.get('OPENAI_API_KEY')
if openai_api_key:
    # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
    # do not change this unless explicitly requested by the user
    openai_client = OpenAI(api_key=openai_api_key)
else:
    openai_client = None
    logger.warning("OpenAI API key not found. OpenAI services will not be available.")

# Initialize Anthropic client
anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
if anthropic_api_key:
    #the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
    anthropic_client = Anthropic(api_key=anthropic_api_key)
else:
    anthropic_client = None
    logger.warning("Anthropic API key not found. Anthropic services will not be available.")

def translate_with_openai(text: str, target_language: str) -> Optional[str]:
    """
    Translate text using OpenAI.
    
    Args:
        text (str): Text to translate
        target_language (str): Target language code or name
        
    Returns:
        str: Translated text or None if translation fails
    """
    if not openai_client:
        logger.warning("OpenAI client not available for translation.")
        return None
        
    try:
        prompt = (
            f"Translate the following text to {target_language}. "
            f"Provide only the translation, no comments or explanations:\n\n{text}"
        )
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error translating with OpenAI: {e}")
        return None

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
    Detect language using AI services.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Detected language code or None if detection fails
    """
    # First try with open source models if available
    if OPEN_SOURCE_AVAILABLE:
        try:
            lang_code = detect_language_with_open_source(text)
            if lang_code:
                logger.info(f"Used open source model for language detection: {lang_code}")
                return lang_code
        except Exception as e:
            logger.error(f"Error detecting language with open source model: {e}")
    
    # Try standard language detection libraries
    try:
        from langdetect import detect
        lang_code = detect(text)
        if lang_code:
            logger.info(f"Used langdetect for language detection: {lang_code}")
            return lang_code
    except Exception as e:
        logger.error(f"Error detecting language with langdetect: {e}")
    
    # If open source and standard libraries fail, try OpenAI
    if openai_client:
        try:
            prompt = (
                "Detect the language of the following text. "
                "Return only the ISO 639-1 language code (e.g., 'en', 'es', 'fr', etc.):\n\n"
                f"{text}"
            )
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10
            )
            
            lang_code = response.choices[0].message.content.strip().lower()
            # Extract only the language code if there's additional text
            if len(lang_code) > 5:
                # Try to find a 2-letter code
                import re
                match = re.search(r'\b[a-z]{2}\b', lang_code)
                if match:
                    lang_code = match.group(0)
                else:
                    # Just take the first two characters
                    lang_code = lang_code[:2]
                    
            logger.info(f"Used OpenAI for language detection: {lang_code}")
            return lang_code
        except Exception as e:
            logger.error(f"Error detecting language with OpenAI: {e}")
    
    # Fall back to Anthropic if OpenAI fails
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
                import re
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
    Translate text to multiple languages with fallback between AI services.
    
    Args:
        text (str): Text to translate
        target_languages (list): List of target language codes
        
    Returns:
        dict: Dictionary mapping language codes to translations
    """
    translations = {}
    
    for lang in target_languages:
        # First try open source models if available
        if OPEN_SOURCE_AVAILABLE:
            # Try DeepSeek first (highest quality open source model)
            translation = translate_with_deepseek(text, lang)
            if translation:
                translations[lang] = translation
                # Log that DeepSeek was used
                logger.info(f"Used DeepSeek for translating to {lang}")
                continue
                
            # Try Llama 4 if DeepSeek fails
            translation = translate_with_llama(text, lang)
            if translation:
                translations[lang] = translation
                # Log that Llama was used
                logger.info(f"Used Llama for translating to {lang}")
                continue
                
            # Try Mixtral if Llama fails
            translation = translate_with_mixtral(text, lang)
            if translation:
                translations[lang] = translation
                # Log that Mixtral was used
                logger.info(f"Used Mixtral for translating to {lang}")
                continue
        
        # If open source models aren't available or fail, try the standard translator
        try:
            from translator import translate_text
            translation = translate_text(text, None, lang)
            if translation and translation != text:
                translations[lang] = translation
                # Log that Google Translate was used
                logger.info(f"Used Google Translate for translating to {lang}")
                continue
        except Exception as e:
            logger.error(f"Error with Google Translate service: {e}")
        
        # If Google Translate fails, try OpenAI
        translation = translate_with_openai(text, lang)
        if translation:
            translations[lang] = translation
            # Log that OpenAI was used
            logger.info(f"Used OpenAI for translating to {lang}")
            continue
            
        # If OpenAI fails, try Anthropic
        translation = translate_with_anthropic(text, lang)
        if translation:
            translations[lang] = translation
            # Log that Anthropic was used
            logger.info(f"Used Anthropic for translating to {lang}")
            continue
            
        # If all services fail, use original text as fallback
        logger.warning(f"All translation services failed for {lang}. Using original text.")
        translations[lang] = text
    
    return translations

def generate_learning_example(word: str, lang_code: str, level: str = 'intermediate') -> Dict[str, Any]:
    """
    Generate learning examples for a word using AI.
    
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
    
    # Try open source models first if available
    if OPEN_SOURCE_AVAILABLE:
        try:
            result = generate_learning_example_with_open_source(word, lang_code, level)
            if result and result.get('example') and result.get('translation'):
                logger.info(f"Used open source model for generating learning example for '{word}'")
                return result
        except Exception as e:
            logger.error(f"Error generating learning example with open source model: {e}")
    
    # If open source fails, try OpenAI
    if openai_client:
        try:
            prompt = (
                f"Create a learning example for the word '{word}' in {lang_code} language at {level} level.\n"
                f"Include an example sentence using the word, an English translation, "
                f"and a brief explanation of the word usage. Format as JSON with these keys: "
                f"'example', 'translation', 'explanation'"
            )
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            result['word'] = word
            logger.info(f"Used OpenAI for generating learning example for '{word}'")
            return result
        except Exception as e:
            logger.error(f"Error generating learning example with OpenAI: {e}")
    
    # Fall back to Anthropic if OpenAI fails
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
            import re
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
    Generate thematic vocabulary list using AI.
    
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
    
    # Try open source models first if available
    if OPEN_SOURCE_AVAILABLE:
        try:
            # First try DeepSeek as it's generally the most capable
            prompt = (
                f"Create a list of {count} {level}-level vocabulary words related to '{theme}' in {lang_code} language.\n"
                f"For each word, include the word itself, English translation, and a simple example sentence.\n"
                f"Format as a JSON array with objects containing 'word', 'translation', and 'example' keys."
            )
            
            # Try DeepSeek
            if DEEPSEEK_SERVER_URL:
                payload = {
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
                
                response = requests.post(
                    urljoin(DEEPSEEK_SERVER_URL, "/v1/completions"),
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        # Extract the JSON from the response
                        import re
                        json_match = re.search(r'\[.*\]', result["choices"][0]["text"], re.DOTALL)
                        if json_match:
                            vocabulary_list = json.loads(json_match.group(0))
                            logger.info(f"Used DeepSeek for generating thematic vocabulary for theme '{theme}'")
                            return vocabulary_list
            
            # If DeepSeek fails, try Llama
            if LLAMA_SERVER_URL:
                payload = {
                    "prompt": prompt,
                    "temperature": 0.7,
                    "max_tokens": 2000
                }
                
                response = requests.post(
                    urljoin(LLAMA_SERVER_URL, "/v1/completions"),
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        # Extract the JSON from the response
                        import re
                        json_match = re.search(r'\[.*\]', result["choices"][0]["text"], re.DOTALL)
                        if json_match:
                            vocabulary_list = json.loads(json_match.group(0))
                            logger.info(f"Used Llama for generating thematic vocabulary for theme '{theme}'")
                            return vocabulary_list
        except Exception as e:
            logger.error(f"Error generating thematic vocabulary with open source models: {e}")
    
    # If open source fails or is not available, try OpenAI
    if openai_client:
        try:
            prompt = (
                f"Create a list of {count} {level}-level vocabulary words related to '{theme}' in {lang_code} language.\n"
                f"For each word, include the word itself, English translation, and a simple example sentence.\n"
                f"Format as a JSON array with objects containing 'word', 'translation', and 'example' keys."
            )
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            if isinstance(result, dict) and 'words' in result:
                logger.info(f"Used OpenAI for generating thematic vocabulary for theme '{theme}'")
                return result['words']
            elif isinstance(result, list):
                logger.info(f"Used OpenAI for generating thematic vocabulary for theme '{theme}'")
                return result
            else:
                logger.error(f"Unexpected format from OpenAI: {result}")
                return default_response
        except Exception as e:
            logger.error(f"Error generating thematic vocabulary with OpenAI: {e}")
    
    # Fall back to Anthropic if OpenAI fails
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
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract the JSON from the response
            import re
            json_match = re.search(r'\[.*\]', response.content[0].text, re.DOTALL)
            if json_match:
                logger.info(f"Used Anthropic for generating thematic vocabulary for theme '{theme}'")
                return json.loads(json_match.group(0))
            else:
                json_match = re.search(r'\{.*\}', response.content[0].text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                    if 'words' in result:
                        logger.info(f"Used Anthropic for generating thematic vocabulary for theme '{theme}'")
                        return result['words']
            
            logger.error(f"Could not extract JSON from Anthropic response")
            return default_response
        except Exception as e:
            logger.error(f"Error generating thematic vocabulary with Anthropic: {e}")
    
    return default_response

def is_ai_service_available() -> bool:
    """Check if any AI service is available."""
    return (openai_client is not None or 
            anthropic_client is not None or
            (OPEN_SOURCE_AVAILABLE and is_open_source_model_available()))