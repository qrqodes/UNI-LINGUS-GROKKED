"""
DeepSeek AI integration for Enhanced Language Bot
Free and open-source AI model with competitive performance.
"""
import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

def translate_with_deepseek(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """
    Translate text using DeepSeek API
    
    Args:
        text: Text to translate
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        Translated text or None if translation fails
    """
    try:
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            logger.info("DeepSeek API key not found")
            return None
            
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Language name mapping for better results
        lang_names = {
            'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh-CN': 'Chinese',
            'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi'
        }
        
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        prompt = f"Translate this text from {source_name} to {target_name}. Only return the translation, no explanations:\n\n{text}"
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": 0.3
        }
        
        response = requests.post(
            'https://api.deepseek.com/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('choices') and len(result['choices']) > 0:
                translated = result['choices'][0]['message']['content'].strip()
                if translated and translated != text:
                    logger.info("Translation successful with DeepSeek")
                    return translated
                    
    except Exception as e:
        logger.warning(f"DeepSeek translation failed: {e}")
        
    return None

def chat_with_deepseek(message: str) -> Optional[str]:
    """
    Chat with DeepSeek AI model
    
    Args:
        message: User message
        
    Returns:
        AI response or None if failed
    """
    try:
        api_key = os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            logger.info("DeepSeek API key not found")
            return None
            
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": message}
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(
            'https://api.deepseek.com/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('choices') and len(result['choices']) > 0:
                response_text = result['choices'][0]['message']['content'].strip()
                if response_text:
                    logger.info("Chat response generated with DeepSeek")
                    return response_text
                    
    except Exception as e:
        logger.warning(f"DeepSeek chat failed: {e}")
        
    return None

def is_deepseek_available() -> bool:
    """Check if DeepSeek API is available"""
    return bool(os.environ.get('DEEPSEEK_API_KEY'))