"""
Enhanced translator module that adds transcription services and AI fallback mechanisms.
"""

import os
import json
import logging
import tempfile
from typing import Dict, List, Any, Optional, Tuple

# Import AI services and transcription modules
from ai_services import (
    translate_text_with_fallback, detect_language_with_ai, 
    is_ai_service_available, translate_with_openai, translate_with_anthropic
)
from transcription import (
    get_transcription, get_pinyin, transliterate_cyrillic,
    add_transcription_to_translations
)
from database import db

# Import primary translation libraries
from langdetect import detect
try:
    from googletrans import Translator
    translator = Translator()
    TRANSLATOR_AVAILABLE = True
except Exception as e:
    TRANSLATOR_AVAILABLE = False
    logging.error(f"GoogleTrans not available: {e}")

# Import TTS library
try:
    from gtts import gTTS
    TTS_AVAILABLE = True
except Exception as e:
    TTS_AVAILABLE = False
    logging.error(f"gTTS not available: {e}")

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Language settings
# Default languages as requested: English, Spanish, Portuguese, Italian, French, Russian, and Chinese
DEFAULT_LANGUAGES = ['en', 'es', 'pt', 'it', 'fr', 'ru', 'zh-CN']

# All supported languages - reduced list as requested
ALL_LANGUAGE_CODES = [
    'en', 'es', 'pt', 'it', 'fr', 'ru', 'zh-CN', 'de', 'ja', 'ko', 
    'ar', 'hi'
]

LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'pt': 'Portuguese',
    'it': 'Italian',
    'fr': 'French',
    'ru': 'Russian',
    'zh-CN': 'Chinese',
    'de': 'German',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi'
}

def get_language_name(lang_code: str) -> str:
    """Get the human-readable name for a language code."""
    return LANGUAGE_NAMES.get(lang_code, lang_code.title())

def detect_language(text: str) -> str:
    """
    Detect the language of the provided text using Claude -> DeepSeek -> Grok hierarchy.
    
    Args:
        text (str): Text to analyze
        
    Returns:
        str: Detected language code
    """
    # Try Claude first
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            prompt = f"Detect the language of this text and return only the ISO 639-1 language code (e.g., 'en', 'es', 'fr', 'zh-CN'). Text: {text}"
            
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}]
            )
            
            detected = message.content[0].text.strip().lower()
            if detected in ALL_LANGUAGE_CODES:
                logger.info(f"Claude detected language: {detected}")
                return detected
    except Exception as e:
        logger.warning(f"Claude language detection failed: {e}")
    
    # Try DeepSeek as fallback
    try:
        from deepseek_integration import is_deepseek_available
        if is_deepseek_available():
            # DeepSeek detection logic would go here
            pass
    except Exception as e:
        logger.warning(f"DeepSeek language detection failed: {e}")
    
    # Try Grok as third choice
    try:
        from xai import is_available as is_grok_available, detect_language_with_grok
        if is_grok_available():
            grok_lang = detect_language_with_grok(text)
            if grok_lang:
                # Convert 2-letter code to the format we use if needed
                if grok_lang == 'zh':
                    grok_lang = 'zh-CN'
                logger.info(f"Grok detected language: {grok_lang}")
                return grok_lang
    except ImportError:
        logger.warning("Grok module not available for language detection")
    except Exception as e:
        logger.error(f"Error using Grok for language detection: {e}")
    
    # Next try using other AI services
    if is_ai_service_available():
        ai_lang = detect_language_with_ai(text)
        if ai_lang:
            # Convert 2-letter code to the format we use if needed
            if ai_lang == 'zh':
                ai_lang = 'zh-CN'
            logger.info(f"AI detected language: {ai_lang}")
            return ai_lang
    
    # Fall back to langdetect as the last resort
    try:
        # Try langdetect
        lang_code = detect(text)
        
        # Convert 2-letter code to the format we use
        if lang_code == 'zh':
            lang_code = 'zh-CN'
            
        logger.info(f"Detected language using langdetect: {lang_code}")
        return lang_code
    except Exception as e:
        logger.error(f"Error with langdetect language detection: {e}")
    
    # If all fails, default to English
    logger.warning("All language detection methods failed, defaulting to English")
    return 'en'

def translate_text(text: str, source_lang: Optional[str], target_lang: str) -> str:
    """
    Translate text with fast Google Translate first, then AI fallbacks.
    
    Args:
        text (str): Text to translate
        source_lang (str): Source language code or None for auto-detection
        target_lang (str): Target language code
        
    Returns:
        str: Translated text
    """
    if not text:
        return ""
        
    # Detect source language if not provided
    if not source_lang:
        source_lang = detect_language(text)
        
    # Skip translation if source and target are the same
    if source_lang == target_lang:
        return text
    
    # FIRST: Use Google Translate for maximum speed (fastest possible)
    if TRANSLATOR_AVAILABLE:
        try:
            # Handle Chinese special case
            src_lang = source_lang
            tgt_lang = target_lang
            if source_lang == 'zh-CN':
                src_lang = 'zh-cn'
            if target_lang == 'zh-CN':
                tgt_lang = 'zh-cn'
                
            translation = translator.translate(
                text, 
                src=src_lang if src_lang != 'detect' else None,
                dest=tgt_lang
            )
            if translation and hasattr(translation, 'text'):
                logger.info(f"Used Google Translate for translation from {source_lang} to {target_lang} (FAST)")
                result = translation.text
                
                # Add transcriptions for Russian and Chinese
                if target_lang == 'ru':
                    from transcription import transliterate_cyrillic
                    try:
                        latin = transliterate_cyrillic(result)
                        if latin and latin != result:
                            result += f"\n[{latin}]"
                    except:
                        pass
                elif target_lang == 'zh-CN':
                    from transcription import get_pinyin
                    try:
                        pinyin = get_pinyin(result)
                        if pinyin and pinyin != result:
                            result += f"\n[{pinyin}]"
                    except:
                        pass
                
                return result
        except Exception as e:
            logger.error(f"Error with Google translation: {e}")
        
    # FALLBACK: Get translation from AI services if Google fails
    grok_translation = None
    try:
        from xai import is_available as is_grok_available, translate_with_grok
        if is_grok_available():
            grok_translation = translate_with_grok(text, LANGUAGE_NAMES.get(target_lang, target_lang))
            if grok_translation:
                logger.info(f"Grok provided fallback translation from {source_lang} to {target_lang}")
                
                # Step 2: Verify and improve with Claude
                try:
                    verified_translation = verify_translation_with_claude(
                        original_text=text,
                        translated_text=grok_translation,
                        source_lang=source_lang,
                        target_lang=target_lang
                    )
                    
                    if verified_translation:
                        logger.info(f"Claude verified/improved translation from {source_lang} to {target_lang}")
                        return verified_translation
                    else:
                        logger.info(f"Using Grok translation (Claude verification unavailable)")
                        return grok_translation
                        
                except Exception as e:
                    logger.error(f"Claude verification failed: {e}, using Grok translation")
                    return grok_translation
                    
    except ImportError:
        logger.warning("xAI module not available for translation")
    except Exception as e:
        logger.error(f"Error using Grok for translation: {e}")
    
    # If Grok fails, try other AI services
    if is_ai_service_available():
        # Try OpenAI 
        ai_translation = translate_with_openai(text, LANGUAGE_NAMES.get(target_lang, target_lang))
        if ai_translation:
            logger.info(f"Used OpenAI for translation from {source_lang} to {target_lang}")
            return ai_translation
            
        # If OpenAI fails, try Anthropic via amurex
        try:
            import amurex_ai
            if amurex_ai.is_available():
                ai_translation = amurex_ai.translate_with_claude(text, LANGUAGE_NAMES.get(target_lang, target_lang))
                if ai_translation:
                    logger.info(f"Used Amurex Claude for translation from {source_lang} to {target_lang}")
                    return ai_translation
        except ImportError:
            logger.warning("amurex_ai module not available for translation")
        except Exception as e:
            logger.error(f"Error using amurex for translation: {e}")
            
        # If amurex fails, try standard Anthropic
        ai_translation = translate_with_anthropic(text, LANGUAGE_NAMES.get(target_lang, target_lang))
        if ai_translation:
            logger.info(f"Used Anthropic Claude for translation from {source_lang} to {target_lang}")
            return ai_translation
    
    # Use Google Translate FIRST for speed (moved up from last resort)
    if TRANSLATOR_AVAILABLE:
        try:
            # Handle Chinese special case
            src_lang = source_lang
            tgt_lang = target_lang
            if source_lang == 'zh-CN':
                src_lang = 'zh-cn'
            if target_lang == 'zh-CN':
                tgt_lang = 'zh-cn'
                
            translation = translator.translate(
                text, 
                src=src_lang if src_lang != 'detect' else None,
                dest=tgt_lang
            )
            if translation and hasattr(translation, 'text'):
                logger.info(f"Used Google Translate for translation from {source_lang} to {target_lang} (FAST)")
                result = translation.text
                
                # Add transcriptions for Russian and Chinese
                if target_lang == 'ru':
                    from transcription import transliterate_cyrillic
                    try:
                        latin = transliterate_cyrillic(result)
                        if latin and latin != result:
                            result += f"\n[{latin}]"
                    except:
                        pass
                elif target_lang == 'zh-CN':
                    from transcription import get_pinyin
                    try:
                        pinyin = get_pinyin(result)
                        if pinyin and pinyin != result:
                            result += f"\n[{pinyin}]"
                    except:
                        pass
                
                return result
        except Exception as e:
            logger.error(f"Error with Google translation: {e}")

    # If Google fails, try AI services as fallback
    
    # If all translation methods fail, return original text
    logger.warning(f"All translation methods failed for {source_lang} to {target_lang}")
    return text

def verify_translation_with_claude(original_text: str, translated_text: str, source_lang: str, target_lang: str) -> str:
    """
    Use Claude to verify and potentially improve a translation.
    
    Args:
        original_text: Original text in source language
        translated_text: Translation to verify
        source_lang: Source language code
        target_lang: Target language code
        
    Returns:
        str: Verified/improved translation or original translation if verification fails
    """
    try:
        # Import Anthropic client
        import anthropic
        import os
        
        # Check if API key is available
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            logger.warning("Anthropic API key not available for translation verification")
            return translated_text
        
        # Initialize Claude client
        #the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
        client = anthropic.Anthropic(api_key=api_key)
        
        # Create verification prompt
        source_lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        target_lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        
        prompt = f"""Please review and improve this translation if needed:

Original text ({source_lang_name}): {original_text}
Translation to review ({target_lang_name}): {translated_text}

Please:
1. Check if the translation is accurate and natural
2. Correct any grammatical errors or awkward phrasing
3. Ensure cultural appropriateness and context
4. Return ONLY the final, best translation (no explanations)

If the translation is already perfect, return it unchanged."""

        # Get Claude's verification
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        verified_translation = response.content[0].text.strip()
        
        if verified_translation and len(verified_translation) > 0:
            logger.info(f"Claude successfully verified translation from {source_lang} to {target_lang}")
            return verified_translation
        else:
            logger.warning(f"Claude verification returned empty result")
            return translated_text
            
    except Exception as e:
        logger.error(f"Error in Claude verification: {e}")
        return translated_text

def translate_to_all(text: str, source_lang: str, target_languages: List[str]) -> Dict[str, Any]:
    """
    Translate text to multiple languages with transcription.
    
    Args:
        text (str): Text to translate
        source_lang (str): Source language code
        target_languages (list): List of target language codes
        
    Returns:
        dict: Dictionary with translations and metadata
    """
    # Use provided source language
    source_lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    
    # Translate to each target language
    translations = {}
    
    for lang in target_languages:
        if lang == source_lang:
            translations[lang] = text
            continue
            
        translation = translate_text(text, source_lang, lang)
        translations[lang] = translation
    
    # Add transcriptions to translations
    enhanced_translations = {}
    for lang, trans_text in translations.items():
        transcription = get_transcription(trans_text, lang)
        enhanced_translations[lang] = {
            'text': trans_text,
            'transcription': transcription
        }
    
    # Get language fact if available
    try:
        from language_facts import get_language_fact
        lang_fact = get_language_fact(source_lang)
    except ImportError:
        lang_fact = None
    
    # Return a dictionary with all translations and metadata
    return {
        'source_text': text,
        'source_lang': source_lang,
        'source_lang_name': source_lang_name,
        'translations': enhanced_translations,
        'language_fact': lang_fact
    }

# Use enhanced audio generation functions from enhanced_audio.py
# This function is kept here for backward compatibility
def generate_audio(text: str, lang_code: str) -> Optional[str]:
    """
    Generate audio for text using enhanced audio module.
    
    Args:
        text (str): Text to convert to speech
        lang_code (str): Language code
        
    Returns:
        str: Path to audio file or None if generation fails
    """
    try:
        # Use the enhanced audio module for better TTS capabilities
        from enhanced_audio import generate_audio as enhanced_generate_audio
        
        # Call the enhanced audio generation function
        audio_path = enhanced_generate_audio(text, lang_code)
        return audio_path
    except ImportError:
        logger.error("Enhanced audio module not available")
        
        # Fall back to the old method if enhanced audio is not available
        if not TTS_AVAILABLE:
            logger.error("TTS not available")
            return None
            
        try:
            # Adjust language code for gTTS if needed
            tts_lang = lang_code
            if lang_code == 'zh-CN':
                tts_lang = 'zh-cn'
            
            # Create audio file
            os.makedirs('audio', exist_ok=True)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                # Generate audio
                tts = gTTS(text=text, lang=tts_lang, slow=False)
                tts.save(temp_file.name)
                return temp_file.name
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return None

def get_flag_emoji(lang_code: str) -> str:
    """Get flag emoji for a language code."""
    flag_map = {
        'en': '🇬🇧',
        'es': '🇪🇸',
        'pt': '🇵🇹',
        'it': '🇮🇹',
        'fr': '🇫🇷',
        'ru': '🇷🇺',
        'zh-CN': '🇨🇳',
        'zh': '🇨🇳',
        'de': '🇩🇪',
        'ja': '🇯🇵',
        'ko': '🇰🇷',
        'ar': '🇸🇦',
        'hi': '🇮🇳',
        'tr': '🇹🇷',
        'nl': '🇳🇱',
        'pl': '🇵🇱',
        'sv': '🇸🇪',
        'vi': '🇻🇳',
        'th': '🇹🇭',
        'id': '🇮🇩',
        'ms': '🇲🇾',
        'he': '🇮🇱',
        'fa': '🇮🇷',
        'uk': '🇺🇦',
        'cs': '🇨🇿',
        'da': '🇩🇰',
        'fi': '🇫🇮',
        'el': '🇬🇷',
        'hu': '🇭🇺',
        'no': '🇳🇴',
        'ro': '🇷🇴'
    }
    return flag_map.get(lang_code, '🇨🇳')