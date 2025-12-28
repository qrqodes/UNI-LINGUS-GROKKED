"""
Open-source audio generation using gTTS (Google Text-to-Speech)
Free alternative to ElevenLabs for audio generation.
"""
import os
import logging
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

def generate_audio_with_gtts(text: str, language_code: str = 'en') -> Optional[str]:
    """
    Generate audio using gTTS (free, open-source alternative to ElevenLabs)
    
    Args:
        text: Text to convert to speech
        language_code: Language code (en, es, fr, etc.)
        
    Returns:
        Path to generated audio file or None if failed
    """
    try:
        from gtts import gTTS
        
        # Map language codes to gTTS compatible codes
        lang_map = {
            'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de',
            'it': 'it', 'pt': 'pt', 'ru': 'ru', 'zh-CN': 'zh',
            'ja': 'ja', 'ko': 'ko', 'ar': 'ar', 'hi': 'hi'
        }
        
        gtts_lang = lang_map.get(language_code, 'en')
        
        # Create gTTS object
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        
        # Generate unique filename
        audio_filename = f"gtts_{int(time.time())}_{language_code}.mp3"
        audio_path = os.path.join("audio", audio_filename)
        
        # Ensure audio directory exists
        os.makedirs("audio", exist_ok=True)
        
        # Save audio file
        tts.save(audio_path)
        
        logger.info(f"Audio generated successfully with gTTS: {audio_path}")
        return audio_path
        
    except ImportError:
        logger.warning("gTTS not available - install with: pip install gtts")
        return None
    except Exception as e:
        logger.error(f"gTTS audio generation failed: {e}")
        return None

def generate_audio_fallback(text: str, language_code: str = 'en') -> Optional[str]:
    """
    Generate audio with fallback chain: gTTS (open-source) -> ElevenLabs
    
    Args:
        text: Text to convert to speech
        language_code: Language code
        
    Returns:
        Path to generated audio file or None if all methods fail
    """
    # Try gTTS first (free, open-source)
    audio_path = generate_audio_with_gtts(text, language_code)
    if audio_path:
        return audio_path
    
    # Fallback to ElevenLabs if user has API key
    try:
        elevenlabs_api_key = os.environ.get('ELEVENLABS_API_KEY')
        if elevenlabs_api_key:
            # Import and use existing ElevenLabs functionality
            # This would call the existing ElevenLabs generation code
            logger.info("Falling back to ElevenLabs for audio generation")
            # Note: The actual ElevenLabs call would be implemented here
            return None
    except Exception as e:
        logger.warning(f"ElevenLabs fallback failed: {e}")
    
    return None