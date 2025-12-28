"""
Enhanced voice recognition module for improved Portuguese, Russian, and Chinese support.
This module provides AI-powered fallback transcription when Google Speech Recognition fails.
"""

import os
import logging
import tempfile
import json
from typing import Optional, List, Dict, Any
import speech_recognition as sr
from pydub import AudioSegment

# Configure logging
logger = logging.getLogger(__name__)

# Check for AI services
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    openai_client = None
    if os.environ.get("OPENAI_API_KEY"):
        try:
            openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            logger.info("OpenAI client initialized for enhanced voice recognition")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            OPENAI_AVAILABLE = False
except ImportError:
    OPENAI_AVAILABLE = False
    openai_client = None

try:
    from xai import is_available as is_grok_available, chat_with_grok
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False
    def is_grok_available():
        return False
    def chat_with_grok(*args, **kwargs):
        return None

def transcribe_with_openai_whisper(audio_path: str, language: Optional[str] = None) -> Optional[str]:
    """
    Transcribe audio using OpenAI Whisper for better multilingual support.
    
    Args:
        audio_path: Path to audio file
        language: Language code to hint Whisper (optional)
        
    Returns:
        Transcribed text or None if transcription fails
    """
    if not openai_client:
        return None
        
    try:
        # Convert to supported format if needed
        audio = AudioSegment.from_file(audio_path)
        
        # Create temporary MP3 file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            audio.export(temp_file.name, format="mp3")
            
            # Transcribe with Whisper
            with open(temp_file.name, 'rb') as audio_file:
                transcript = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language if language else None,
                    response_format="text"
                )
                
            # Clean up temp file
            os.unlink(temp_file.name)
            
            if transcript and transcript.strip():
                logger.info(f"OpenAI Whisper transcription successful: {transcript[:50]}...")
                return transcript.strip()
                
    except Exception as e:
        logger.error(f"Error with OpenAI Whisper transcription: {e}")
        
    return None

def transcribe_with_grok_ai(audio_path: str, language: Optional[str] = None) -> Optional[str]:
    """
    Use Grok AI to help with transcription by analyzing audio characteristics.
    
    Args:
        audio_path: Path to audio file
        language: Language code hint
        
    Returns:
        Transcribed text or None if transcription fails
    """
    if not GROK_AVAILABLE or not is_grok_available():
        return None
        
    try:
        # Get basic transcription attempt first
        basic_transcription = basic_speech_recognition(audio_path, language)
        
        if basic_transcription:
            # Use Grok to improve/correct the transcription
            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert linguist. The following text was transcribed from audio in {language or 'unknown'} language but may contain errors. Please correct any obvious transcription errors and return only the corrected text, no explanations."
                },
                {
                    "role": "user",
                    "content": f"Correct this transcription: {basic_transcription}"
                }
            ]
            
            corrected_text = chat_with_grok(messages, temperature=0.3)
            if corrected_text and corrected_text.strip():
                logger.info(f"Grok AI transcription correction successful")
                return corrected_text.strip()
        
    except Exception as e:
        logger.error(f"Error with Grok AI transcription assistance: {e}")
        
    return None

def basic_speech_recognition(audio_path: str, language: Optional[str] = None) -> Optional[str]:
    """
    Basic speech recognition using Google Speech Recognition with enhanced language support.
    
    Args:
        audio_path: Path to audio file
        language: Language code
        
    Returns:
        Transcribed text or None if transcription fails
    """
    try:
        # Convert to WAV format
        wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        audio = AudioSegment.from_file(audio_path)
        audio.export(wav_path, format="wav")
        
        # Enhanced language mapping for better recognition
        language_variants = {
            'pt': ['pt-PT', 'pt-BR', 'pt'],
            'ru': ['ru-RU', 'ru'],
            'zh-CN': ['zh-CN', 'zh', 'zh-TW'],
            'zh': ['zh-CN', 'zh', 'zh-TW'],
            'es': ['es-ES', 'es-MX', 'es-US', 'es'],
            'fr': ['fr-FR', 'fr-CA', 'fr'],
            'it': ['it-IT', 'it'],
            'de': ['de-DE', 'de'],
            'ja': ['ja-JP', 'ja'],
            'ko': ['ko-KR', 'ko'],
            'ar': ['ar-SA', 'ar-EG', 'ar'],
            'hi': ['hi-IN', 'hi'],
            'en': ['en-US', 'en-GB', 'en-AU', 'en']
        }
        
        # Try speech recognition with multiple language variants
        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = r.record(source)
            
            # Get language variants to try
            langs_to_try = language_variants.get(language, ['en-US']) if language else ['en-US']
            
            # Try each language variant
            for lang_code in langs_to_try:
                try:
                    logger.info(f"Trying recognition with language: {lang_code}")
                    text = r.recognize_google(audio_data, language=lang_code)
                    if text and text.strip():
                        logger.info(f"Google Speech Recognition successful with {lang_code}: {text}")
                        return text.strip()
                except sr.UnknownValueError:
                    logger.debug(f"No speech recognized with {lang_code}")
                    continue
                except sr.RequestError as e:
                    logger.error(f"Google Speech Recognition error with {lang_code}: {e}")
                    continue
        
        # Clean up temp file
        os.unlink(wav_path)
        
    except Exception as e:
        logger.error(f"Error in basic speech recognition: {e}")
        
    return None

def enhanced_voice_transcription(audio_path: str, language: Optional[str] = None) -> Optional[str]:
    """
    Enhanced voice transcription with multiple fallback methods for better multilingual support.
    
    Args:
        audio_path: Path to audio file
        language: Language code hint (optional)
        
    Returns:
        Transcribed text or None if all methods fail
    """
    logger.info(f"Starting enhanced voice transcription for language: {language}")
    
    # Method 1: Try OpenAI Whisper first (best for multilingual)
    if openai_client:
        logger.info("Trying OpenAI Whisper transcription...")
        whisper_result = transcribe_with_openai_whisper(audio_path, language)
        if whisper_result:
            return whisper_result
    
    # Method 2: Try enhanced Google Speech Recognition
    logger.info("Trying enhanced Google Speech Recognition...")
    google_result = basic_speech_recognition(audio_path, language)
    if google_result:
        return google_result
    
    # Method 3: Try Grok AI assistance (if available)
    if GROK_AVAILABLE and is_grok_available():
        logger.info("Trying Grok AI transcription assistance...")
        grok_result = transcribe_with_grok_ai(audio_path, language)
        if grok_result:
            return grok_result
    
    logger.warning(f"All transcription methods failed for language: {language}")
    return None

def get_supported_languages() -> List[str]:
    """
    Get list of languages with enhanced support.
    
    Returns:
        List of language codes with enhanced voice recognition support
    """
    enhanced_languages = [
        'en',    # English - Full support
        'es',    # Spanish - Full support  
        'pt',    # Portuguese - Enhanced support
        'it',    # Italian - Full support
        'fr',    # French - Enhanced support
        'ru',    # Russian - Enhanced support
        'zh-CN', # Chinese - Enhanced support
        'de',    # German - Enhanced support
        'ja',    # Japanese - Enhanced support
        'ko',    # Korean - Enhanced support
        'ar',    # Arabic - Enhanced support
        'hi'     # Hindi - Enhanced support
    ]
    
    return enhanced_languages

def test_enhanced_transcription(test_file: str, language: str = 'en') -> None:
    """
    Test the enhanced transcription system.
    
    Args:
        test_file: Path to test audio file
        language: Language to test
    """
    logger.info(f"Testing enhanced transcription for {language}")
    result = enhanced_voice_transcription(test_file, language)
    
    if result:
        logger.info(f"Transcription successful: {result}")
    else:
        logger.warning(f"Transcription failed for {language}")