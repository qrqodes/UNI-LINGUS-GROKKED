"""
VoiceClone-TTS implementation for enhanced multilingual speech recognition.
Uses the Hugging Face VoiceClone-TTS model from https://huggingface.co/spaces/ginigen/VoiceClone-TTS
This model should provide better Italian and Spanish recognition.
"""

import logging
import os
import tempfile
import requests
import json
import uuid
from typing import Optional, Dict, Any
from gradio_client import Client

logger = logging.getLogger(__name__)

class VoiceCloneTTS:
    """VoiceClone-TTS client for better multilingual recognition"""
    
    def __init__(self):
        self.client = None
        self.api_url = "https://ginigen-voiceclone-tts.hf.space"
        self.hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        
    def initialize_client(self):
        """Initialize the Gradio client for VoiceClone-TTS model"""
        try:
            self.client = Client(self.api_url)
            logger.info("Successfully connected to VoiceClone-TTS model")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize VoiceClone client: {e}")
            return False
    
    def transcribe_audio(self, audio_path: str, target_language: str = "auto") -> Optional[dict]:
        """
        Transcribe audio using VoiceClone-TTS model with language preference.
        
        Args:
            audio_path: Path to audio file
            target_language: Preferred target language (auto, it, es, en, etc.)
            
        Returns:
            Dictionary with transcription results or None if failed
        """
        if not self.client and not self.initialize_client():
            logger.error("VoiceClone client not available")
            return None
            
        try:
            # Create unique filename
            audio_id = str(uuid.uuid4())[:8]
            
            logger.info(f"Transcribing with VoiceClone-TTS, target language: {target_language}")
            
            # Call the VoiceClone-TTS model
            result = self.client.predict(
                audio_file=audio_path,
                language=target_language,
                api_name="/transcribe"
            )
            
            if result:
                # VoiceClone-TTS typically returns text directly or in a structured format
                transcribed_text = ""
                detected_lang = target_language
                confidence = 0.8
                
                if isinstance(result, str):
                    transcribed_text = result.strip()
                elif isinstance(result, dict):
                    transcribed_text = result.get('text', '').strip()
                    detected_lang = result.get('language', target_language)
                    confidence = result.get('confidence', 0.8)
                elif isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], str):
                        transcribed_text = result[0].strip()
                    elif isinstance(result[0], dict):
                        transcribed_text = result[0].get('text', '').strip()
                        detected_lang = result[0].get('language', target_language)
                
                if transcribed_text and len(transcribed_text) > 2:
                    # Detect the actual language if auto was used
                    if target_language == "auto":
                        detected_lang = self.detect_language_from_text(transcribed_text)
                    
                    logger.info(f"VoiceClone transcribed: {transcribed_text[:100]}...")
                    logger.info(f"Detected language: {detected_lang}")
                    
                    return {
                        'text': transcribed_text,
                        'language': detected_lang,
                        'confidence': confidence,
                        'method': 'voiceclone_tts'
                    }
                else:
                    logger.warning("VoiceClone returned empty or invalid transcription")
                    return None
            else:
                logger.error("VoiceClone model returned no result")
                return None
                
        except Exception as e:
            logger.error(f"Error transcribing with VoiceClone-TTS: {e}")
            return None
    
    def detect_language_from_text(self, text: str) -> str:
        """
        Enhanced language detection prioritizing Italian and Spanish.
        """
        if not text or len(text.strip()) < 3:
            return 'en'
        
        text_lower = text.lower()
        
        # Priority patterns for Italian and Spanish
        italian_patterns = ['il', 'la', 'è', 'e', 'di', 'che', 'in', 'un', 'una', 'con', 'per', 'sono', 'ho', 'hai']
        spanish_patterns = ['el', 'la', 'es', 'y', 'de', 'que', 'en', 'un', 'una', 'con', 'por', 'para', 'soy', 'tengo']
        
        # Count matches
        italian_score = sum(1 for pattern in italian_patterns if pattern in text_lower)
        spanish_score = sum(1 for pattern in spanish_patterns if pattern in text_lower)
        
        # Check for specific characters
        if any(char in text for char in ['ò', 'ù', 'à', 'ì']):
            italian_score += 3
        if any(char in text for char in ['ñ', 'ü', '¿', '¡']):
            spanish_score += 3
        
        # Return the language with highest score
        if italian_score > spanish_score and italian_score > 0:
            return 'it'
        elif spanish_score > 0:
            return 'es'
        elif any(char in text for char in ['é', 'è', 'ê', 'ç']):
            return 'fr'
        elif any(char in text for char in ['ä', 'ö', 'ü', 'ß']):
            return 'de'
        elif any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'zh-CN'
        elif any('\u0400' <= char <= '\u04ff' for char in text):
            return 'ru'
        else:
            return 'en'
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages for VoiceClone-TTS model"""
        return {
            "auto": "Auto-detect",
            "it": "Italian",
            "es": "Spanish", 
            "en": "English",
            "fr": "French",
            "de": "German",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic"
        }

# Global instance
voiceclone_client = VoiceCloneTTS()

def transcribe_with_voiceclone(audio_path: str, preferred_languages: list = None) -> Optional[dict]:
    """
    Transcribe audio using VoiceClone-TTS with language preference.
    
    Args:
        audio_path: Path to audio file
        preferred_languages: List of preferred languages to try first
        
    Returns:
        Dictionary with transcription results or None if failed
    """
    if preferred_languages is None:
        preferred_languages = ["it", "es", "auto"]  # Prioritize Italian and Spanish
    
    best_result = None
    highest_confidence = 0.0
    
    # Try each preferred language
    for lang in preferred_languages:
        try:
            result = voiceclone_client.transcribe_audio(audio_path, lang)
            if result and result.get('confidence', 0) > highest_confidence:
                highest_confidence = result['confidence']
                best_result = result
                
                # If we get high confidence, we can stop
                if highest_confidence > 0.85:
                    break
                    
        except Exception as e:
            logger.debug(f"Language {lang} failed: {e}")
            continue
    
    return best_result

def is_voiceclone_available() -> bool:
    """Check if VoiceClone-TTS model is available"""
    return voiceclone_client.initialize_client()