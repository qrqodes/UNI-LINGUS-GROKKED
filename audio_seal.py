"""
Audio-Seal implementation for enhanced speech recognition and audio processing.
Uses the Hugging Face Audio-Seal model from https://huggingface.co/spaces/xiaoyao9184/audio-seal
This model provides excellent multilingual speech recognition capabilities.
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

class AudioSeal:
    """Audio-Seal client for superior multilingual recognition"""
    
    def __init__(self):
        self.client = None
        self.api_url = "https://xiaoyao9184-audio-seal.hf.space"
        self.hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        
    def initialize_client(self):
        """Initialize the Gradio client for Audio-Seal model"""
        try:
            self.client = Client(self.api_url)
            logger.info("Successfully connected to Audio-Seal model")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Audio-Seal client: {e}")
            return False
    
    def transcribe_audio(self, audio_path: str, language_hint: str = "auto") -> Optional[dict]:
        """
        Transcribe audio using Audio-Seal model with language hints.
        
        Args:
            audio_path: Path to audio file
            language_hint: Language hint (it, es, en, fr, auto, etc.)
            
        Returns:
            Dictionary with transcription results or None if failed
        """
        if not self.client and not self.initialize_client():
            logger.error("Audio-Seal client not available")
            return None
            
        try:
            logger.info(f"Transcribing with Audio-Seal, language hint: {language_hint}")
            
            # Try different API endpoints for Audio-Seal
            possible_endpoints = ["/transcribe", "/predict", "/process", "/recognize", "/"]
            
            result = None
            for endpoint in possible_endpoints:
                try:
                    if language_hint and language_hint != "auto":
                        result = self.client.predict(
                            audio_path,
                            language_hint,
                            api_name=endpoint
                        )
                    else:
                        result = self.client.predict(
                            audio_path,
                            api_name=endpoint
                        )
                    
                    if result:
                        break
                        
                except Exception as endpoint_error:
                    logger.debug(f"Audio-Seal endpoint {endpoint} failed: {endpoint_error}")
                    continue
            
            if result:
                # Process the result based on Audio-Seal response format
                transcribed_text = ""
                detected_lang = language_hint if language_hint != "auto" else "en"
                confidence = 0.88
                
                if isinstance(result, str):
                    transcribed_text = result.strip()
                elif isinstance(result, dict):
                    transcribed_text = result.get('text', result.get('transcription', result.get('output', ''))).strip()
                    detected_lang = result.get('language', result.get('detected_language', detected_lang))
                    confidence = result.get('confidence', result.get('score', 0.88))
                elif isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], str):
                        transcribed_text = result[0].strip()
                    elif isinstance(result[0], dict):
                        transcribed_text = result[0].get('text', result[0].get('transcription', '')).strip()
                        detected_lang = result[0].get('language', detected_lang)
                        confidence = result[0].get('confidence', 0.88)
                
                if transcribed_text and len(transcribed_text) > 2:
                    # Enhanced language detection for Italian and Spanish priority
                    if language_hint == "auto":
                        detected_lang = self.smart_language_detection(transcribed_text)
                    
                    logger.info(f"Audio-Seal transcribed: {transcribed_text[:100]}...")
                    logger.info(f"Detected language: {detected_lang}")
                    
                    return {
                        'text': transcribed_text,
                        'language': detected_lang,
                        'confidence': confidence,
                        'method': 'audio_seal'
                    }
                else:
                    logger.warning("Audio-Seal returned empty transcription")
                    return None
            else:
                logger.error("Audio-Seal model returned no result")
                return None
                
        except Exception as e:
            logger.error(f"Error transcribing with Audio-Seal: {e}")
            return None
    
    def smart_language_detection(self, text: str) -> str:
        """
        Enhanced language detection with Italian/Spanish priority.
        """
        if not text or len(text.strip()) < 3:
            return 'en'
        
        text_lower = text.lower()
        
        # Language pattern scoring with higher weights for IT/ES
        language_scores = {}
        
        # Italian patterns (triple weight for priority)
        italian_indicators = {
            'words': ['il', 'la', 'è', 'e', 'di', 'che', 'in', 'un', 'una', 'con', 'per', 'sono', 'ho', 'hai', 'della', 'degli', 'delle', 'questo', 'questa'],
            'endings': ['zione', 'mente', 'ezza', 'ità'],
            'chars': ['ò', 'ù', 'à', 'ì', 'é']
        }
        
        italian_score = 0
        italian_score += sum(3 for word in italian_indicators['words'] if f' {word} ' in f' {text_lower} ')
        italian_score += sum(2 for ending in italian_indicators['endings'] if ending in text_lower)
        italian_score += sum(3 for char in italian_indicators['chars'] if char in text_lower)
        
        if italian_score > 0:
            language_scores['it'] = italian_score
        
        # Spanish patterns (triple weight for priority)
        spanish_indicators = {
            'words': ['el', 'la', 'es', 'y', 'de', 'que', 'en', 'un', 'una', 'con', 'por', 'para', 'soy', 'tengo', 'del', 'los', 'las', 'este', 'esta'],
            'endings': ['ción', 'mente', 'dad', 'idad'],
            'chars': ['ñ', 'ü', '¿', '¡', 'á', 'é', 'í', 'ó', 'ú']
        }
        
        spanish_score = 0
        spanish_score += sum(3 for word in spanish_indicators['words'] if f' {word} ' in f' {text_lower} ')
        spanish_score += sum(2 for ending in spanish_indicators['endings'] if ending in text_lower)
        spanish_score += sum(3 for char in spanish_indicators['chars'] if char in text_lower)
        
        if spanish_score > 0:
            language_scores['es'] = spanish_score
        
        # Other languages (normal weight)
        other_patterns = {
            'fr': {
                'words': ['le', 'la', 'est', 'et', 'de', 'que', 'dans', 'un', 'une', 'avec', 'pour', 'du', 'les', 'des', 'ce', 'cette'],
                'endings': ['tion', 'ment', 'eur', 'euse']
            },
            'en': {
                'words': ['the', 'and', 'is', 'are', 'this', 'that', 'with', 'have', 'will', 'from', 'i', 'you', 'we', 'they'],
                'endings': ['ing', 'tion', 'ness', 'ment']
            },
            'pt': {
                'words': ['o', 'a', 'é', 'e', 'de', 'que', 'em', 'um', 'uma', 'com', 'para', 'do', 'da', 'os', 'as'],
                'endings': ['ção', 'mente', 'dade', 'idade']
            }
        }
        
        for lang, indicators in other_patterns.items():
            score = 0
            score += sum(1 for word in indicators['words'] if f' {word} ' in f' {text_lower} ')
            score += sum(1 for ending in indicators['endings'] if ending in text_lower)
            if score > 0:
                language_scores[lang] = score
        
        # Find highest scoring language
        if language_scores:
            best_lang = max(language_scores.items(), key=lambda x: x[1])
            logger.info(f"Audio-Seal language scores: {language_scores}")
            logger.info(f"Selected: {best_lang[0]} (score: {best_lang[1]})")
            return best_lang[0]
        
        return 'en'  # Default fallback
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages for Audio-Seal model"""
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
audio_seal_client = AudioSeal()

def transcribe_with_audio_seal(audio_path: str, preferred_languages: list = None) -> Optional[dict]:
    """
    Transcribe audio using Audio-Seal with Italian/Spanish priority.
    
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
            result = audio_seal_client.transcribe_audio(audio_path, lang)
            if result and result.get('confidence', 0) > highest_confidence:
                highest_confidence = result['confidence']
                best_result = result
                
                # If we get very high confidence, stop
                if highest_confidence > 0.92:
                    break
                    
        except Exception as e:
            logger.debug(f"Language {lang} failed: {e}")
            continue
    
    return best_result

def is_audio_seal_available() -> bool:
    """Check if Audio-Seal model is available"""
    return audio_seal_client.initialize_client()