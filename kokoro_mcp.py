"""
MCP-Kokoro implementation for enhanced multilingual speech recognition.
Uses the Hugging Face MCP-Kokoro model from https://huggingface.co/spaces/aiqcamp/MCP-kokoro
This model should provide excellent Italian and Spanish recognition.
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

class KokoroMCP:
    """MCP-Kokoro client for superior multilingual recognition"""
    
    def __init__(self):
        self.client = None
        self.api_url = "https://aiqcamp-mcp-kokoro.hf.space"
        self.hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        
    def initialize_client(self):
        """Initialize the Gradio client for MCP-Kokoro model"""
        try:
            self.client = Client(self.api_url)
            logger.info("Successfully connected to MCP-Kokoro model")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Kokoro client: {e}")
            return False
    
    def transcribe_audio(self, audio_path: str, language_hint: str = "auto") -> Optional[dict]:
        """
        Transcribe audio using MCP-Kokoro model with language hints.
        
        Args:
            audio_path: Path to audio file
            language_hint: Language hint (it, es, en, fr, auto, etc.)
            
        Returns:
            Dictionary with transcription results or None if failed
        """
        if not self.client and not self.initialize_client():
            logger.error("Kokoro client not available")
            return None
            
        try:
            logger.info(f"Transcribing with MCP-Kokoro, language hint: {language_hint}")
            
            # Call the MCP-Kokoro model - try different API endpoints
            possible_endpoints = ["/transcribe", "/predict", "/process_audio", "/"]
            
            for endpoint in possible_endpoints:
                try:
                    result = self.client.predict(
                        audio_path,
                        language_hint,
                        api_name=endpoint
                    )
                    
                    if result:
                        break
                        
                except Exception as endpoint_error:
                    logger.debug(f"Endpoint {endpoint} failed: {endpoint_error}")
                    continue
            
            if not result:
                # Try without language hint
                result = self.client.predict(audio_path)
            
            if result:
                # Process the result based on MCP-Kokoro response format
                transcribed_text = ""
                detected_lang = language_hint
                confidence = 0.85
                
                if isinstance(result, str):
                    transcribed_text = result.strip()
                elif isinstance(result, dict):
                    transcribed_text = result.get('text', result.get('transcription', '')).strip()
                    detected_lang = result.get('language', result.get('detected_language', language_hint))
                    confidence = result.get('confidence', 0.85)
                elif isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], str):
                        transcribed_text = result[0].strip()
                    elif isinstance(result[0], dict):
                        transcribed_text = result[0].get('text', result[0].get('transcription', '')).strip()
                        detected_lang = result[0].get('language', language_hint)
                
                if transcribed_text and len(transcribed_text) > 2:
                    # Enhance language detection for Italian and Spanish
                    if language_hint == "auto":
                        detected_lang = self.detect_language_priority(transcribed_text)
                    
                    logger.info(f"MCP-Kokoro transcribed: {transcribed_text[:100]}...")
                    logger.info(f"Detected language: {detected_lang}")
                    
                    return {
                        'text': transcribed_text,
                        'language': detected_lang,
                        'confidence': confidence,
                        'method': 'mcp_kokoro'
                    }
                else:
                    logger.warning("MCP-Kokoro returned empty transcription")
                    return None
            else:
                logger.error("MCP-Kokoro model returned no result")
                return None
                
        except Exception as e:
            logger.error(f"Error transcribing with MCP-Kokoro: {e}")
            return None
    
    def detect_language_priority(self, text: str) -> str:
        """
        Priority language detection focusing on Italian and Spanish.
        """
        if not text or len(text.strip()) < 3:
            return 'en'
        
        text_lower = text.lower()
        
        # Enhanced patterns with higher weights for Italian/Spanish
        language_scores = {}
        
        # Italian patterns (highest priority)
        italian_patterns = ['il', 'la', 'è', 'e', 'di', 'che', 'in', 'un', 'una', 'con', 'per', 'sono', 'ho', 'hai', 'della', 'degli', 'delle']
        italian_score = sum(2 for pattern in italian_patterns if pattern in text_lower)  # Double weight
        if italian_score > 0:
            language_scores['it'] = italian_score
        
        # Spanish patterns (highest priority)
        spanish_patterns = ['el', 'la', 'es', 'y', 'de', 'que', 'en', 'un', 'una', 'con', 'por', 'para', 'soy', 'tengo', 'del', 'los', 'las']
        spanish_score = sum(2 for pattern in spanish_patterns if pattern in text_lower)  # Double weight
        if spanish_score > 0:
            language_scores['es'] = spanish_score
        
        # Other languages (normal weight)
        other_patterns = {
            'fr': ['le', 'la', 'est', 'et', 'de', 'que', 'dans', 'un', 'une', 'avec', 'pour', 'du', 'les', 'des'],
            'en': ['the', 'and', 'is', 'are', 'this', 'that', 'with', 'have', 'will', 'from', 'i', 'you', 'we'],
            'de': ['der', 'die', 'das', 'und', 'ist', 'in', 'zu', 'ein', 'eine', 'mit', 'für', 'von'],
            'pt': ['o', 'a', 'é', 'e', 'de', 'que', 'em', 'um', 'uma', 'com', 'para', 'do', 'da']
        }
        
        for lang, patterns in other_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                language_scores[lang] = score
        
        # Check for special characters (extra boost for Italian/Spanish)
        if any(char in text for char in ['ò', 'ù', 'à', 'ì']):
            language_scores['it'] = language_scores.get('it', 0) + 5
        if any(char in text for char in ['ñ', 'ü', '¿', '¡']):
            language_scores['es'] = language_scores.get('es', 0) + 5
        
        # Find highest scoring language
        if language_scores:
            best_lang = max(language_scores.items(), key=lambda x: x[1])
            logger.info(f"Language scores: {language_scores}")
            logger.info(f"Selected: {best_lang[0]} (score: {best_lang[1]})")
            return best_lang[0]
        
        return 'en'  # Default fallback
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages for MCP-Kokoro model"""
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
kokoro_client = KokoroMCP()

def transcribe_with_kokoro(audio_path: str, preferred_languages: list = None) -> Optional[dict]:
    """
    Transcribe audio using MCP-Kokoro with Italian/Spanish priority.
    
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
            result = kokoro_client.transcribe_audio(audio_path, lang)
            if result and result.get('confidence', 0) > highest_confidence:
                highest_confidence = result['confidence']
                best_result = result
                
                # If we get very high confidence, stop
                if highest_confidence > 0.9:
                    break
                    
        except Exception as e:
            logger.debug(f"Language {lang} failed: {e}")
            continue
    
    return best_result

def is_kokoro_available() -> bool:
    """Check if MCP-Kokoro model is available"""
    return kokoro_client.initialize_client()