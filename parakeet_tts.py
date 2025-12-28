"""
NVIDIA Parakeet TDT 0.6B v2 Text-to-Speech implementation.
Uses the official Hugging Face model from https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v2
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

class ParakeetTTS:
    """NVIDIA Parakeet TDT 0.6B v2 Text-to-Speech client"""
    
    def __init__(self):
        self.client = None
        self.api_url = "https://nvidia-parakeet-tdt-0-6b-v2.hf.space"
        self.hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        
    def initialize_client(self):
        """Initialize the Gradio client for Parakeet model"""
        try:
            self.client = Client(self.api_url)
            logger.info("Successfully connected to NVIDIA Parakeet TDT 0.6B v2 model")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Parakeet client: {e}")
            return False
    
    def generate_speech(self, text: str, language: str = "en", speaker_id: int = 0) -> Optional[str]:
        """
        Generate speech using NVIDIA Parakeet TDT 0.6B v2 model.
        
        Args:
            text: Text to convert to speech
            language: Language code (en, es, fr, de, etc.)
            speaker_id: Speaker voice ID (0-9 for different voices)
            
        Returns:
            Path to generated audio file or None if failed
        """
        if not self.client and not self.initialize_client():
            logger.error("Parakeet client not available")
            return None
            
        try:
            # Create unique filename
            audio_id = str(uuid.uuid4())[:8]
            output_path = f"audio/parakeet_{audio_id}.wav"
            os.makedirs("audio", exist_ok=True)
            
            # Call the Parakeet model
            logger.info(f"Generating speech with Parakeet TDT for: {text[:50]}...")
            
            result = self.client.predict(
                text=text,
                language=language,
                speaker=speaker_id,
                api_name="/synthesize"
            )
            
            # The result should be an audio file path
            if result and os.path.exists(result):
                # Copy to our audio directory
                import shutil
                shutil.copy2(result, output_path)
                logger.info(f"Successfully generated speech with Parakeet: {output_path}")
                return output_path
            else:
                logger.error("Parakeet model returned invalid result")
                return None
                
        except Exception as e:
            logger.error(f"Error generating speech with Parakeet: {e}")
            return None
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get supported languages for Parakeet TDT model"""
        return {
            "en": "English",
            "es": "Spanish", 
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi"
        }
    
    def get_voice_info(self) -> Dict[int, str]:
        """Get available voice/speaker information"""
        return {
            0: "Default Female Voice",
            1: "Male Voice 1",
            2: "Female Voice 2", 
            3: "Male Voice 2",
            4: "Neutral Voice 1",
            5: "Neutral Voice 2",
            6: "Young Female",
            7: "Mature Male",
            8: "Cheerful Female",
            9: "Deep Male"
        }

# Global instance
parakeet_client = ParakeetTTS()

def generate_parakeet_audio(text: str, language_code: str = "en", voice_id: int = 0) -> Optional[str]:
    """
    Generate audio using NVIDIA Parakeet TDT 0.6B v2 model.
    
    Args:
        text: Text to convert to speech
        language_code: Language code
        voice_id: Voice/speaker ID (0-9)
        
    Returns:
        Path to audio file or None if failed
    """
    # Map language codes to supported languages
    lang_mapping = {
        "en": "en",
        "es": "es", 
        "fr": "fr",
        "de": "de",
        "it": "it",
        "pt": "pt",
        "ru": "ru",
        "zh": "zh",
        "zh-CN": "zh",
        "ja": "ja",
        "ko": "ko",
        "ar": "ar",
        "hi": "hi"
    }
    
    # Get the proper language code
    lang = lang_mapping.get(language_code.split('-')[0], "en")
    
    # Generate speech
    return parakeet_client.generate_speech(text, lang, voice_id)

def is_parakeet_available() -> bool:
    """Check if Parakeet TDT model is available"""
    return parakeet_client.initialize_client()

def get_parakeet_capabilities() -> Dict[str, Any]:
    """Get Parakeet model capabilities"""
    return {
        "model_name": "NVIDIA Parakeet TDT 0.6B v2",
        "supported_languages": parakeet_client.get_supported_languages(),
        "available_voices": parakeet_client.get_voice_info(),
        "max_text_length": 500,  # Recommended max length
        "output_format": "WAV",
        "sample_rate": 22050
    }