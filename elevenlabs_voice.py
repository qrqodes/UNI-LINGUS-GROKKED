"""
ElevenLabs AI Voice Services - Complete voice functionality using only ElevenLabs.
Handles both text-to-speech and speech-to-text using ElevenLabs API.
"""

import os
import requests
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class ElevenLabsVoice:
    def __init__(self):
        self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
        
    def is_available(self) -> bool:
        """Check if ElevenLabs API is available."""
        return bool(self.api_key)
    
    def text_to_speech(self, text: str, language: str = "en") -> Optional[str]:
        """
        Convert text to speech using ElevenLabs AI.
        
        Args:
            text: Text to convert to speech
            language: Language code (optional)
            
        Returns:
            Path to generated audio file or None if failed
        """
        if not self.is_available():
            logger.error("ElevenLabs API key not available")
            return None
            
        try:
            # Use different voice IDs for different languages
            voice_id = self._get_voice_for_language(language)
            
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "style": 0.5,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Save audio file
                import time
                filename = f"elevenlabs_tts_{int(time.time())}_{language}.mp3"
                filepath = os.path.join('audio', filename)
                
                os.makedirs('audio', exist_ok=True)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"ElevenLabs TTS generated: {filepath}")
                return filepath
            else:
                logger.error(f"ElevenLabs TTS failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            return None
    
    def speech_to_text(self, audio_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Convert speech to text using ElevenLabs AI.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary with transcription results or None if failed
        """
        if not self.is_available():
            logger.error("ElevenLabs API key not available")
            return None
            
        try:
            url = f"{self.base_url}/speech-to-text"
            
            headers = {
                "xi-api-key": self.api_key
            }
            
            with open(audio_file_path, 'rb') as f:
                files = {"audio": f}
                response = requests.post(url, headers=headers, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                
                if text:
                    logger.info(f"ElevenLabs STT successful: {text[:50]}...")
                    return {
                        'text': text,
                        'confidence': 0.9,  # ElevenLabs doesn't provide confidence scores
                        'model': 'ElevenLabs AI'
                    }
                else:
                    logger.warning("ElevenLabs STT returned empty text")
                    return None
            else:
                logger.error(f"ElevenLabs STT failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"ElevenLabs STT error: {e}")
            return None
    
    def _get_voice_for_language(self, language: str) -> str:
        """
        Get appropriate voice ID for the given language.
        
        Args:
            language: Language code
            
        Returns:
            Voice ID string
        """
        # Voice mapping for different languages
        voice_mapping = {
            'en': '21m00Tcm4TlvDq8ikWAM',  # Rachel - English
            'es': 'ThT5KcBeYPX3keUQqHPh',  # Dorothy - Spanish
            'fr': 'XB0fDUnXU5powFXDhCwa',  # Charlotte - French
            'de': 'ErXwobaYiN019PkySvjV',  # Antoni - German
            'it': 'MF3mGyEYCl7XYWbV9V6O',  # Elli - Italian
            'pt': 'TxGEqnHWrfWFTfGW9XjX',  # Josh - Portuguese
            'ru': 'VR6AewLTigWG4xSOukaG',  # Arnold - Russian
            'zh': 'oWAxZDx7w5VEj9dCyTzz',  # Grace - Chinese
            'ja': 'CYw3kZ02Hs0563khs1Fj',  # Gigi - Japanese
            'ko': 'XrExE9yKIg1WjnnlVkGX'   # Liam - Korean
        }
        
        return voice_mapping.get(language, voice_mapping['en'])  # Default to English

# Global instance
elevenlabs_voice = ElevenLabsVoice()

def generate_elevenlabs_audio(text: str, language_code: str = "en") -> Optional[str]:
    """
    Generate audio using ElevenLabs TTS - wrapper function for compatibility.
    
    Args:
        text: Text to convert to speech
        language_code: Language code (ISO format)
        
    Returns:
        Path to generated audio file or None if failed
    """
    return elevenlabs_voice.text_to_speech(text, language_code)