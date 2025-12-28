"""
Manus AI agenticSeek Integration for Enhanced Language Bot
Uses the open-source, self-hosted agenticSeek model for translation and chat.
"""
import os
import sys
import logging
import requests
import json
import time
from typing import Optional, Dict, Any

# Add agenticSeek to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agenticSeek'))

try:
    from sources.llm_provider import Provider
    from sources.interaction import Interaction
    from sources.schemas import QueryRequest, QueryResponse
    AGENTICSEEK_AVAILABLE = True
except ImportError:
    AGENTICSEEK_AVAILABLE = False

logger = logging.getLogger(__name__)

class ManusAIClient:
    """Client for interacting with the Manus AI agenticSeek model"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if the agenticSeek service is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate text using agenticSeek model"""
        if not self.available:
            return None
            
        try:
            prompt = f"Translate this text from {source_lang} to {target_lang}. Only return the translation:\n\n{text}"
            
            payload = {
                "query": prompt,
                "mode": "translation",
                "language": target_lang
            }
            
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
                
        except Exception as e:
            logger.error(f"Manus AI translation failed: {e}")
            
        return None
    
    def chat_with_ai(self, message: str) -> Optional[str]:
        """Chat with agenticSeek AI model"""
        if not self.available:
            return None
            
        try:
            payload = {
                "query": message,
                "mode": "chat",
                "language": "en"
            }
            
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
                
        except Exception as e:
            logger.error(f"Manus AI chat failed: {e}")
            
        return None
    
    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """Transcribe audio using agenticSeek speech-to-text"""
        if not self.available:
            return None
            
        try:
            with open(audio_path, 'rb') as audio_file:
                files = {'audio': audio_file}
                payload = {"mode": "transcribe"}
                
                response = requests.post(
                    f"{self.base_url}/transcribe",
                    files=files,
                    data=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("text", "").strip()
                    
        except Exception as e:
            logger.error(f"Manus AI transcription failed: {e}")
            
        return None
    
    def generate_speech(self, text: str, language_code: str = "en") -> Optional[str]:
        """Generate speech using agenticSeek text-to-speech"""
        if not self.available:
            return None
            
        try:
            payload = {
                "text": text,
                "language": language_code,
                "voice": "af_nova"  # Default voice
            }
            
            response = requests.post(
                f"{self.base_url}/tts",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                # Save audio file
                audio_filename = f"manus_tts_{int(time.time())}_{language_code}.wav"
                audio_path = os.path.join("audio", audio_filename)
                
                os.makedirs("audio", exist_ok=True)
                with open(audio_path, 'wb') as f:
                    f.write(response.content)
                    
                return audio_path
                
        except Exception as e:
            logger.error(f"Manus AI TTS failed: {e}")
            
        return None

# Initialize global client
manus_client = ManusAIClient()

def translate_with_manus_ai(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    """Primary translation function using Manus AI"""
    return manus_client.translate_text(text, source_lang, target_lang)

def chat_with_manus_ai(message: str) -> Optional[str]:
    """Primary chat function using Manus AI"""
    return manus_client.chat_with_ai(message)

def is_manus_ai_available() -> bool:
    """Check if Manus AI is available"""
    return manus_client.available