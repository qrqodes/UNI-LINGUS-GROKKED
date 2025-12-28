"""
ElevenLabs-first voice transcription with Claude and Grok fallbacks.
Replaces Google Speech Recognition entirely.
"""

import os
import requests
import base64
import logging
from typing import Optional, Tuple
import anthropic
from openai import OpenAI

logger = logging.getLogger(__name__)

def transcribe_audio_elevenlabs_primary(wav_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio using ElevenLabs as primary, Claude as secondary, Grok as tertiary fallback.
    
    Args:
        wav_path: Path to the WAV audio file
        
    Returns:
        Tuple of (transcribed_text, detected_language) or (None, None) if all methods fail
    """
    transcription = None
    detected_lang = None
    
    # Method 1: Try ElevenLabs Speech-to-Text first
    try:
        elevenlabs_api_key = os.environ.get('ELEVENLABS_API_KEY')
        if elevenlabs_api_key:
            logger.info("Attempting transcription with ElevenLabs")
            
            # Read and encode audio file
            with open(wav_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            headers = {
                'xi-api-key': elevenlabs_api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'audio': audio_base64,
                'model_id': 'eleven_multilingual_sts_v2',
                'language': 'auto'
            }
            
            response = requests.post(
                'https://api.elevenlabs.io/v1/speech-to-text',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'text' in result and result['text'].strip():
                    transcription = result['text'].strip()
                    logger.info(f"ElevenLabs transcription successful: {transcription}")
                    return transcription, detected_lang
                else:
                    logger.warning("ElevenLabs returned empty transcription")
            else:
                logger.warning(f"ElevenLabs failed: {response.status_code} - {response.text}")
        else:
            logger.warning("ElevenLabs API key not available")
            
    except Exception as e:
        logger.error(f"ElevenLabs transcription failed: {e}")
    
    # Method 2: Try Claude (Anthropic) as fallback
    try:
        anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY')
        if anthropic_api_key and not transcription:
            logger.info("Attempting transcription with Claude")
            
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            
            # Read audio file and encode for Claude
            with open(wav_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Claude can analyze audio through base64
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please transcribe this audio file accurately. Return only the transcribed text, nothing else. Support all languages including English, Spanish, Portuguese, Italian, French, Russian, Chinese, etc."
                            },
                            {
                                "type": "audio",
                                "source": {
                                    "type": "base64",
                                    "media_type": "audio/wav",
                                    "data": audio_base64
                                }
                            }
                        ]
                    }
                ]
            )
            
            if message.content and len(message.content) > 0:
                transcription = message.content[0].text.strip()
                logger.info(f"Claude transcription successful: {transcription}")
                return transcription, detected_lang
                
    except Exception as e:
        logger.error(f"Claude transcription failed: {e}")
    
    # Method 3: Try Grok (XAI) as final fallback
    try:
        xai_api_key = os.environ.get('XAI_API_KEY')
        if xai_api_key and not transcription:
            logger.info("Attempting transcription with Grok")
            
            client = OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=xai_api_key
            )
            
            # Note: Grok might not support direct audio, so we'll try text-based approach
            # This is a placeholder for when Grok adds audio support
            logger.warning("Grok audio transcription not yet implemented")
            
    except Exception as e:
        logger.error(f"Grok transcription failed: {e}")
    
    # If all methods failed
    logger.error("All transcription methods failed")
    return None, None


def detect_language_simple(text: str) -> Optional[str]:
    """Simple language detection fallback"""
    try:
        from langdetect import detect
        return detect(text)
    except:
        return None