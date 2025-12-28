"""
AgenticSeek transcription module as fallback for voice recognition.
Based on: https://github.com/Fosowl/agenticSeek
"""

import os
import logging
import requests
import json
from typing import Optional
import base64

logger = logging.getLogger(__name__)


def transcribe_with_agenticseek(audio_path: str) -> Optional[str]:
    """
    Transcribe audio using AgenticSeek model as fallback.
    """
    try:
        # Check if file exists and has content
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            logger.error(f"Audio file does not exist or is empty: {audio_path}")
            return None
        
        # Read and encode audio file
        with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # Use OpenAI Whisper API as AgenticSeek backend
        openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not openai_api_key:
            logger.error("OpenAI API key not found for AgenticSeek fallback")
            return None
        
        url = "https://api.openai.com/v1/audio/transcriptions"
        
        with open(audio_path, 'rb') as audio_file:
            files = {
                'file': audio_file,
                'model': (None, 'whisper-1'),
                'language': (None, 'auto')
            }
            headers = {
                'Authorization': f'Bearer {openai_api_key}'
            }
            
            logger.info(f"Attempting AgenticSeek transcription for: {audio_path}")
            response = requests.post(url, files=files, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                if text:
                    logger.info(f"AgenticSeek transcription successful: {text[:50]}...")
                    return text
                else:
                    logger.warning("AgenticSeek returned empty transcription")
                    return None
            else:
                logger.error(f"AgenticSeek transcription failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"Error in AgenticSeek transcription: {e}")
        return None


def transcribe_with_huggingface_fallback(audio_path: str) -> Optional[str]:
    """
    Additional fallback using HuggingFace ASR models.
    """
    try:
        hf_token = os.environ.get('HUGGINGFACE_API_TOKEN')
        if not hf_token:
            logger.error("HuggingFace API token not found")
            return None
        
        url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "application/octet-stream"
        }
        
        with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        logger.info(f"Attempting HuggingFace transcription for: {audio_path}")
        response = requests.post(url, headers=headers, data=audio_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('text', '').strip()
            if text:
                logger.info(f"HuggingFace transcription successful: {text[:50]}...")
                return text
            else:
                logger.warning("HuggingFace returned empty transcription")
                return None
        else:
            logger.error(f"HuggingFace transcription failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error in HuggingFace transcription: {e}")
        return None


def transcribe_with_multiple_fallbacks(audio_path: str) -> Optional[str]:
    """
    Try multiple transcription services with fallbacks.
    """
    # First try ElevenLabs (if available)
    try:
        from elevenlabs_voice_handler import transcribe_with_elevenlabs
        result = transcribe_with_elevenlabs(audio_path)
        if result:
            logger.info("Transcription successful with ElevenLabs")
            return result
    except Exception as e:
        logger.warning(f"ElevenLabs transcription failed: {e}")
    
    # Fallback to AgenticSeek (OpenAI Whisper)
    result = transcribe_with_agenticseek(audio_path)
    if result:
        logger.info("Transcription successful with AgenticSeek")
        return result
    
    # Final fallback to HuggingFace
    result = transcribe_with_huggingface_fallback(audio_path)
    if result:
        logger.info("Transcription successful with HuggingFace")
        return result
    
    logger.error("All transcription methods failed")
    return None