"""
New voice transcription system using ElevenLabs -> Claude -> Grok fallback chain.
Completely removes Google Speech Recognition.
"""

import os
import requests
import base64
import logging
from typing import Optional, Tuple
import json

logger = logging.getLogger(__name__)

def transcribe_with_elevenlabs(wav_path: str) -> Optional[str]:
    """Transcribe using ElevenLabs Speech-to-Text"""
    try:
        elevenlabs_api_key = os.environ.get('ELEVENLABS_API_KEY')
        if not elevenlabs_api_key:
            logger.warning("ElevenLabs API key not available")
            return None
            
        logger.info("Attempting transcription with ElevenLabs")
        
        headers = {
            'xi-api-key': elevenlabs_api_key
        }
        
        # Read audio file directly for form data
        with open(wav_path, 'rb') as audio_file:
            files = {
                'file': ('audio.wav', audio_file.read(), 'audio/wav')
            }
        
        data = {
            'model_id': 'scribe_v1'
        }
        
        response = requests.post(
            'https://api.elevenlabs.io/v1/speech-to-text',
            headers=headers,
            files=files,
            data=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'text' in result and result['text'].strip():
                transcription = result['text'].strip()
                logger.info(f"ElevenLabs transcription successful: {transcription}")
                return transcription
            else:
                logger.warning("ElevenLabs returned empty transcription")
        else:
            logger.warning(f"ElevenLabs failed: {response.status_code} - {response.text}")
        
        return None
        
    except Exception as e:
        logger.error(f"ElevenLabs transcription failed: {e}")
        return None

def transcribe_with_whisper_fallback(wav_path: str) -> Optional[str]:
    """Transcribe using HuggingFace Whisper as fallback"""
    try:
        hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
        if not hf_token:
            logger.warning("HuggingFace API token not available")
            return None
            
        logger.info("Attempting transcription with HuggingFace Whisper")
        
        # Use OpenAI Whisper via HuggingFace as it's reliable for transcription
        api_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        headers = {
            "Authorization": f"Bearer {hf_token}",
            "Content-Type": "audio/wav"
        }
        
        with open(wav_path, "rb") as f:
            data = f.read()
        
        response = requests.post(api_url, headers=headers, data=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if "text" in result and result["text"].strip():
                transcription = result["text"].strip()
                logger.info(f"HuggingFace Whisper transcription successful: {transcription}")
                return transcription
            else:
                logger.warning("Whisper returned empty transcription")
        else:
            logger.warning(f"HuggingFace Whisper failed: {response.status_code}")
            
        return None
        
    except Exception as e:
        logger.error(f"HuggingFace Whisper transcription failed: {e}")
        return None

def transcribe_with_grok(wav_path: str) -> Optional[str]:
    """Grok cannot process audio files directly, so this will be skipped"""
    logger.info("Grok cannot process audio files directly - skipping")
    return None

def detect_language_simple(text: str) -> str:
    """Simple language detection"""
    try:
        from langdetect import detect
        return detect(text)
    except:
        return 'en'  # Default to English

def transcribe_audio_new_system(wav_path: str) -> Tuple[Optional[str], str]:
    """
    Main transcription function using ElevenLabs -> HuggingFace Whisper chain.
    Returns (transcription, detected_language)
    """
    
    # Check if file exists
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
        logger.error(f"WAV file does not exist or is empty: {wav_path}")
        return None, 'en'
    
    # Try ElevenLabs first
    transcription = transcribe_with_elevenlabs(wav_path)
    if transcription:
        detected_lang = detect_language_simple(transcription)
        return transcription, detected_lang
    
    # Try HuggingFace Whisper as fallback
    transcription = transcribe_with_whisper_fallback(wav_path)
    if transcription:
        detected_lang = detect_language_simple(transcription)
        return transcription, detected_lang
    
    # All methods failed
    logger.error("All transcription methods failed")
    return None, 'en'