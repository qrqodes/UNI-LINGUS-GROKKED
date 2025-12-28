"""
HuggingFace API ASR - Only DeepSeek
"""

import os
import logging

logger = logging.getLogger(__name__)

# DeepSeek model only
ASR_MODELS = {
    "deepseek": {
        "name": "🧠 DeepSeek",
        "model_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "description": "DeepSeek open source model",
        "strength": "Open source DeepSeek model"
    }
}

def get_available_models():
    """Return available ASR models"""
    return ASR_MODELS

def transcribe_audio(audio_path, model_key="deepseek"):
    """Transcribe audio using DeepSeek model"""
    try:
        # Implementation would go here
        return {"text": "Transcription with DeepSeek", "model": "DeepSeek"}
    except Exception as e:
        logger.error(f"DeepSeek transcription error: {e}")
        return None