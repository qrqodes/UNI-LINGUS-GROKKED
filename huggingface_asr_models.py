"""
HuggingFace ASR Models - Only DeepSeek
"""

import os
import logging

logger = logging.getLogger(__name__)

# Check for HuggingFace API token
HUGGINGFACE_API_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")

if not HUGGINGFACE_API_TOKEN:
    logger.error("HUGGINGFACE_API_TOKEN not found in environment variables")
else:
    logger.info("Hugging Face API token configured successfully")

# DeepSeek ASR model only
ASR_MODELS = {
    "deepseek": {
        "name": "🧠 DeepSeek",
        "model_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "description": "DeepSeek open source model",
        "languages": ["multilingual"],
        "strength": "Open source DeepSeek model"
    }
}

def get_asr_models():
    """Return available ASR models"""
    return ASR_MODELS

def get_model_info(model_key):
    """Get information about a specific model"""
    return ASR_MODELS.get(model_key, None)