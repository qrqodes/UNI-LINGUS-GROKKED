"""
Gradio Models - Only DeepSeek
"""

import os
import logging

logger = logging.getLogger(__name__)

# DeepSeek model only
GRADIO_MODELS = {
    "deepseek": {
        "name": "🧠 DeepSeek",
        "space": "deepseek-ai/deepseek-coder-1.3b-instruct",
        "description": "DeepSeek open source model",
        "strength": "Open source DeepSeek model"
    }
}

def get_gradio_models():
    """Return available Gradio models"""
    return GRADIO_MODELS

def get_model_info(model_key):
    """Get information about a specific model"""
    return GRADIO_MODELS.get(model_key, None)