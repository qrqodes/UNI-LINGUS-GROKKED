"""
Audio generation module for enhanced language bot.
This is a simplified version that provides compatibility with the enhanced_bot.py
and has been reorganized from older versions.
"""
import os
import time
import logging
import tempfile
from typing import Optional, List
from enhanced_audio import generate_audio

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

def generate_audio_for_ai_message(text: str, lang_code: str = 'en') -> Optional[str]:
    """
    Generate audio for AI-generated text.
    This is a compatibility wrapper around enhanced_audio.generate_audio.
    
    Args:
        text (str): Text to convert to speech
        lang_code (str): Language code
        
    Returns:
        str: Path to audio file or None if generation fails
    """
    try:
        return generate_audio(text, lang_code)
    except Exception as e:
        logger.error(f"Error generating audio for AI message: {e}")
        return None

def clean_old_audio_files(max_age_hours: int = 24) -> None:
    """
    Remove audio files older than the specified time.
    
    Args:
        max_age_hours (int): Maximum age in hours
    """
    try:
        # Ensure audio directory exists
        audio_dir = "audio"
        if not os.path.exists(audio_dir):
            return
            
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        # Check all files in the audio directory
        for filename in os.listdir(audio_dir):
            if not filename.endswith('.mp3') and not filename.endswith('.ogg'):
                continue
                
            file_path = os.path.join(audio_dir, filename)
            
            # Get file's last modification time
            file_mtime = os.path.getmtime(file_path)
            
            # If file is older than the maximum age, delete it
            if (current_time - file_mtime) > max_age_seconds:
                try:
                    os.remove(file_path)
                    logger.info(f"Removed old audio file: {file_path}")
                except Exception as e:
                    logger.error(f"Error removing old audio file {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error cleaning old audio files: {e}")