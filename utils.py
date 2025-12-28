"""
Utility functions for the translation bot.
This module includes functions for audio generation, caching, and error handling.
"""
import os
import time
import json
import tempfile
import logging
import hashlib
from pathlib import Path
from functools import wraps
from gtts import gTTS
from typing import Dict, List, Any, Optional, Callable, Union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a directory for cache files
CACHE_DIR = Path('cache')
AUDIO_CACHE_DIR = CACHE_DIR / 'audio'
TRANSLATION_CACHE_DIR = CACHE_DIR / 'translations'

def ensure_cache_dirs():
    """Ensure that cache directories exist."""
    for directory in [CACHE_DIR, AUDIO_CACHE_DIR, TRANSLATION_CACHE_DIR]:
        directory.mkdir(exist_ok=True, parents=True)


def cache_key(text: str, lang_code: str) -> str:
    """Generate a cache key for text and language."""
    # Use a hash to create a filename-safe key
    key = hashlib.md5(f"{text}_{lang_code}".encode()).hexdigest()
    return key


def cache_audio(func):
    """Decorator to cache audio files."""
    @wraps(func)
    def wrapper(text: str, lang_code: str, *args, **kwargs):
        ensure_cache_dirs()
        key = cache_key(text, lang_code)
        cache_path = AUDIO_CACHE_DIR / f"{key}.mp3"
        
        # If cached file exists and is less than a day old, use it
        if cache_path.exists() and (time.time() - cache_path.stat().st_mtime < 86400):
            logger.info(f"Using cached audio for {lang_code}: {text[:20]}...")
            return str(cache_path)
        
        # Otherwise generate new audio file
        audio_path = func(text, lang_code, *args, **kwargs)
        
        # Copy to cache
        if os.path.exists(audio_path) and audio_path != str(cache_path):
            with open(audio_path, 'rb') as src:
                with open(cache_path, 'wb') as dest:
                    dest.write(src.read())
            logger.info(f"Cached audio for {lang_code}: {text[:20]}...")
        
        return audio_path
    
    return wrapper


@cache_audio
def generate_audio(text: str, lang_code: str) -> str:
    """
    Generate audio for the text in the specified language.
    
    Args:
        text: The text to convert to speech
        lang_code: The language code for the text
    
    Returns:
        str: Path to the audio file
    """
    # Handle language code differences between gTTS and our app
    tts_lang = lang_code.split('-')[0] if lang_code != 'en' else 'en'
    
    # Create a temporary filename to prevent conflicts
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        temp_path = temp_file.name
        
    try:
        # Generate speech
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        tts.save(temp_path)
        logger.info(f"Generated audio for {lang_code}: {text[:20]}...")
        return temp_path
    except Exception as e:
        logger.error(f"Audio generation failed for {lang_code}: {str(e)}")
        # If generation fails, return None
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def cache_translation(func):
    """Decorator to cache translations."""
    @wraps(func)
    def wrapper(text: str, source_lang: str, target_lang: str, *args, **kwargs):
        ensure_cache_dirs()
        key = cache_key(f"{text}_{source_lang}_{target_lang}", "translation")
        cache_path = TRANSLATION_CACHE_DIR / f"{key}.json"
        
        # If cached translation exists and is less than a week old, use it
        if cache_path.exists() and (time.time() - cache_path.stat().st_mtime < 604800):  # 7 days
            logger.info(f"Using cached translation for {source_lang}->{target_lang}: {text[:20]}...")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Otherwise perform new translation
        translation = func(text, source_lang, target_lang, *args, **kwargs)
        
        # Cache the result
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(translation, f, ensure_ascii=False)
        logger.info(f"Cached translation for {source_lang}->{target_lang}: {text[:20]}...")
        
        return translation
    
    return wrapper


def clean_old_cache_files(max_age_days: int = 7):
    """Remove cache files older than specified days."""
    ensure_cache_dirs()
    now = time.time()
    max_age_seconds = max_age_days * 86400
    
    for cache_dir in [AUDIO_CACHE_DIR, TRANSLATION_CACHE_DIR]:
        for file_path in cache_dir.glob('*'):
            if file_path.is_file() and (now - file_path.stat().st_mtime > max_age_seconds):
                file_path.unlink()
                logger.info(f"Removed old cache file: {file_path}")


def create_temp_audio_dir() -> str:
    """Create a temporary directory for audio files that will be automatically cleaned up."""
    temp_dir = tempfile.mkdtemp(prefix="audio_")
    return temp_dir


def cleanup_temp_files(directory: str):
    """Clean up temporary files in the specified directory."""
    if not os.path.exists(directory):
        return
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {str(e)}")
    
    try:
        os.rmdir(directory)
    except Exception as e:
        logger.error(f"Error removing directory {directory}: {str(e)}")


def format_progress_bar(percentage: float) -> str:
    """
    Format a progress bar string for display.
    
    Args:
        percentage: Percentage complete (0-100)
        
    Returns:
        str: Formatted progress bar
    """
    bar_length = 20
    filled_length = int(bar_length * percentage / 100)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    return f"[{bar}] {percentage:.1f}%"


def is_valid_language_code(code: str) -> bool:
    """Check if a language code is supported by the application."""
    valid_codes = ['en', 'es', 'fr', 'it', 'pt', 'ru', 'zh-CN']
    return code in valid_codes


def safe_filename(text: str) -> str:
    """Convert text to a safe filename."""
    # Replace any non-alphanumeric characters with underscores
    return "".join(c if c.isalnum() else '_' for c in text).strip('_')


# Error handling functions
def handle_api_error(e: Exception, feature_name: str) -> str:
    """Format an error message for API failures."""
    logger.error(f"{feature_name} API error: {str(e)}")
    return f"Sorry, there was an error with the {feature_name} service. Please try again later."


# Performance monitoring
def measure_execution_time(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"Function {func.__name__} took {execution_time:.2f} seconds to execute.")
        return result
    return wrapper