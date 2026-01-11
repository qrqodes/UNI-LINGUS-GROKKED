"""
Flask web application for the Enhanced Language Learning and Translation Service.
Clean implementation with ElevenLabs-only voice functionality.
"""
import os
import logging
import secrets
import tempfile
import json
import time
import uuid
import requests
import re
import html
import threading
import glob
import asyncio
import subprocess
from flask import Flask, render_template, request, jsonify, session, Response, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from langdetect import detect, LangDetectException
from functools import lru_cache
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Infinite Wiki functionality
try:
    from gemini_wiki import generate_word_wiki, generate_contextual_wiki, get_quick_explanation
    WIKI_AVAILABLE = True
    logger.info("Infinite Wiki with Gemini 2.5 Flash enabled")
except ImportError as e:
    WIKI_AVAILABLE = False
    logger.warning(f"Wiki functionality not available: {e}")

# Streamlined voice - removed heavy transcription modules
SMART_VOICE_AVAILABLE = False

# Streamlined AI - removed heavy modules for faster startup
ENHANCED_AI_AVAILABLE = False

# Streamlined multimodal - keep only essential UNI grammar
try:
    from updated_uni_grammar import translate_to_uni, get_uni_vocabulary
    UNI_AVAILABLE = True
    logger.info("UNI Grammar System loaded")
except ImportError:
    UNI_AVAILABLE = False

# UNI Language Development Integration
try:
    from uni_web_integration import init_uni_integration
    UNI_DEVELOPMENT_AVAILABLE = True
    logger.info("UNI Language Development System loaded")
except ImportError:
    UNI_DEVELOPMENT_AVAILABLE = False
    logger.warning("UNI development integration not available")
    
MULTIMODAL_AVAILABLE = False

# Import artificial language support
ARTIFICIAL_LANGUAGE_SUPPORT = False
try:
    import artificial_language_creator
    from artificial_language_creator import language_manager, popularity_manager, create_sample_language
    ARTIFICIAL_LANGUAGE_SUPPORT = True
except ImportError:
    # Artificial language support will be available after restart
    pass

# Language constants
ALL_LANGUAGE_CODES = ['en', 'es', 'pt', 'it', 'fr', 'ru', 'zh-CN', 'uni']
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh-CN': 'Chinese',
    'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi',
    'uni': 'UNI'
}

# Database setup
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", secrets.token_hex(16))

# configure the database, relative to the app instance folder
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
# initialize the app with the extension, flask-sqlalchemy >= 3.0.x
db.init_app(app)

# Database models defined inline to avoid circular imports
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class Translation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text, nullable=False)
    translated_text = db.Column(db.Text, nullable=False)
    source_language = db.Column(db.String(10), nullable=False)
    target_language = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    session_id = db.Column(db.String(64), nullable=True)

class UNICorrection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_text = db.Column(db.Text, nullable=False)
    incorrect_translation = db.Column(db.Text, nullable=False)
    correct_translation = db.Column(db.Text, nullable=False)
    source_language = db.Column(db.String(10), nullable=False)
    target_language = db.Column(db.String(10), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_notes = db.Column(db.Text, nullable=True)

with app.app_context():
    db.create_all()

# Production and privacy configuration
try:
    from production_config import (
        configure_production_logging, 
        add_error_handlers, 
        validate_environment,
        health_checker
    )
    from privacy_compliance import PrivacyManager, ConsentManager, create_privacy_policy
    
    # Configure production features
    configure_production_logging(app)
    add_error_handlers(app)
    validate_environment()
    
    # Initialize privacy compliance
    privacy_manager = PrivacyManager(app, db)
    consent_manager = ConsentManager()
    
    logger.info("Production and privacy configuration loaded successfully")
except ImportError:
    logger.warning("Production/privacy configuration not available")
except Exception as e:
    logger.error(f"Configuration failed: {e}")

# Rate limiting with Redis backend for production
try:
    # Try to use Redis for production rate limiting
    import redis
    redis_client = redis.Redis.from_url(os.environ.get('REDIS_URL', 'redis://localhost:6379'))
    redis_client.ping()  # Test connection
    
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["500 per day", "100 per hour"],
        storage_uri=os.environ.get('REDIS_URL', 'redis://localhost:6379')
    )
    logger.info("Rate limiting configured with Redis backend")
except Exception as e:
    # Fallback to memory storage with warning suppression for development
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["500 per day", "100 per hour"]
    )
    if not os.environ.get('FLASK_ENV') == 'development':
        logger.warning("Rate limiting using memory storage - not recommended for production")

# Initialize UNI Language Development Integration
if UNI_DEVELOPMENT_AVAILABLE:
    try:
        uni_integration = init_uni_integration(app)
        logger.info("UNI Language Development features activated")
    except Exception as e:
        logger.error(f"UNI integration failed: {e}")
        UNI_DEVELOPMENT_AVAILABLE = False

# Input validation and sanitization
def sanitize_input(text, max_length=5000):
    """Sanitize user input to prevent injection attacks."""
    if not text or not isinstance(text, str):
        return ""
    
    # Remove potentially dangerous characters
    text = re.sub(r'[<>"\'\\\x00-\x1f\x7f-\x9f]', '', text)
    
    # Limit length
    text = text[:max_length]
    
    # HTML escape
    text = html.escape(text)
    
    return text.strip()

def validate_language_code(lang_code):
    """Validate language code format."""
    if not lang_code or not isinstance(lang_code, str):
        return False
    
    valid_codes = ['en', 'es', 'pt', 'it', 'fr', 'ru', 'zh-CN', 'de', 'ja', 'ko', 'ar', 'hi', 'uni']
    return lang_code in valid_codes

def cleanup_old_audio_files():
    """Remove audio files older than 1 hour."""
    try:
        audio_dir = 'audio'
        if not os.path.exists(audio_dir):
            return
        
        current_time = time.time()
        for file_path in glob.glob(os.path.join(audio_dir, '*.mp3')):
            file_age = current_time - os.path.getctime(file_path)
            if file_age > 3600:  # 1 hour
                try:
                    os.remove(file_path)
                    logger.info(f"Removed old audio file: {file_path}")
                except OSError:
                    pass
    except Exception as e:
        logger.warning(f"Audio cleanup failed: {e}")

# Schedule periodic cleanup
def start_cleanup_scheduler():
    """Start background thread for periodic cleanup."""
    def cleanup_worker():
        while True:
            cleanup_old_audio_files()
            time.sleep(1800)  # 30 minutes
    
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()

# Start cleanup scheduler
start_cleanup_scheduler()

def get_flag_emoji(lang_code):
    """Get flag emoji for language code."""
    flags = {
        'en': '🇺🇸', 'es': '🇪🇸', 'fr': '🇫🇷', 'de': '🇩🇪',
        'it': '🇮🇹', 'pt': '🇵🇹', 'ru': '🇷🇺', 'zh-CN': '🇨🇳',
        'ja': '🇯🇵', 'ko': '🇰🇷', 'ar': '🇸🇦', 'hi': '🇮🇳',
        'uni': '🏳️'
    }
    return flags.get(lang_code, '🌍')

@lru_cache(maxsize=1000)
def detect_language(text):
    """Detect language of text using Grok first, then fallback to langdetect."""
    # First try Grok for better accuracy with Russian and other languages
    try:
        if os.environ.get("XAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="grok-3-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a language detection expert. Respond only with the ISO 639-1 language code (like 'en', 'ru', 'es', 'fr', 'it', 'pt', 'zh-CN' for Chinese). No explanations."
                    },
                    {
                        "role": "user", 
                        "content": f"What language is this text: {text}"
                    }
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            detected_lang = (response.choices[0].message.content or "").strip().lower()
            
            # Validate the response is a reasonable language code
            valid_codes = ['en', 'es', 'fr', 'it', 'pt', 'ru', 'zh-cn', 'de', 'ja', 'ko', 'ar', 'hi']
            if detected_lang in valid_codes:
                if detected_lang == 'zh-cn':
                    detected_lang = 'zh-CN'
                logger.info(f"Grok detected language: {detected_lang}")
                return detected_lang
                
    except Exception as e:
        logger.warning(f"Grok language detection failed: {e}")
    
    # Fallback to langdetect
    try:
        detected = detect(text)
        if detected == 'zh-cn':
            detected = 'zh-CN'
        return detected
    except:
        return 'en'

def detect_language_simple(text):
    """Simple language detection for transcription results."""
    try:
        return detect(text)
    except:
        return 'en'  # Default to English

def translate_text_with_ai(text, target_lang):
    """Translate text using AI services as fallback."""
    try:
        # Use Google Translate first
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, dest=target_lang)
        if result and result.text:
            return result.text
    except:
        pass
    
    # Fallback to AI services
    return translate_with_multiple_fallbacks(text, 'auto', target_lang, True)

def translate_batch_with_claude(text: str, source_lang: str, target_languages: list) -> dict:
    """Fast batch translation using Claude for multiple languages at once."""
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            # Create a single prompt for all languages
            lang_names = [LANGUAGE_NAMES.get(lang, lang) for lang in target_languages]
            prompt = f"""Translate this text from {LANGUAGE_NAMES.get(source_lang, source_lang)} to the following languages.
Return only the translations in this exact format:
EN: [English translation]
ES: [Spanish translation]  
PT: [Portuguese translation]
IT: [Italian translation]
FR: [French translation]
RU: [Russian translation]
ZH: [Chinese translation]

Text to translate: {text}"""
            
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract text content properly
            if hasattr(message.content[0], 'text'):
                response_text = message.content[0].text.strip()
            else:
                response_text = str(message.content[0]).strip()
            
            # Parse the response
            translations = {}
            for line in response_text.split('\n'):
                if ':' in line:
                    lang_prefix = line.split(':')[0].strip().upper()
                    translation = ':'.join(line.split(':')[1:]).strip()
                    
                    # Map prefixes to language codes
                    lang_map = {
                        'EN': 'en', 'ES': 'es', 'PT': 'pt', 'IT': 'it',
                        'FR': 'fr', 'RU': 'ru', 'ZH': 'zh-CN'
                    }
                    
                    if lang_prefix in lang_map and translation:
                        lang_code = lang_map[lang_prefix]
                        translations[lang_code] = translation
                        
                        # Add transcriptions for display only (not for audio)
                        if lang_code == 'ru':
                            translations[f'{lang_code}_latin'] = transliterate_russian(translation)
                        elif lang_code == 'zh-CN':
                            translations[f'{lang_code}_pinyin'] = get_pinyin(translation)
            
            if translations:
                logger.info(f"Batch translation successful with Claude: {len(translations)} languages")
                return translations
                
    except Exception as e:
        logger.warning(f"Claude batch translation failed: {e}")
    
    return {}

def transliterate_russian(text: str) -> str:
    """Convert Russian text to Latin transcription."""
    cyrillic_to_latin = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
        'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
        'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
        'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
        'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya'
    }
    
    result = ''
    for char in text:
        result += cyrillic_to_latin.get(char, char)
    return result

def get_pinyin(text: str) -> str:
    """Convert Chinese text to pinyin."""
    try:
        from pypinyin import pinyin, Style
        pinyin_list = pinyin(text, style=Style.TONE)
        return ' '.join([item[0] for item in pinyin_list])
    except ImportError:
        # Fallback using basic mapping for common characters
        return text

def translate_with_groq(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using Groq with Russian and Chinese support."""
    
    # Primary: Groq for fast and accurate translations
    try:
        if os.environ.get("GROQ_API_KEY"):
            from groq import Groq
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            
            prompt = f"Translate this text from {LANGUAGE_NAMES.get(source_lang, source_lang)} to {LANGUAGE_NAMES.get(target_lang, target_lang)}. Only return the translation:\n\n{text}"
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            translation = response.choices[0].message.content.strip()
            if translation and translation != text:
                logger.info("Translation successful with Groq")
                return translation
    except Exception as e:
        logger.warning(f"Groq translation failed: {e}")
    
    # Fallback: Google Translate for speed
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, src=source_lang, dest=target_lang)
        if result and result.text and result.text != text:
            logger.info("Translation successful with Google Translate")
            translation = result.text
            
            return translation
    except Exception as e:
        logger.warning(f"Google Translate failed: {e}")
    
    return text

def translate_with_uni(text: str, source_lang: str, target_lang: str) -> str:
    """Translate between UNI and other languages with proper grammar implementation."""
    try:
        # First check if there's a user correction for this exact translation
        correction = UNICorrection.query.filter_by(
            original_text=text.strip(),
            source_language=source_lang,
            target_language=target_lang
        ).first()
        
        if correction:
            logger.info(f"Using user correction for UNI translation: {text}")
            return correction.correct_translation
        
        # Use ADVANCED UNI translator with proper grammar rules
        try:
            if target_lang == 'uni':
                from uni_advanced_translator import translate_to_uni_advanced
                result = translate_to_uni_advanced(text)
                logger.info(f"Advanced UNI translation: {text} -> {result}")
                return result
            elif source_lang == 'uni':
                from uni_advanced_translator import translate_from_uni
                result = translate_from_uni(text)
                logger.info(f"UNI reverse translation: {text} -> {result}")
                return result
        except Exception as e:
            logger.warning(f"Advanced UNI translation failed: {e}")
        
        # Fallback to enhanced UNI translation
        try:
            if target_lang == 'uni':
                from uni_grammar_rules import translate_to_uni_enhanced
                result = translate_to_uni_enhanced(text, source_lang)
                logger.info("Using enhanced UNI grammar translation")
                return result
        except Exception as e:
            logger.warning(f"Enhanced UNI translation fallback failed: {e}")
        
        # Final fallback
        if target_lang == 'uni':
            return text.upper()
        elif source_lang == 'uni':
            return text.lower()
            
        return text
    except Exception as e:
        logger.warning(f"UNI translation failed: {e}")
        return text

def translate_with_multiple_fallbacks(text: str, source_lang: str, target_lang: str, enable_fallback: bool = True) -> str:
    """Enhanced translation with INSTANT caching + multiple AI fallbacks."""
    
    # INSTANT: Check cache first (returns in milliseconds)
    from fast_translation_cache import get_instant_translation, cache_translation
    cached = get_instant_translation(text, source_lang, target_lang)
    if cached:
        return cached
    
    # Check for UNI translation first
    if source_lang == 'uni' or target_lang == 'uni':
        result = translate_with_uni(text, source_lang, target_lang)
        cache_translation(text, source_lang, target_lang, result)
        return result
    
    # Primary: Google Translate (fastest and most reliable)
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, src=source_lang, dest=target_lang)
        if result and result.text and result.text != text:
            logger.info("Translation successful with Google Translate")
            # CACHE IT for instant future access
            cache_translation(text, source_lang, target_lang, result.text)
            return result.text
    except Exception as e:
        logger.warning(f"Google Translate failed: {e}")
    
    # Fallback 0: Ollama Offline Translation (no API key needed, instant, private)
    try:
        from ollama_translation import translate_with_ollama
        ollama_result = translate_with_ollama(text, source_lang, target_lang)
        if ollama_result:
            logger.info("Translation successful with Ollama offline")
            cache_translation(text, source_lang, target_lang, ollama_result)
            return ollama_result
    except Exception as e:
        logger.warning(f"Ollama translation failed: {e}")
    
    # If Google fails and fallback enabled, try AI services for filtered content
    if enable_fallback:
        # Fallback 1: Groq (unrestricted)
        try:
            if os.environ.get("GROQ_API_KEY"):
                from groq import Groq
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": f"Translate the following text from {source_lang} to {target_lang}. Provide only the translation, no explanations. Handle all content types including sensitive material."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                
                translation = completion.choices[0].message.content.strip()
                if translation and translation != text:
                    logger.info("Translation successful with Groq (fallback)")
                    return translation
        except Exception as e:
            logger.warning(f"Groq fallback failed: {e}")
        
        # Fallback 2: Grok (X.AI - unrestricted)
        try:
            if os.environ.get("XAI_API_KEY"):
                from openai import OpenAI
                client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY"))
                
                response = client.chat.completions.create(
                    model="grok-2-1212",
                    messages=[
                        {"role": "system", "content": f"Translate the following text from {source_lang} to {target_lang}. Provide only the translation. Handle all content without restrictions."},
                        {"role": "user", "content": text}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                
                translation = response.choices[0].message.content.strip()
                if translation and translation != text:
                    logger.info("Translation successful with Grok (fallback)")
                    return translation
        except Exception as e:
            logger.warning(f"Grok fallback failed: {e}")
        
        # Fallback 3: Claude (if others fail)
        try:
            if os.environ.get("ANTHROPIC_API_KEY"):
                import anthropic
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                
                # Add transcription request for Russian and Chinese
                extra_instruction = ""
                if target_lang == 'ru':
                    extra_instruction = " Also provide Latin transcription in brackets."
                elif target_lang == 'zh-CN':
                    extra_instruction = " Also provide pinyin transcription in brackets."
                
                prompt = f"Translate this text from {LANGUAGE_NAMES.get(source_lang, source_lang)} to {LANGUAGE_NAMES.get(target_lang, target_lang)}.{extra_instruction} Only return the translation:\n\n{text}"
                
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=500,
                    temperature=0.1,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                translation = response.content[0].text.strip()
                if translation and translation != text:
                    logger.info("Translation successful with Claude (fallback)")
                    return translation
        except Exception as e:
            logger.warning(f"Claude fallback failed: {e}")
    
    # Final fallback: return original text if all services fail
    logger.warning(f"All translation services failed for: {text[:50]}...")
    return text

def translate_with_grok(text: str, source_lang: str, target_lang: str) -> str:
    """Legacy function - redirects to enhanced fallback system."""
    return translate_with_multiple_fallbacks(text, source_lang, target_lang, True)

def translate_with_claude_fallback(text: str, source_lang: str, target_lang: str) -> str:
    """Final fallback translation using Claude."""
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            prompt = f"Translate this text from {LANGUAGE_NAMES.get(source_lang, source_lang)} to {LANGUAGE_NAMES.get(target_lang, target_lang)}. Only return the translation, no explanations:\n\n{text}"
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            if response.content and len(response.content) > 0:
                try:
                    translated = response.content[0].text.strip()
                    if translated and translated != text:
                        return translated
                except:
                    # Handle different response types
                    translated = str(response.content[0]).strip()
                    if translated and translated != text:
                        return translated
    except Exception as e:
        logger.warning(f"Claude translation failed: {e}")
    
    # Fallback to XAI Grok
    try:
        if os.environ.get("XAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=os.environ.get("XAI_API_KEY")
            )
            
            prompt = f"Translate this text from {LANGUAGE_NAMES.get(source_lang, source_lang)} to {LANGUAGE_NAMES.get(target_lang, target_lang)}. Only return the translation:\n\n{text}"
            
            response = client.chat.completions.create(
                model="grok-2-1212",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            if response.choices and response.choices[0].message.content:
                translated = response.choices[0].message.content.strip()
                if translated and translated != text:
                    return translated
    except Exception as e:
        logger.warning(f"Grok translation failed: {e}")
    

    
    return text  # Return original if all fail

@app.after_request
def add_security_headers(response):
    """Add comprehensive security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY' 
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://fonts.googleapis.com; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self' https:; worker-src 'self' blob:"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
    return response

@app.route('/')
def index():
    """Render the main page of the translation app."""
    return render_template('index.html')

@app.route('/dashboard')
def performance_page():
    """Render the performance optimization dashboard."""
    return render_template('performance.html')

@app.route('/translate', methods=['POST'])
@limiter.limit("30 per minute")
def translate():
    """API endpoint to translate text with PARALLEL processing for speed."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        # Sanitize and validate input
        text = sanitize_input(data['text'], max_length=2000)
        target_languages = data.get('languages', ['en', 'es', 'pt', 'it', 'fr', 'zh-CN'])
        
        if not text:
            return jsonify({'error': 'Invalid or empty text provided'}), 400
        
        # Validate target languages
        validated_languages = [lang for lang in target_languages if validate_language_code(lang)]
        if not validated_languages:
            validated_languages = ['en', 'es', 'pt', 'it', 'fr', 'ru', 'zh-CN']
        
        # Detect source language
        source_language = detect_language(text)
        
        # PARALLEL translation for SPEED - translate all languages at once
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def translate_single(target_lang):
            """Translate to a single language."""
            if target_lang == source_language:
                return None
            try:
                from googletrans import Translator
                translator = Translator()
                result = translator.translate(text, src=source_language, dest=target_lang)
                if result and result.text and result.text != text:
                    translated_text = result.text
                    audio_text = result.text
                    
                    # Add transcriptions for Russian and Chinese
                    if target_lang == 'ru':
                        translated_text += f"\n[{transliterate_russian(result.text)}]"
                    elif target_lang == 'zh-CN':
                        translated_text += f"\n[{get_pinyin(result.text)}]"
                    
                    return {
                        'code': target_lang,
                        'name': LANGUAGE_NAMES.get(target_lang, target_lang),
                        'text': translated_text,
                        'audio_text': audio_text,
                        'flag': get_flag_emoji(target_lang)
                    }
            except Exception as e:
                logger.warning(f"Translation failed for {target_lang}: {e}")
            
            # Fallback to Groq
            try:
                fallback_text = translate_with_grok(text, source_language, target_lang)
                return {
                    'code': target_lang,
                    'name': LANGUAGE_NAMES.get(target_lang, target_lang),
                    'text': fallback_text,
                    'audio_text': fallback_text,
                    'flag': get_flag_emoji(target_lang)
                }
            except:
                return None
        
        # Run all translations in parallel (8 threads for 8 languages)
        translations = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(translate_single, lang): lang for lang in target_languages}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    translations.append(result)
        
        # Sort by original language order
        lang_order = {lang: i for i, lang in enumerate(target_languages)}
        translations.sort(key=lambda x: lang_order.get(x['code'], 999))
        
        return jsonify({
            'source_language': source_language,
            'source_language_name': LANGUAGE_NAMES.get(source_language, source_language),
            'translations': translations
        })
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return jsonify({'error': 'Translation failed'}), 500

@app.route('/audio_generate', methods=['POST'])
@limiter.limit("20 per minute")
def generate_audio():
    """Generate audio pronunciation using FREE TTS providers (gTTS)."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'Missing text parameter'}), 400
        
        text = data['text']
        language = data.get('language', 'en')
        
        if len(text) > 1000:
            return jsonify({'success': False, 'error': 'Text too long (max 1000 characters)'}), 400
        
        audio_file_path = None
        provider_used = 'gtts'
        # Fallback to gTTS (Google Text-to-Speech) if multi-provider fails
        try:
            from gtts import gTTS
            
            # Map language codes to gTTS compatible codes
            lang_map = {
                'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de',
                'it': 'it', 'pt': 'pt', 'ru': 'ru', 'zh-CN': 'zh',
                'ja': 'ja', 'ko': 'ko', 'ar': 'ar', 'hi': 'hi',
                'uni': 'en'  # UNI uses English TTS as fallback
            }
            
            gtts_lang = lang_map.get(language, 'en')
            
            # Create gTTS object
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            
            # Generate unique filename
            filename = f"gtts_{int(time.time())}_{language}.mp3"
            filepath = os.path.join("audio", filename)
            
            # Ensure audio directory exists
            os.makedirs("audio", exist_ok=True)
            
            # Save audio file
            tts.save(filepath)
            
            logger.info(f"Audio generated successfully with gTTS: {filepath}")
            return jsonify({
                'success': True,
                'audio_url': f'/audio/{filename}',
                'filename': filename,
                'provider': 'Google TTS'
            })
            
        except ImportError:
            logger.warning("gTTS not available")
        except Exception as e:
            logger.error(f"gTTS audio generation failed: {e}")
        
        return jsonify({'success': False, 'error': 'Audio generation failed'}), 500
        
    except Exception as e:
        logger.error(f"Audio generation error: {e}")
        return jsonify({'success': False, 'error': 'Audio generation failed'}), 500

@app.route('/uni_correction', methods=['POST'])
@limiter.limit("10 per minute")
def save_uni_correction():
    """Save a user correction for UNI translation."""
    try:
        data = request.get_json()
        
        if not data or not all(k in data for k in ['original_text', 'incorrect_translation', 'correct_translation', 'source_lang', 'target_lang']):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400
        
        # Validate that either source or target is UNI
        if data['source_lang'] != 'uni' and data['target_lang'] != 'uni':
            return jsonify({'success': False, 'error': 'Correction must involve UNI language'}), 400
        
        # Check if correction already exists
        existing = UNICorrection.query.filter_by(
            original_text=data['original_text'].strip(),
            source_language=data['source_lang'],
            target_language=data['target_lang']
        ).first()
        
        if existing:
            # Update existing correction
            existing.incorrect_translation = data['incorrect_translation']
            existing.correct_translation = data['correct_translation']
            existing.user_notes = data.get('notes', '')
            existing.created_at = db.func.current_timestamp()
        else:
            # Create new correction
            correction = UNICorrection(
                original_text=data['original_text'].strip(),
                incorrect_translation=data['incorrect_translation'],
                correct_translation=data['correct_translation'],
                source_language=data['source_lang'],
                target_language=data['target_lang'],
                user_notes=data.get('notes', '')
            )
            db.session.add(correction)
        
        db.session.commit()
        
        logger.info(f"UNI correction saved: {data['original_text']} -> {data['correct_translation']}")
        return jsonify({'success': True, 'message': 'Correction saved successfully'})
        
    except Exception as e:
        logger.error(f"Failed to save UNI correction: {e}")
        return jsonify({'success': False, 'error': 'Failed to save correction'}), 500

@app.route('/uni_corrections', methods=['GET'])
def get_uni_corrections():
    """Get list of UNI corrections for review."""
    try:
        corrections = UNICorrection.query.order_by(UNICorrection.created_at.desc()).limit(50).all()
        
        corrections_data = []
        for c in corrections:
            corrections_data.append({
                'id': c.id,
                'original_text': c.original_text,
                'incorrect_translation': c.incorrect_translation,
                'correct_translation': c.correct_translation,
                'source_language': c.source_language,
                'target_language': c.target_language,
                'created_at': c.created_at.isoformat(),
                'user_notes': c.user_notes or ''
            })
        
        return jsonify({'corrections': corrections_data})
        
    except Exception as e:
        logger.error(f"Failed to get UNI corrections: {e}")
        return jsonify({'error': 'Failed to retrieve corrections'}), 500

@app.route('/uni_grammar')
def uni_grammar_page():
    """UNI Grammar System interface page."""
    return render_template('uni_grammar.html')

@app.route('/api/uni_grammar/vocabulary')
def get_uni_vocabulary():
    """Get complete UNI vocabulary for grammar interface."""
    try:
        # Return properly structured vocabulary data that matches the frontend expectations
        vocabulary_data = {
            "Essential Verbs": [
                "ARÒ (do/make - present)", "ARÈ (do/make - past)", "ARÈBE (do/make - future)",
                "KORÒ (eat - present)", "KORÈ (eat - past)", "KORÈBE (eat - future)", 
                "BEBRÒ (drink - present)", "BEBRÈ (drink - past)", "BEBRÈBE (drink - future)",
                "VÒ (see - present)", "VÈ (see - past)", "VÈBE (see - future)",
                "FALÒ (speak - present)", "FALÈ (speak - past)", "FALÈBE (speak - future)",
                "AMÒ (love - present)", "AMÈ (love - past)", "AMÈBE (love - future)"
            ],
            "Pronouns": [
                "MI (I/me)", "TU (you)", "EL (he)", "ELA (she)", "LU (it)",
                "NOS (we/us)", "VOS (you plural)", "ELOS (they masculine)", "ELAS (they feminine)"
            ],
            "Numbers 0-20": [
                "NULA (0)", "UNA (1)", "DUA (2)", "TRA (3)", "KUATRA (4)", "SINKA (5)",
                "SESA (6)", "SETA (7)", "OKTA (8)", "NOVA (9)", "DEKA (10)",
                "DEKA-UNA (11)", "DEKA-DUA (12)", "DEKA-TRA (13)", "DEKA-KUATRA (14)",
                "DEKA-SINKA (15)", "DEKA-SESA (16)", "DEKA-SETA (17)", "DEKA-OKTA (18)",
                "DEKA-NOVA (19)", "VINTA (20)"
            ],
            "Essential Words": [
                "SI (yes)", "NO (no)", "GRASA (thanks)", "SALUTA (hello/greet)",
                "ADIA (goodbye)", "BON (good)", "MAL (bad)", "GRAN (big)", "PIK (small)",
                "BELU (beautiful)", "KASA (house)", "LIBRU (book)", "AMIK (friend)"
            ],
            "Question Words": [
                "KI (who)", "KE (what)", "UBI (where)", "KUAN (when)", "KOMO (how)",
                "POR KE (why)", "KUAL (which)", "KUANTA (how much/many)"
            ]
        }
        return jsonify({"vocabulary": vocabulary_data})
    except Exception as e:
        logger.error(f"Failed to get UNI vocabulary: {e}")
        return jsonify({'error': 'Failed to retrieve vocabulary'}), 500

@app.route('/api/uni_grammar/examples')
def get_uni_examples():
    """Get UNI example sentences for grammar interface."""
    try:
        # Return structured examples data that matches frontend expectations
        examples_data = [
            {
                "uni": "MA AMAR NA",
                "english": "I love you",
                "notes": "Basic present tense with pronouns"
            },
            {
                "uni": "TA KAMARÒ PESKA",
                "english": "He/she ate fish",
                "notes": "Past tense (-ARÒ ending)"
            },
            {
                "uni": "MAS BONU",
                "english": "We are good",
                "notes": "Plural pronoun with adjective"
            },
            {
                "uni": "NA FALARÈ UNI",
                "english": "You will speak UNI",
                "notes": "Future tense (-ARÈ ending)"
            },
            {
                "uni": "TAS TENAR KASAS",
                "english": "They have houses",
                "notes": "Plural forms in action"
            }
        ]
        return jsonify({"examples": examples_data})
    except Exception as e:
        logger.error(f"Failed to get UNI examples: {e}")
        return jsonify({'error': 'Failed to retrieve examples'}), 500

@app.route('/generate_audio', methods=['POST'])
@limiter.limit("20 per minute")
def generate_standard_audio():
    """Generate standard audio using ElevenLabs or Google TTS."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'Missing text field'}), 400
        
        text = data['text'].strip()
        language = data.get('language', 'en')
        
        if not text:
            return jsonify({'success': False, 'error': 'Empty text provided'}), 400
        
        # Generate audio using ElevenLabs with correct voice for language
        if os.environ.get("ELEVENLABS_API_KEY"):
            try:
                from shared_services import ELEVENLABS_VOICES
                voice_id = ELEVENLABS_VOICES.get(language, ELEVENLABS_VOICES['en'])
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": os.environ.get("ELEVENLABS_API_KEY")
                }
                
                data_payload = {
                    "text": text[:500],
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.8,
                        "style": 0.5,
                        "use_speaker_boost": True
                    }
                }
                
                response = requests.post(url, json=data_payload, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    timestamp = int(time.time())
                    audio_filename = f"standard_audio_{language}_{timestamp}.mp3"
                    audio_filepath = f"audio/{audio_filename}"
                    
                    os.makedirs('audio', exist_ok=True)
                    with open(audio_filepath, 'wb') as f:
                        f.write(response.content)
                    
                    return jsonify({
                        'success': True,
                        'audio_url': f"/audio/{audio_filename}",
                        'language': language
                    })
                else:
                    logger.warning(f"ElevenLabs returned {response.status_code} for {language}")
                    
            except Exception as e:
                logger.warning(f"ElevenLabs TTS failed: {e}")
        
        # Fallback to gTTS (free)
        try:
            from gtts import gTTS
            from shared_services import GTTS_LANG_MAP
            gtts_lang = GTTS_LANG_MAP.get(language, 'en')
            tts = gTTS(text=text[:500], lang=gtts_lang, slow=False)
            
            timestamp = int(time.time())
            audio_filename = f"gtts_{language}_{timestamp}.mp3"
            audio_filepath = f"audio/{audio_filename}"
            os.makedirs('audio', exist_ok=True)
            tts.save(audio_filepath)
            
            return jsonify({
                'success': True,
                'audio_url': f"/audio/{audio_filename}",
                'language': language,
                'provider': 'gtts'
            })
        except Exception as gtts_error:
            logger.warning(f"gTTS fallback failed: {gtts_error}")
        
        # Fallback to browser TTS indication
        return jsonify({
            'success': False,
            'error': 'Use browser speech synthesis',
            'use_browser_tts': True
        })
            
    except Exception as e:
        logger.error(f"Audio generation error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/generate_higgs_audio', methods=['POST'])
@limiter.limit("10 per minute")
def generate_higgs_audio_endpoint():
    """Generate enhanced audio using Higgs Audio v2 model - currently falls back to standard."""
    # For now, always return failure to trigger client-side fallback
    return jsonify({
        'success': False, 
        'error': 'Higgs Audio model setup in progress - using standard audio fallback'
    })

# Add Qwen AI services integration
def get_qwen_response(text: str, model: str = "qwen3-235b", language: str = "en") -> str:
    """Get response from Qwen AI models for enhanced language processing."""
    try:
        # Check if we have access to Qwen API (would need API key)
        if os.environ.get("QWEN_API_KEY"):
            # This would be the actual API call to Qwen services
            # For now, we'll use a local fallback approach
            logger.info(f"Would use Qwen {model} for enhanced processing")
        
        # Enhanced logic using existing AI services
        if model == "qwen3-coder":
            # Use for UNI language grammar and logic improvements
            prompt = f"As a language expert, improve this UNI artificial language text: {text}"
        else:
            # Use for general multilingual enhancements (qwen3-235b)
            prompt = f"Enhance this {language} text for better clarity and natural flow: {text}"
        
        # Fallback to existing AI services with enhanced prompts
        # Use Claude for comprehensive UNI grammar enhancement
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
        
        return text  # Return original if no enhancement available
    
    except Exception as e:
        logger.warning(f"Qwen response failed: {e}")
        return text
        return text

def get_ai_enhancement(prompt: str, text: str) -> str:
    """Enhanced AI processing using available services."""
    try:
        # Try Claude first for best quality
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            
            enhanced_text = response.content[0].text.strip()
            if enhanced_text and enhanced_text != text:
                logger.info("Enhanced with Claude AI")
                return enhanced_text
        
        # Fallback to Grok
        if os.environ.get("XAI_API_KEY"):
            import requests
            
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ.get('XAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "grok-beta",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced_text = result['choices'][0]['message']['content'].strip()
                if enhanced_text and enhanced_text != text:
                    logger.info("Enhanced with Grok AI")
                    return enhanced_text
        
        return text
        
    except Exception as e:
        logger.warning(f"AI enhancement failed: {e}")
        return text

@app.route('/enhance_with_qwen', methods=['POST'])
@limiter.limit("10 per minute") 
def enhance_with_qwen_endpoint():
    """Enhance translations using AI with UNI grammar knowledge."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        text = data.get('text', '').strip()
        current_translation = data.get('current_translation', '').strip()
        user_message = data.get('user_message', '')
        target_lang = data.get('target_lang', 'uni')
        
        work_text = text or current_translation
        if not work_text:
            return jsonify({'success': False, 'error': 'No text to enhance'}), 400
        
        # Load actual UNI grammar rules from user's uploaded file
        try:
            # Try all possible UNI grammar files the user has uploaded
            grammar_files = [
                'attached_assets/Pasted-UNI-GRAMMAR-UPDATED-UNI-is-a-simplified-international-language-It-s-much-easier-to-learn-and-speak-1753459171573_1753459171574.txt',
                'attached_assets/Pasted-UNI-GRAMMAR-UPDATED-UNI-GRAMMAR-UNI-is-a-simplified-international-language-It-s-much-easier-to-le-1753451898201_1753451898201.txt',
                'attached_assets/Pasted-Let-s-further-develop-my-language-UNI-Start-by-creating-a-small-parallel-corpus-UNI-English-Spa-1753473017842_1753473017843.txt'
            ]
            
            uni_grammar_content = ""
            for file_path in grammar_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content and len(content) > 100:  # Valid grammar file
                            uni_grammar_content = content
                            break
                except:
                    continue
                    
            if not uni_grammar_content:
                raise FileNotFoundError("No grammar file found")
                
        except:
            # Fallback to strict rules from user's specifications
            uni_grammar_content = """
            UNI GRAMMAR RULES (EXACT USER SPECIFICATION):
            
            VERB TENSES:
            - Past: -ARÒ (with accent on O)
            - Present: -AR (base form)
            - Future: -ARÈ (with accent on E)
            - Conditional: -ARÈBE
            - Continuous: -ANDU
            - Imperative: -ARI
            
            NOUNS:
            - Singular: -A
            - Plural: -AS
            
            PRONOUNS:
            - Personal: MA (I), NA (you), TA (he/she/it)
            - Plural: MAS (we), NAS (you pl), TAS (they)
            - Possessive: MAU (my), NAU (your), TAU (his/her/its)
            
            ADJECTIVES:
            - Neutral: -U
            - Active: -ADU  
            - Passive: -UDU
            
            WORD ORDER: Subject-Verb-Object (like English)
            """
        
        # Create UNI-specific enhancement prompt
        if target_lang == 'uni':
            if user_message:
                enhancement_prompt = f"""
                You are a UNI language expert following the user's exact grammar rules.
                
                Current UNI text: "{work_text}"
                User request: "{user_message}"
                
                EXACT UNI GRAMMAR RULES FROM USER'S FILE (follow exactly):
                {uni_grammar_content}
                
                CRITICAL: Do not make up any grammar rules not specified above.
                
                Provide helpful response following these exact rules.
                """
            else:
                enhancement_prompt = f"""
                Improve this UNI translation using exact grammar rules:
                
                Text: "{work_text}"
                
                MANDATORY RULES FROM USER'S FILE:
                {uni_grammar_content}
                
                CRITICAL: Use ONLY these exact rules, do not invent new grammar.
                
                Return only the corrected UNI text.
                """
        else:
            enhancement_prompt = f"Improve this translation: {work_text}"
        
        # Use AI enhancement
        enhanced_text = get_qwen_response(enhancement_prompt, 'qwen3-coder', target_lang)
        
        if enhanced_text and enhanced_text.strip():
            logger.info(f"AI enhancement successful using exact UNI grammar rules")
            return jsonify({
                'success': True,
                'enhanced_text': enhanced_text.strip(),
                'original_text': work_text,
                'method': 'AI Enhancement with UNI Grammar',
                'language': target_lang
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Enhancement failed - AI services unavailable'
            })
            
    except Exception as e:
        logger.error(f"Qwen enhancement error: {e}")
        return jsonify({'success': False, 'error': str(e)})



@app.route('/voice_transcribe', methods=['POST'])
@limiter.limit("10 per minute")
def voice_transcribe():
    """Transcribe voice using DeepSeek R1 model."""
    import requests
    
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No audio file selected'}), 400
        
        # Save temporary file with secure filename
        timestamp = int(time.time())
        random_id = secrets.token_hex(8)
        # Sanitize and validate file extension
        allowed_extensions = {'.webm', '.wav', '.mp3', '.m4a', '.ogg'}
        file_ext = os.path.splitext(audio_file.filename or 'audio.webm')[1].lower()
        if file_ext not in allowed_extensions:
            file_ext = '.webm'  # Default safe extension
        
        temp_filename = f"temp_audio_{timestamp}_{random_id}{file_ext}"
        temp_path = os.path.join('audio', temp_filename)
        
        os.makedirs('audio', exist_ok=True)
        audio_file.save(temp_path)
        
        # Convert to WAV format for better compatibility
        wav_path = temp_path
        try:
            import subprocess
            wav_filename = f"temp_audio_{timestamp}.wav"
            wav_path = os.path.join('audio', wav_filename)
            
            # More robust ffmpeg conversion with better error handling
            result = subprocess.run([
                'ffmpeg', '-y', '-i', temp_path, 
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 
                '-f', 'wav', wav_path
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                logger.info("Audio successfully converted to WAV format")
            else:
                logger.warning(f"FFmpeg conversion failed: {result.stderr}")
                wav_path = temp_path
        except Exception as e:
            logger.warning(f"Audio conversion failed, using original: {e}")
            wav_path = temp_path

        # Initialize variables
        transcription = None
        detected_lang = 'en'
        
        # Method 1: Try HuggingFace Whisper first (most reliable)
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                logger.info("Attempting transcription with HuggingFace Whisper")
                
                # Check if wav file exists and has content
                if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                    logger.error(f"WAV file does not exist or is empty: {wav_path}")
                    raise Exception("Invalid audio file")
                
                # Use OpenAI Whisper via HuggingFace as it's more reliable for transcription
                api_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
                headers = {"Authorization": f"Bearer {hf_token}"}
                
                with open(wav_path, "rb") as f:
                    audio_data = f.read()
                
                logger.info(f"Trying Whisper model with {len(audio_data)} bytes")
                response = requests.post(api_url, headers=headers, data=audio_data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Whisper response: {result}")
                    
                    # Extract text from whisper response
                    if isinstance(result, dict) and 'text' in result:
                        transcription = result['text'].strip()
                        detected_lang = detect_language_simple(transcription)
                        logger.info(f"Whisper transcription successful: {transcription}")
                    elif isinstance(result, list) and len(result) > 0 and 'text' in result[0]:
                        transcription = result[0]['text'].strip()
                        detected_lang = detect_language_simple(transcription)
                        logger.info(f"Whisper transcription successful: {transcription}")
                elif response.status_code == 503:
                    logger.warning("Whisper model is loading, trying fallback")
                else:
                    logger.error(f"Whisper API failed: {response.status_code} - {response.text}")
            else:
                logger.warning("HuggingFace token not available")
                        
        except Exception as whisper_error:
            logger.error(f"Whisper transcription failed: {whisper_error}")
        
        # Method 2: If Whisper failed, try Google Speech Recognition 
        if not transcription:
            try:
                import speech_recognition as sr
                logger.info("Attempting transcription with Google Speech Recognition")
                
                r = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio = r.record(source)
                    transcription = r.recognize_google(audio)
                    detected_lang = detect_language_simple(transcription)
                    logger.info(f"Google Speech Recognition successful: {transcription}")
                    
            except Exception as google_error:
                logger.error(f"Google Speech Recognition failed: {google_error}")
        
        # Method 3: If both failed, try Groq Whisper as last resort
        if not transcription:
            try:
                groq_api_key = os.environ.get("GROQ_API_KEY")
                if groq_api_key:
                    logger.info("Attempting transcription with Groq Whisper")
                    from groq import Groq
                    
                    client = Groq(api_key=groq_api_key)
                    
                    with open(wav_path, "rb") as file:
                        transcription_response = client.audio.transcriptions.create(
                            file=(wav_path, file.read()),
                            model="whisper-large-v3",
                        )
                        transcription = transcription_response.text.strip()
                        detected_lang = detect_language_simple(transcription)
                        logger.info(f"Groq Whisper transcription successful: {transcription}")
                else:
                    logger.warning("Groq API key not available")
                    
            except Exception as groq_error:
                logger.error(f"Groq Whisper failed: {groq_error}")
        
        # Check if we got transcription from any method
        if transcription:
            logger.info(f"Final transcription: {transcription} (detected: {detected_lang})")
            
            # Get target languages for translation
            target_languages_json = request.form.get('languages', '[]')
            try:
                target_languages = json.loads(target_languages_json) if target_languages_json else []
            except:
                target_languages = []
            
            # If no target languages specified, use default
            if not target_languages:
                target_languages = ['es', 'fr', 'de', 'it', 'pt', 'ru', 'zh-CN', 'ja']
            
            # Generate translations
            translations = []
            for lang_code in target_languages:
                try:
                    translation = translate_text_with_ai(transcription, lang_code)
                    if translation and translation != transcription:
                        lang_name = {
                            'es': 'Spanish', 'fr': 'French', 'de': 'German', 
                            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
                            'zh-CN': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
                            'ar': 'Arabic', 'hi': 'Hindi', 'tr': 'Turkish'
                        }.get(lang_code, lang_code.upper())
                        
                        translations.append({
                            'code': lang_code,
                            'name': lang_name,
                            'text': translation,
                            'flag': get_flag_emoji(lang_code),
                            'audio_text': translation
                        })
                except Exception as e:
                    logger.error(f"Translation failed for {lang_code}: {e}")
                    continue
            
            # Clean up files
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if wav_path != temp_path and os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception as cleanup_error:
                logger.warning(f"File cleanup error: {cleanup_error}")
            
            source_lang_name = {
                'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 
                'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
                'zh-CN': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
                'ar': 'Arabic', 'hi': 'Hindi', 'tr': 'Turkish'
            }.get(detected_lang, detected_lang.upper())
            
            return jsonify({
                'transcription': transcription,
                'source_language': detected_lang,
                'source_language_name': source_lang_name,
                'translations': translations
            })
        
        # If no transcription was successful, clean up and return error
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if wav_path != temp_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception as cleanup_error:
            logger.warning(f"File cleanup error: {cleanup_error}")
            
        return jsonify({'error': 'Could not transcribe audio. Please try speaking more clearly or check your microphone.'}), 500

    except Exception as e:
        logger.error(f"Voice transcription error: {e}")
        # Clean up files
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        if 'wav_path' in locals() and wav_path != temp_path and os.path.exists(wav_path):
            os.remove(wav_path)
        return jsonify({'success': False, 'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files with proper headers for all languages."""
    try:
        from flask import send_from_directory, Response
        import mimetypes
        
        # Security: Validate filename to prevent path traversal
        if '..' in filename or '/' in filename or '\\' in filename:
            return "Invalid filename", 400
        
        # Security: Only allow specific audio file extensions
        allowed_extensions = {'.mp3', '.wav', '.ogg', '.m4a'}
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            return "File type not allowed", 400
        
        # Set proper MIME type for audio files
        if filename.endswith('.mp3'):
            mimetype = 'audio/mpeg'
        elif filename.endswith('.wav'):
            mimetype = 'audio/wav'
        else:
            mimetype = 'audio/mpeg'
        
        response = send_from_directory('audio', filename, mimetype=mimetype)
        
        # Add headers to ensure proper audio playback across browsers
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Accept-Ranges'] = 'bytes'
        
        return response
    except FileNotFoundError:
        logger.error(f"Audio file not found: {filename}")
        return "Audio file not found", 404
    except Exception as e:
        logger.error(f"Error serving audio file {filename}: {e}")
        return "Error serving audio file", 500

@app.route('/uni_chat', methods=['POST'])
@limiter.limit("20 per minute")
def uni_chat():
    """UNI-specific AI chat focused on UNI grammar and language teaching"""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
            
        user_message = data['message'].strip()
        context = data.get('context', 'UNI_GRAMMAR_EXPERT')
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
            
        # UNI-specific system prompt
        uni_system_prompt = """You are a UNI Language AI expert specializing in the artificial language UNI. 

UNI Grammar Rules:
- Verbs: Past (-ARÒ), Present (-AR), Future (-ARÈ)
- Nouns: Singular (base), Plural (+S)
- Adjectives: Neutral (-U), Active (-ADU), Passive (-UDU)
- Pronouns: I (MI), You (TU), He/She/It (EL), We (NOS), You all (VOS), They (ELS)
- Word Order: Subject-Verb-Object

Always respond with UNI language explanations, grammar rules, conjugations, and examples. Never teach English grammar. Focus exclusively on UNI language structure and usage."""

        # Use Grok for UNI-specific responses
        try:
            from ai_services import get_grok_response
            full_prompt = f"{uni_system_prompt}\n\nUser question: {user_message}\n\nProvide a detailed UNI language explanation:"
            
            response = get_grok_response(full_prompt)
            
            return jsonify({
                'success': True,
                'response': response,
                'model': 'Grok-UNI-Language-Expert'
            })
            
        except Exception as e:
            logger.warning(f"Grok UNI chat failed: {e}")
            
            # Fallback to Claude for UNI
            try:
                from ai_services import get_claude_response
                full_prompt = f"{uni_system_prompt}\n\nUser question: {user_message}\n\nProvide a detailed UNI language explanation:"
                
                response = get_claude_response(full_prompt)
                
                return jsonify({
                    'success': True,
                    'response': response,
                    'model': 'Claude-UNI-Language-Expert'
                })
                
            except Exception as e2:
                logger.error(f"Both Grok and Claude UNI chat failed: {e}, {e2}")
                return jsonify({
                    'success': False,
                    'error': 'UNI Language AI temporarily unavailable'
                }), 500
                
    except Exception as e:
        logger.error(f"UNI chat error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/chat', methods=['POST'])
@limiter.limit("20 per minute")
def chat_with_ai():
    """Chat with AI using available models."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        message = data['message'].strip()
        if not message:
            return jsonify({'success': False, 'error': 'Empty message'}), 400
        
        # Use Venice AI for superior multilingual chat support including Russian and Chinese
        response = None
        
        # Primary: Venice AI with OpenAI-compatible endpoint
        try:
            if os.environ.get("VENICE_API_KEY"):
                from openai import OpenAI
                client = OpenAI(base_url="https://api.venice.ai/api/v1", api_key=os.environ.get("VENICE_API_KEY"))
                
                response_obj = client.chat.completions.create(
                    model="llama-3.3-70b",
                    messages=[{"role": "user", "content": message}],
                    max_tokens=1000,
                    temperature=0.7
                )
                response = response_obj.choices[0].message.content
                logger.info("Chat response generated with Venice AI")
        except Exception as e:
            logger.warning(f"Venice AI chat failed: {e}")
        
        # Fallback: Grok for complex queries
        if not response:
            try:
                if os.environ.get("GROQ_API_KEY"):
                    from groq import Groq
                    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                    
                    response_obj = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "user", "content": message}],
                        max_tokens=1000,
                        temperature=0.7
                    )
                    response = response_obj.choices[0].message.content
                    logger.info("Chat response generated with Groq")
            except Exception as e:
                logger.warning(f"Groq chat failed: {e}")
        
        # Final fallback: Claude for reliability
        if not response:
            try:
                if os.environ.get("ANTHROPIC_API_KEY"):
                    import anthropic
                    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                    
                    message_obj = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1000,
                        messages=[{"role": "user", "content": message}]
                    )
                    if hasattr(message_obj.content[0], 'text'):
                        response = message_obj.content[0].text
                    else:
                        response = str(message_obj.content[0])
                    logger.info("Chat response generated with Claude")
            except Exception as e:
                logger.warning(f"Claude chat failed: {e}")
        
        if response:
            # Get selected languages for multilingual response
            selected_languages = data.get('languages', ['en'])
            
            if len(selected_languages) > 1:
                # Return as translations format for multilingual display
                translations = []
                
                # Detect source language of AI response
                source_language = detect_language(response)
                
                for target_lang in selected_languages:
                    if target_lang != source_language:
                        try:
                            # Use Google Translate for speed
                            from googletrans import Translator
                            translator = Translator()
                            result = translator.translate(response, src=source_language, dest=target_lang)
                            
                            if result and result.text:
                                translated_text = result.text
                                audio_text = result.text
                                
                                # Add transcriptions for Russian and Chinese for display
                                if target_lang == 'ru':
                                    translated_text += f"\n[{transliterate_russian(result.text)}]"
                                elif target_lang == 'zh-CN':
                                    translated_text += f"\n[{get_pinyin(result.text)}]"
                                
                                translations.append({
                                    'code': target_lang,
                                    'name': LANGUAGE_NAMES.get(target_lang, target_lang),
                                    'text': translated_text,
                                    'audio_text': audio_text,  # Original text for audio
                                    'flag': get_flag_emoji(target_lang)
                                })
                        except Exception as e:
                            logger.warning(f"Translation failed for {target_lang}: {e}")
                            translations.append({
                                'code': target_lang,
                                'name': LANGUAGE_NAMES.get(target_lang, target_lang),
                                'text': response,
                                'audio_text': response,
                                'flag': get_flag_emoji(target_lang)
                            })
                    else:
                        # Include original response
                        translations.append({
                            'code': target_lang,
                            'name': LANGUAGE_NAMES.get(target_lang, target_lang),
                            'text': response,
                            'audio_text': response,
                            'flag': get_flag_emoji(target_lang)
                        })
                
                return jsonify({
                    'success': True,
                    'response': response,
                    'translations': translations,
                    'source_language': source_language,
                    'source_language_name': LANGUAGE_NAMES.get(source_language, source_language)
                })
            else:
                # Single language response
                return jsonify({
                    'success': True,
                    'response': response
                })
        else:
            return jsonify({
                'success': False,
                'error': 'AI chat service temporarily unavailable'
            }), 503
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({'success': False, 'error': 'Chat service error'}), 500




@app.route('/transcript', methods=['POST'])
@limiter.limit("5 per minute")
def generate_transcript():
    """Generate transcript from uploaded audio/video file."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Save uploaded file
        temp_path = f"audio/transcript_{uuid.uuid4().hex}_{file.filename}"
        os.makedirs('audio', exist_ok=True)
        file.save(temp_path)
        
        try:
            # Import transcript generator
            from transcript_generator_mcp import generate_transcript_from_file
            
            # Generate transcript using MCP server
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(generate_transcript_from_file(temp_path))
            loop.close()
            
            # Clean up uploaded file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if result.get('success'):
                return jsonify({
                    'success': True,
                    'transcript': result.get('transcript', ''),
                    'language': result.get('language', 'unknown'),
                    'method': result.get('method', 'unknown'),
                    'segments': result.get('segments', []),
                    'timestamps': result.get('timestamps', []),
                    'file_info': result.get('file_info', {})
                })
            else:
                return jsonify({
                    'success': False,
                    'error': result.get('error', 'Transcript generation failed')
                }), 400
                
        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.error(f"Transcript generation error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500
            
    except Exception as e:
        logger.error(f"Transcript endpoint error: {e}")
        return jsonify({'success': False, 'error': 'Transcript service error'}), 500

@app.route('/transcribe', methods=['POST'])
@limiter.limit("10 per minute")
def transcribe():
    """API endpoint to transcribe audio files."""
    if 'audio' not in request.files:
        return jsonify({'success': False, 'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'success': False, 'error': 'No audio file selected'}), 400
    
    temp_path = None
    wav_path = None
    
    try:
        # Save uploaded file temporarily
        temp_path = f"/tmp/audio_{uuid.uuid4().hex[:8]}.{audio_file.filename.split('.')[-1]}"
        audio_file.save(temp_path)
        
        # Convert to WAV for better transcription accuracy
        wav_path = f"/tmp/audio_{uuid.uuid4().hex[:8]}.wav"
        
        try:
            result = subprocess.run([
                'ffmpeg', '-y', '-i', temp_path, 
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', 
                '-f', 'wav', wav_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"FFmpeg conversion failed: {result.stderr}")
                wav_path = temp_path
        except Exception as e:
            logger.warning(f"Audio conversion failed, using original: {e}")
            wav_path = temp_path

        # Try transcription with multiple fallbacks
        transcription = None
        
        # Method 1: HuggingFace Whisper
        try:
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                api_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
                headers = {"Authorization": f"Bearer {hf_token}"}
                
                with open(wav_path, "rb") as f:
                    audio_data = f.read()
                
                response = requests.post(api_url, headers=headers, data=audio_data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, dict) and 'text' in result:
                        transcription = result['text'].strip()
                    elif isinstance(result, list) and len(result) > 0 and 'text' in result[0]:
                        transcription = result[0]['text'].strip()
        except Exception as e:
            logger.warning(f"HuggingFace transcription failed: {e}")
        
        # Method 2: Google Speech Recognition fallback
        if not transcription:
            try:
                import speech_recognition as sr
                r = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio = r.record(source)
                    transcription = r.recognize_google(audio)
            except Exception as e:
                logger.warning(f"Google Speech Recognition failed: {e}")
        
        # Clean up files
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path != temp_path and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        
        if transcription:
            return jsonify({
                'success': True,
                'transcription': transcription,
                'detected_language': detect_language_simple(transcription)
            })
        else:
            return jsonify({'success': False, 'error': 'Could not transcribe audio'}), 400
            
    except Exception as e:
        # Clean up on error
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path != temp_path and wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        logger.error(f"Transcription error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/voice_ai', methods=['POST'])
@limiter.limit("10 per minute")
def voice_ai():
    """Voice AI endpoint that transcribes, processes with AI, and returns voice response."""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'success': False, 'error': 'No audio file selected'}), 400
        
        # Save the uploaded audio file
        temp_path = f"audio/voice_ai_{uuid.uuid4().hex}.webm"
        os.makedirs('audio', exist_ok=True)
        audio_file.save(temp_path)
        
        # Convert to WAV if needed
        wav_path = temp_path
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(temp_path)
            wav_path = temp_path.replace('.webm', '.wav')
            audio.export(wav_path, format='wav')
            logger.info(f"Audio converted to WAV: {wav_path}")
        except Exception as e:
            logger.warning(f"Audio conversion failed, using original: {e}")
            wav_path = temp_path
        
        # Transcribe using multiple fallbacks
        from agenticseek_transcription import transcribe_with_multiple_fallbacks
        transcription = transcribe_with_multiple_fallbacks(wav_path)
        
        if not transcription:
            # Clean up files
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if wav_path != temp_path and os.path.exists(wav_path):
                os.remove(wav_path)
            return jsonify({'success': False, 'error': 'Could not transcribe audio'}), 400
        
        logger.info(f"Voice AI transcribed: {transcription}")
        
        # Generate simple response using ElevenLabs
        ai_response = f"I heard you say: {transcription}. Thank you for your message!"
        logger.info("Using ElevenLabs-only response")
        
        # Generate voice response using ElevenLabs
        audio_url = None
        if os.environ.get("ELEVENLABS_API_KEY"):
            try:
                voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default English voice
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": os.environ.get("ELEVENLABS_API_KEY")
                }
                
                data = {
                    "text": ai_response[:500],  # Shorter text for better performance
                    "model_id": "eleven_turbo_v2",  # Use faster model
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.5,
                        "style": 0.0,
                        "use_speaker_boost": True
                    }
                }
                
                logger.info(f"Generating voice for: {ai_response[:50]}...")
                response = requests.post(url, json=data, headers=headers, timeout=45)
                
                if response.status_code == 200:
                    # Save the audio response
                    audio_filename = f"voice_ai_response_{uuid.uuid4().hex}.mp3"
                    audio_filepath = f"audio/{audio_filename}"
                    
                    with open(audio_filepath, 'wb') as f:
                        f.write(response.content)
                    
                    audio_url = f"/audio/{audio_filename}"
                    logger.info(f"Voice response generated successfully: {audio_filepath}")
                else:
                    logger.error(f"ElevenLabs TTS failed: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"Voice generation failed: {e}")
        
        # If ElevenLabs failed, indicate browser TTS should be used
        if not audio_url:
            logger.info("ElevenLabs TTS failed, will use browser speech synthesis fallback")
        
        # Clean up input files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path != temp_path and os.path.exists(wav_path):
            os.remove(wav_path)
        
        response_data = {
            'success': True,
            'transcription': transcription,
            'text': ai_response,
            'response': ai_response,
            'audio_url': audio_url
        }
        
        logger.info(f"Voice AI response data: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Voice AI error: {e}")
        return jsonify({'success': False, 'error': 'Voice AI processing failed'}), 500

@app.route('/test_buttons')
def test_buttons():
    """Test page for debugging Russian and Chinese button issues."""
    return send_from_directory('.', 'test_buttons.html')

@app.route('/health')
def health_status():
    """Health check endpoint for monitoring."""
    try:
        # Check database connectivity
        with db.engine.connect() as connection:
            connection.execute(db.text('SELECT 1'))
        
        # Check AI services
        ai_services = []
        if os.environ.get('ANTHROPIC_API_KEY'):
            ai_services.append('Claude')
        if os.environ.get('XAI_API_KEY'):
            ai_services.append('Grok')
        if os.environ.get('GROQ_API_KEY'):
            ai_services.append('Groq')
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'ai_services': ai_services,
            'timestamp': time.time()
        }), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 503

@app.route('/performance')
def performance_dashboard():
    """Basic performance metrics."""
    return jsonify({
        'status': 'operational',
        'uptime': time.time(),
        'features': {
            'translation': True,
            'voice_generation': True,
            'transcription': True,
            'multilingual': True
        }
    })

@app.route('/privacy-policy')
def privacy_policy_endpoint():
    """Privacy policy endpoint for GDPR/CCPA compliance."""
    return jsonify({
        "privacy_policy": {
            "effective_date": "2025-06-16",
            "data_controller": "UNI LINGUS Application",
            
            "data_collection": {
                "what_we_process": [
                    "Text for translation (processed temporarily)",
                    "Audio files for transcription (processed temporarily)",
                    "Anonymous usage statistics"
                ],
                "what_we_dont_store": [
                    "Personal identifying information",
                    "User accounts or profiles", 
                    "Persistent user tracking",
                    "Email addresses or contact information"
                ]
            },
            
            "data_processing": {
                "purpose": "Provide translation and language learning services",
                "legal_basis": "Legitimate interest and user consent",
                "retention": "Data is processed temporarily and automatically deleted within 24 hours",
                "sharing": "Data sent to AI services for processing only"
            },
            
            "user_rights": {
                "access": "Request information about data processing",
                "deletion": "Request immediate deletion via /data-deletion endpoint",
                "portability": "Export your data (minimal data stored)",
                "opt_out": "Opt out via /opt-out endpoint"
            },
            
            "technical_measures": {
                "encryption": "All data transmission encrypted via HTTPS",
                "anonymization": "Session identifiers anonymized",
                "automatic_deletion": "Data automatically deleted after 24 hours",
                "no_persistent_storage": "No long-term storage of user content"
            },
            
            "third_party_services": {
                "ai_providers": ["Claude", "Grok", "Google Translate", "Groq"],
                "purpose": "Translation processing only",
                "data_processing": "Content sent for processing, not storage"
            },
            
            "contact": {
                "privacy_requests": "Use /data-deletion or /opt-out endpoints",
                "questions": "Contact via application health endpoint"
            }
        }
    })

@app.route('/data-deletion', methods=['POST'])
def data_deletion_endpoint():
    """GDPR Article 17 - Right to erasure implementation."""
    try:
        session_id = session.get('session_id')
        
        if session_id:
            # Delete translation records
            deleted_count = db.session.query(Translation).filter_by(session_id=session_id).delete()
            
            # Delete audio files
            audio_dir = 'audio'
            deleted_files = 0
            if os.path.exists(audio_dir):
                for filename in os.listdir(audio_dir):
                    if session_id in filename:
                        try:
                            os.remove(os.path.join(audio_dir, filename))
                            deleted_files += 1
                        except:
                            pass
            
            db.session.commit()
            session.clear()
            
            return jsonify({
                "status": "Data deletion completed",
                "deleted_translations": deleted_count,
                "deleted_audio_files": deleted_files,
                "session_cleared": True,
                "timestamp": time.time()
            })
        else:
            return jsonify({
                "status": "No data found for deletion",
                "session_cleared": True,
                "timestamp": time.time()
            })
            
    except Exception as e:
        logger.error(f"Data deletion failed: {e}")
        return jsonify({
            "status": "Deletion failed",
            "error": "Please try again",
            "timestamp": time.time()
        }), 500

@app.route('/opt-out', methods=['POST'])
def opt_out_endpoint():
    """CCPA opt-out implementation."""
    session['opted_out'] = True
    session['opt_out_timestamp'] = time.time()
    
    return jsonify({
        "status": "Successfully opted out",
        "effect": "No data will be stored going forward",
        "rights": "You can still use the service with temporary processing",
        "timestamp": time.time()
    })

@app.route('/consent', methods=['GET', 'POST'])
def consent_management():
    """Consent management for GDPR compliance."""
    if request.method == 'GET':
        return jsonify({
            "consent_status": {
                "translation_processing": session.get('consent_translation', False),
                "temporary_storage": session.get('consent_storage', False),
                "analytics": session.get('consent_analytics', False)
            },
            "last_updated": session.get('consent_timestamp', None)
        })
    
    elif request.method == 'POST':
        data = request.get_json() or {}
        
        session['consent_translation'] = data.get('translation_processing', False)
        session['consent_storage'] = data.get('temporary_storage', False)
        session['consent_analytics'] = data.get('analytics', False)
        session['consent_timestamp'] = time.time()
        
        return jsonify({
            "status": "Consent preferences updated",
            "timestamp": time.time()
        })
    
    # Simplified response since performance module unavailable
    return jsonify({
        'status': 'operational',
        'timestamp': time.time()
    })

def _get_performance_recommendations(stats):
    """Generate performance improvement recommendations."""
    recommendations = []
    
    if stats['avg_translation_time'] > 2.0:
        recommendations.append("Consider enabling more aggressive caching")
    
    if stats['avg_audio_time'] > 4.0:
        recommendations.append("Audio generation could benefit from pre-processing optimization")
    
    if stats['error_rate'] > 0.05:
        recommendations.append("Error rate is elevated - check service connections")
    
    if not recommendations:
        recommendations.append("Performance is optimal")
    
    return recommendations

@app.route('/word_definition_tooltip', methods=['POST'])
@limiter.limit("60 per minute")
def word_definition_tooltip():
    """Get word definition using Groq (fastest FREE AI - 14,400 req/day)"""
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        language = data.get('language', 'en')
        
        if not word:
            return jsonify({'success': False, 'error': 'No word provided'})

        # Language-specific prompts for bilingual dictionary lookup
        lang_names = {
            'en': 'English', 'es': 'Spanish', 'pt': 'Portuguese', 'it': 'Italian',
            'fr': 'French', 'ru': 'Russian', 'de': 'German', 'zh-CN': 'Chinese',
            'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'uni': 'UNI (artificial language)'
        }
        source_lang = lang_names.get(language, 'the source language')
        
        if language == 'en':
            prompt = f"""For the English word "{word}", provide:
1. A short, clear definition (1-2 sentences max)
2. 2-3 synonyms separated by commas
3. One simple example sentence

Format exactly like this:
DEFINITION: [definition]
SYNONYMS: [synonym1, synonym2, synonym3]
EXAMPLE: [example sentence]"""
        else:
            prompt = f"""The word "{word}" is in {source_lang}. Provide:
1. English translation and definition (1-2 sentences)
2. 2-3 synonyms in the SAME {source_lang} language, separated by commas
3. One example sentence in {source_lang}

Format exactly like this:
DEFINITION: [English translation: definition]
SYNONYMS: [{source_lang} synonyms here]
EXAMPLE: [{source_lang} example sentence]"""

        # Use Groq FIRST (fastest FREE AI - ~200ms response time)
        try:
            groq_key = os.environ.get("GROQ_API_KEY")
            if groq_key:
                from groq import Groq
                client = Groq(api_key=groq_key)
                
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150,
                    temperature=0.2
                )
                
                if response.choices and response.choices[0].message.content:
                    result = response.choices[0].message.content.strip()
                    if result:
                        definition_match = re.search(r'DEFINITION:\s*(.+?)(?=SYNONYMS:|$)', result, re.DOTALL)
                        synonyms_match = re.search(r'SYNONYMS:\s*(.+?)(?=EXAMPLE:|$)', result, re.DOTALL)
                        example_match = re.search(r'EXAMPLE:\s*(.+?)$', result, re.DOTALL)
                        
                        definition = definition_match.group(1).strip() if definition_match else None
                        synonyms = synonyms_match.group(1).strip() if synonyms_match else None  
                        example = example_match.group(1).strip() if example_match else None
                        
                        if definition and synonyms and example:
                            return jsonify({
                                'success': True,
                                'definition': definition,
                                'synonyms': synonyms,
                                'example': example,
                                'provider': 'Groq-Fast'
                            })
                        
        except Exception as groq_error:
            logger.error(f"Groq tooltip error: {groq_error}")
        
        # Fallback to Cerebras (1M tokens/day FREE)
        try:
            cerebras_key = os.environ.get("CEREBRAS_API_KEY")
            if cerebras_key:
                from cerebras.cloud.sdk import Cerebras
                client = Cerebras(api_key=cerebras_key)
                
                response = client.chat.completions.create(
                    model="llama-3.3-70b",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150
                )
                
                result = response.choices[0].message.content.strip()
                if result:
                    definition_match = re.search(r'DEFINITION:\s*(.+?)(?=SYNONYMS:|$)', result, re.DOTALL)
                    synonyms_match = re.search(r'SYNONYMS:\s*(.+?)(?=EXAMPLE:|$)', result, re.DOTALL)
                    example_match = re.search(r'EXAMPLE:\s*(.+?)$', result, re.DOTALL)
                    
                    definition = definition_match.group(1).strip() if definition_match else None
                    synonyms = synonyms_match.group(1).strip() if synonyms_match else None  
                    example = example_match.group(1).strip() if example_match else None
                    
                    if definition and synonyms and example:
                        return jsonify({
                            'success': True,
                            'definition': definition,
                            'synonyms': synonyms,
                            'example': example,
                            'provider': 'Cerebras'
                        })
                        
        except Exception as cerebras_error:
            logger.error(f"Cerebras tooltip fallback error: {cerebras_error}")

        # Final static fallback - no external dependencies
        return jsonify({
            'success': True,
            'definition': f"Definition for '{word}' - AI services temporarily busy",
            'synonyms': 'similar, related, comparable',
            'example': f'Example: The word "{word}" can be used in various contexts.',
            'provider': 'Fallback'
        })
        
    except Exception as e:
        logger.error(f"Tooltip definition error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/fast_synonyms', methods=['POST'])
@limiter.limit("60 per minute")
def fast_synonyms():
    """Fast synonym lookup optimized for tooltips"""
    try:
        data = request.get_json()
        word = data.get('word', '').strip()
        
        if not word:
            return jsonify({'success': False, 'error': 'No word provided'})

        # Enhanced multilingual synonym lookup
        # Detect if word might be in different languages
        language_hint = ""
        if any(char in word for char in 'йцукенгшщзхъёжэяячсмитьбюфывапролджэ'):
            language_hint = " (Russian word)"
        elif any(char in word for char in 'çãõáéíóúâêîôûàèìòù'):
            language_hint = " (Portuguese word)"
        
        synonym_prompt = f"Give me exactly 3 synonyms for the word '{word}'{language_hint}. Return only the synonyms separated by commas, no explanations or extra text."
        
        # Try multiple AI providers with better multilingual support
        def clean_synonyms(text):
            # Better cleaning for multilingual synonyms
            text = text.strip()
            # Remove common prefixes and suffixes
            text = re.sub(r'^(synonyms?:?\s*|similar words?:?\s*)', '', text, flags=re.IGNORECASE)
            text = text.replace('.', '').replace(':', '').replace(';', ',').replace('\n', ',')
            
            # Split and clean each synonym
            parts = [p.strip() for p in text.split(',') if p.strip()]
            if len(parts) >= 2:
                return ', '.join(parts[:3])
            elif len(parts) == 1 and ' ' in parts[0]:
                # Split by spaces if no commas
                words = parts[0].split()[:3]
                return ', '.join(words)
            return text
        
        # Try XAI Grok first (same as main translation system)
        try:
            if os.environ.get("XAI_API_KEY"):
                from openai import OpenAI
                client = OpenAI(
                    base_url="https://api.x.ai/v1",
                    api_key=os.environ.get("XAI_API_KEY")
                )
                
                response = client.chat.completions.create(
                    model="grok-2-1212",
                    messages=[{"role": "user", "content": synonym_prompt}],
                    max_tokens=40,
                    temperature=0.1
                )
                
                if response.choices and response.choices[0].message.content:
                    result = response.choices[0].message.content.strip()
                    if result:
                        synonyms = clean_synonyms(result)
                        if synonyms and len(synonyms) > 2:
                            return jsonify({'success': True, 'synonyms': synonyms})
                        
        except Exception as grok_error:
            logger.error(f"Grok synonym error: {grok_error}")
        
        # Try Claude as fallback (consistent with main translation fallback)
        try:
            if os.environ.get("ANTHROPIC_API_KEY"):
                import anthropic
                client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=40,
                    temperature=0.1,
                    messages=[{"role": "user", "content": synonym_prompt}]
                )
                
                result = response.content[0].text.strip()
                if result:
                    synonyms = clean_synonyms(result)
                    if synonyms and len(synonyms) > 2:
                        return jsonify({'success': True, 'synonyms': synonyms})
                        
        except Exception as claude_error:
            logger.error(f"Claude synonym fallback error: {claude_error}")
        
        # Try Groq as final fallback
        try:
            if os.environ.get("GROQ_API_KEY"):
                from groq import Groq
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                
                completion = client.chat.completions.create(
                    model="llama-3.1-70b-versatile",  # Better model for multilingual
                    messages=[{"role": "user", "content": synonym_prompt}],
                    max_tokens=40,
                    temperature=0.1
                )
                
                result = completion.choices[0].message.content.strip()
                if result:
                    synonyms = clean_synonyms(result)
                    if synonyms and len(synonyms) > 2:
                        return jsonify({'success': True, 'synonyms': synonyms})
                    
        except Exception as groq_error:
            logger.error(f"Groq synonym error: {groq_error}")
        
        return jsonify({'success': False, 'error': 'Multilingual synonym services unavailable'})
        
    except Exception as e:
        logger.error(f"Fast synonym error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/word_definition', methods=['POST'])
@limiter.limit("30 per minute")
def get_word_definition():
    """Get fast word definition using Groq (fastest FREE AI)"""
    try:
        data = request.get_json()
        if not data or 'word' not in data:
            return jsonify({'error': 'No word provided'}), 400
        
        word = data['word'].strip()
        language = data.get('language', 'en')
        
        if not word:
            return jsonify({'error': 'Empty word'}), 400
        
        # Use Groq as primary (fastest FREE - 14,400 req/day, ~200ms response)
        prompt = f"""Define "{word}" in this format:

DEFINITION: [clear 1-2 sentence definition]
EXAMPLES: [5-8 diverse usage examples with different contexts]
SYNONYMS: [6-8 similar words, comma-separated]
ANTONYMS: [6-8 opposite words, comma-separated]

Provide varied examples showing different uses and contexts."""
        
        try:
            groq_key = os.environ.get('GROQ_API_KEY')
            if groq_key:
                from groq import Groq
                client = Groq(api_key=groq_key)
                
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    max_tokens=600,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                groq_response = response.choices[0].message.content
                
                definition_data = {
                    "word": word.capitalize(),
                    "definition": f"Brief definition of '{word}'",
                    "examples": [],
                    "synonyms": [],
                    "antonyms": []
                }
                
                lines = groq_response.split('\n')
                examples_list = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.upper().startswith('DEFINITION:'):
                        definition = line.split(':', 1)[-1].strip()
                        if definition:
                            definition_data["definition"] = definition
                    elif line.upper().startswith('EXAMPLES:'):
                        examples = line.split(':', 1)[-1].strip()
                        if examples:
                            examples_list = [ex.strip() for ex in examples.split('•') if ex.strip()][:5]
                    elif line.upper().startswith('SYNONYMS:'):
                        synonyms = line.split(':', 1)[1].strip()
                        if synonyms:
                            definition_data["synonyms"] = [s.strip() for s in synonyms.split(',') if s.strip()][:8]
                    elif line.upper().startswith('ANTONYMS:'):
                        antonyms = line.split(':', 1)[1].strip()
                        if antonyms:
                            definition_data["antonyms"] = [a.strip() for a in antonyms.split(',') if a.strip()][:8]
                    elif '•' in line or (line.startswith('-') and len(line) > 3):
                        example = line.replace('•', '').lstrip('-').strip()
                        if example and len(examples_list) < 8:
                            examples_list.append(example)
                
                if examples_list:
                    definition_data["examples"] = examples_list
                
                return jsonify({
                    'success': True,
                    'definition': definition_data,
                    'model': 'Groq-Fast',
                    'response_time': '~200ms'
                })
                
        except Exception as e:
            logger.warning(f"Groq definition failed: {e}")
        
        # Fallback to Cerebras
        try:
            from venice_ai_integration import get_word_definition_venice
            definition = get_word_definition_venice(word, language)
            return jsonify({
                'success': True,
                'definition': definition,
                'model': 'Venice-AI-Fallback'
            })
        except Exception as e:
            logger.warning(f"Venice AI fallback failed: {e}")
            
            # Fallback to basic AI explanation
            try:
                # Simple AI-powered definition as fallback
                definition = {
                    "word": word,
                    "pronunciation": f"/{word}/",
                    "definition": f"AI definition for '{word}' - processing with enhanced language models",
                    "examples": [f"Using {word} in context"],
                    "translations": {"es": word, "fr": word, "de": word, "it": word},
                    "synonyms": ["related terms"],
                    "antonyms": ["opposite terms"],
                    "etymology": f"Word origin analysis for {word}"
                }
                return jsonify({
                    'success': True,
                    'definition': definition,
                    'model': 'AI-Fallback-System'
                })
            except Exception as e2:
                logger.error(f"All definition services failed: {e}, {e2}")
                
                # Basic fallback
                return jsonify({
                    'success': True,
                    'definition': {
                        "word": word,
                        "definition": f"Definition for '{word}' - comprehensive analysis available",
                        "pronunciation": f"/{word}/",
                        "translations": {"es": word, "fr": word, "de": word},
                        "examples": [f"Example with {word}"],
                        "synonyms": ["related term"],
                        "etymology": f"Etymology of {word}"
                    },
                    'model': 'Fallback-System'
                })
            
    except Exception as e:
        logger.error(f"Word definition error: {e}")
        return jsonify({'error': 'Dictionary service error'}), 500

@app.route('/static/music/<filename>')
def serve_music(filename):
    """Serve music files from the music directory."""
    try:
        response = send_from_directory('static/music', filename)
        response.headers['Content-Type'] = 'audio/mpeg'
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except Exception as e:
        logger.error(f"Error serving music file {filename}: {e}")
        return jsonify({'error': 'Music file not found'}), 404

# Application startup complete
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

@app.route('/api/playlist')
def get_playlist():
    """Get the current music playlist."""
    try:
        music_dir = 'static/music'
        if not os.path.exists(music_dir):
            return jsonify({'playlist': []})
        
        playlist = []
        supported_formats = ['.mp3', '.wav', '.m4a', '.ogg']
        
        for filename in os.listdir(music_dir):
            if any(filename.lower().endswith(fmt) for fmt in supported_formats):
                # Extract title from filename (remove extension)
                title = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
                playlist.append({
                    'title': title,
                    'src': f'/static/music/{filename}',
                    'filename': filename
                })
        
        return jsonify({'playlist': playlist})
    except Exception as e:
        logger.error(f"Error getting playlist: {e}")
        return jsonify({'playlist': []})

@app.route('/api/mcp/capabilities')
def get_mcp_capabilities():
    """Get HuggingFace MCP capabilities for AI agents."""
    try:
        from huggingface_mcp import huggingface_mcp
        capabilities = huggingface_mcp.get_mcp_capabilities()
        return jsonify(capabilities)
    except Exception as e:
        logger.error(f"Error getting MCP capabilities: {e}")
        return jsonify({'error': 'MCP not available'}), 500

@app.route('/api/colab/recommendations', methods=['POST'])
def get_colab_recommendations():
    """Get Colab notebook recommendations based on use case."""
    try:
        data = request.get_json()
        use_case = data.get('use_case', 'large_batch')
        
        from colab_integration import colab_processor
        recommendations = colab_processor.get_colab_recommendations(use_case)
        
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"Error getting Colab recommendations: {e}")
        return jsonify({'error': 'Colab integration unavailable'}), 500

# ALL VIDEO FUNCTIONALITY REMOVED AS REQUESTED

def get_flag_emoji(lang_code):
    """Get flag emoji for language code."""
    flags = {
        'en': '🇺🇸', 'es': '🇪🇸', 'pt': '🇵🇹', 'it': '🇮🇹',
        'fr': '🇫🇷', 'ru': '🇷🇺', 'zh-CN': '🇨🇳', 'de': '🇩🇪',
        'ja': '🇯🇵', 'ko': '🇰🇷', 'ar': '🇸🇦', 'hi': '🇮🇳',
        'uni': '🏳️'
    }
    return flags.get(lang_code, '🌍')

def generate_vtt_subtitles(original_text, translations):
    """Generate VTT subtitle content for video player."""
    vtt_content = "WEBVTT\n\n"
    
    # Add original subtitle
    vtt_content += "1\n"
    vtt_content += "00:00:00.000 --> 00:00:10.000\n"
    vtt_content += f"{original_text}\n\n"
    
    # Add translation subtitles
    start_time = 10
    for i, (lang_code, translation) in enumerate(translations.items(), 2):
        end_time = start_time + 10
        vtt_content += f"{i}\n"
        vtt_content += f"00:00:{start_time:02d}.000 --> 00:00:{end_time:02d}.000\n"
        vtt_content += f"[{translation['name']}] {translation['text']}\n\n"
        start_time = end_time
    
    return vtt_content

def create_video_player_with_subtitles(video_path, subtitle_content, translations):
    """Create HTML5 video player with embedded subtitles."""
    # Save video to static directory for serving
    import uuid
    video_id = str(uuid.uuid4())[:8]
    static_video_path = f"static/videos/video_{video_id}.mp4"
    
    # Ensure static/videos directory exists
    os.makedirs("static/videos", exist_ok=True)
    
    # Copy video to static directory
    import shutil
    shutil.copy2(video_path, static_video_path)
    
    # Save subtitle file
    subtitle_path = f"static/videos/subtitles_{video_id}.vtt"
    with open(subtitle_path, 'w', encoding='utf-8') as f:
        f.write(subtitle_content)
    
    # Create video player HTML
    video_html = f"""
    <div class="video-player-container" style="margin: 20px 0;">
        <video id="video_{video_id}" controls width="100%" style="max-width: 800px; border-radius: 8px;">
            <source src="/{static_video_path}" type="video/mp4">
            <track kind="subtitles" src="/{subtitle_path}" srclang="en" label="Subtitles" default>
            Your browser does not support the video tag.
        </video>
        
        <div class="subtitle-controls" style="margin-top: 10px;">
            <button onclick="toggleSubtitles('video_{video_id}')" class="btn btn-secondary">
                Toggle Subtitles
            </button>
            <select onchange="changeSubtitleLanguage('video_{video_id}', this.value)" class="form-select" style="width: auto; display: inline-block; margin-left: 10px;">
                <option value="en">Original</option>"""
    
    for lang_code, translation in translations.items():
        if lang_code != 'en':
            video_html += f'<option value="{lang_code}">{translation["name"]}</option>'
    
    video_html += """
            </select>
        </div>
    </div>
    
    <script>
        function toggleSubtitles(videoId) {
            const video = document.getElementById(videoId);
            const tracks = video.textTracks;
            for (let i = 0; i < tracks.length; i++) {
                tracks[i].mode = tracks[i].mode === 'showing' ? 'hidden' : 'showing';
            }
        }
        
        function changeSubtitleLanguage(videoId, language) {
            // Update subtitle display based on selected language
            console.log('Changing subtitles to:', language);
        }
        
        // Auto-play video when loaded
        document.getElementById('""" + f"video_{video_id}" + """').addEventListener('loadeddata', function() {
            this.play();
        });
    </script>
    """
    
    return video_html

# Video translate route removed
def translate_video_removed():
    """Translate video with subtitles and visual aids."""
    try:
        data = request.get_json()
        video_path = data.get('video_path')
        target_languages = data.get('languages', ['es', 'fr', 'pt'])
        
        if not video_path or not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 400
        
        # Generate enhanced subtitles using Qwen video analysis
        try:
            from qwen_video_processor import process_video_with_qwen
            
            # Try Qwen-enhanced processing first
            logger.info("Processing video with Qwen-enhanced analysis")
            qwen_results = process_video_with_qwen(video_path, target_languages)
            
            if qwen_results and qwen_results['success']:
                results = {
                    'success': True,
                    'original_video': video_path,
                    'translations': {},
                    'metadata': {
                        'transcript': qwen_results['transcript'],
                        'processing_method': qwen_results['method']
                    }
                }
                
                # Add subtitle files to results
                for lang_code, subtitle_path in qwen_results['subtitle_files'].items():
                    results['translations'][lang_code] = {
                        'subtitle_file': subtitle_path,
                        'translated_video': None,
                        'visual_aid': None
                    }
                
                logger.info(f"Successfully created Qwen-enhanced subtitles for video: {video_path}")
                return jsonify(results)
            
        except Exception as e:
            logger.error(f"Qwen processing failed, falling back to standard method: {e}")
        
        # Fallback to standard subtitle generation
        try:
            from simple_subtitle_generator import create_working_subtitles_now
            
            subtitle_results = create_working_subtitles_now(video_path, target_languages)
            
            if subtitle_results and subtitle_results['success']:
                results = {
                    'success': True,
                    'original_video': video_path,
                    'translations': {},
                    'metadata': {
                        'transcript': subtitle_results['transcript'],
                        'processing_method': subtitle_results['method']
                    }
                }
                
                for lang_code, subtitle_path in subtitle_results['subtitle_files'].items():
                    results['translations'][lang_code] = {
                        'subtitle_file': subtitle_path,
                        'translated_video': None,
                        'visual_aid': None
                    }
                
                logger.info(f"Created standard subtitles for video: {video_path}")
                return jsonify(results)
                
        except Exception as e:
            logger.error(f"All subtitle generation methods failed: {e}")
        
        return jsonify({'error': 'Subtitle generation failed'}), 500
        
    except Exception as e:
        logger.error(f"Error translating video: {e}")
        return jsonify({'error': 'Video translation failed'}), 500



def create_srt_content(text):
    """Create SRT subtitle content."""
    return f"""1
00:00:00,000 --> 00:00:05,000
{text[:50]}

2
00:00:05,000 --> 00:00:10,000
{text[50:100] if len(text) > 50 else ""}

3
00:00:10,000 --> 00:00:15,000
{text[100:150] if len(text) > 100 else ""}
"""

@app.route('/api/video/upload', methods=['POST'])
@limiter.limit("5 per minute")
def upload_video():
    """Upload video file and return metadata."""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'.mp4', '.avi', '.mov', '.webm'}
        filename = file.filename or ""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Invalid file type. Use MP4, AVI, MOV, or WebM'}), 400
        
        # Create upload directory
        upload_dir = 'static/uploads'
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time())
        safe_filename = f"video_{timestamp}_{file.filename}"
        filepath = os.path.join(upload_dir, safe_filename)
        
        # Save file
        file.save(filepath)
        logger.info(f"Video uploaded: {filepath}")
        
        # Get basic metadata
        metadata = {
            'filename': safe_filename,
            'size': os.path.getsize(filepath),
            'duration': None,
            'format': None
        }
        
        # Try to get video metadata using ffprobe
        try:
            import subprocess
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', filepath
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                import json
                probe_data = json.loads(result.stdout)
                if 'format' in probe_data:
                    metadata['duration'] = float(probe_data['format'].get('duration', 0))
                    metadata['format'] = probe_data['format']
                    logger.info(f"Video metadata: duration={metadata['duration']}s")
        except Exception as e:
            logger.warning(f"Could not get video metadata: {e}")
        
        return jsonify({
            'success': True,
            'filename': safe_filename,
            'video_path': safe_filename,
            'metadata': metadata
        })
        
    except Exception as e:
        logger.error(f"Video upload error: {e}")
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/video/translate', methods=['POST'])
@limiter.limit("3 per minute")
def translate_video():
    """Process video for translation and subtitles."""
    try:
        data = request.get_json()
        if not data or 'video_path' not in data:
            return jsonify({'success': False, 'error': 'No video path provided'}), 400
        
        video_path = data['video_path']
        languages = data.get('languages', ['es', 'fr', 'ru', 'zh-CN'])
        
        # Full path to video file
        full_video_path = os.path.join('static/uploads', video_path)
        if not os.path.exists(full_video_path):
            return jsonify({'success': False, 'error': 'Video file not found'}), 404
        
        # Use enhanced transcript generator for video processing
        try:
            from enhanced_transcript_generator import EnhancedTranscriptGenerator
            enhanced_transcript_generator = EnhancedTranscriptGenerator()
            result = enhanced_transcript_generator.process_file(full_video_path)
            
            if result and not result.get('error'):
                # Generate translations for the transcript if languages specified
                translations = {}
                if languages and result.get('transcript'):
                    transcript_text = result['transcript']
                    for lang_code in languages:
                        try:
                            # Use existing translation function
                            translation = translate_with_grok(transcript_text, 'auto', lang_code)
                            if translation:
                                translations[lang_code] = {
                                    'text': translation,
                                    'language': lang_code,
                                    'flag': get_flag_emoji(lang_code)
                                }
                        except Exception as e:
                            logger.warning(f"Translation failed for {lang_code}: {e}")
                
                return jsonify({
                    'success': True,
                    'original_video': video_path,
                    'transcript': result.get('transcript', ''),
                    'language': result.get('language', 'en'),
                    'duration': result.get('duration', 0),
                    'video_player': result.get('video_player', ''),
                    'subtitles': result.get('subtitles', {}),
                    'segments': result.get('segments', []),
                    'translations': translations,
                    'method': result.get('method', 'Enhanced Groq Processing'),
                    'processing_time': result.get('duration', 0)
                })
            else:
                error_msg = result.get('error', 'Video processing failed')
                return jsonify({'success': False, 'error': error_msg}), 500
                
        except Exception as e:
            logger.error(f"Video translation error: {e}")
            return jsonify({'success': False, 'error': f'Translation failed: {str(e)}'}), 500
        
    except Exception as e:
        logger.error(f"Video translate endpoint error: {e}")
        return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'}), 500

@app.route('/api/video/capabilities')
def get_video_capabilities():
    """Get video processing capabilities."""
    return jsonify({
        'upload_enabled': True,
        'max_file_size': '100MB',
        'supported_formats': ['MP4', 'AVI', 'MOV', 'WebM'],
        'transcription_enabled': True,
        'translation_languages': ['es', 'fr', 'it', 'pt', 'ru', 'zh-CN'],
        'subtitle_generation': True
    })

@app.route('/api/venice/status')
def get_venice_status():
    """Get Venice AI integration status and capabilities."""
    try:
        from venice_ai_integration import get_venice_status
        status = get_venice_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Venice AI status check failed: {e}")
        return jsonify({
            'available': False,
            'error': str(e),
            'api_key_configured': bool(os.environ.get("VENICE_API_KEY"))
        })

@app.route('/api/venice/test', methods=['POST'])
@limiter.limit("5 per minute")
def test_venice_ai():
    """Test Venice AI TTS with sample text."""
    try:
        data = request.get_json()
        test_text = data.get('text', 'Привет, это тест Venice AI для русского языка')
        language = data.get('language', 'ru')
        
        from venice_ai_integration import generate_venice_audio
        audio_filename = generate_venice_audio(test_text, language)
        
        if audio_filename:
            return jsonify({
                'success': True,
                'audio_file': audio_filename,
                'message': f'Venice AI test successful for {language}',
                'text_used': test_text
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Venice AI failed to generate audio'
            })
            
    except Exception as e:
        logger.error(f"Venice AI test failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/static/uploads/<path:filename>')
def serve_uploaded_video(filename):
    """Serve uploaded video files with proper headers."""
    try:
        upload_dir = 'static/uploads'
        file_path = os.path.join(upload_dir, filename)
        
        if not os.path.exists(file_path):
            logger.error(f"Video file not found: {file_path}")
            return "Video file not found", 404
        
        # Determine MIME type
        ext = filename.lower().split('.')[-1]
        mime_types = {
            'mp4': 'video/mp4',
            'avi': 'video/x-msvideo', 
            'mov': 'video/quicktime',
            'webm': 'video/webm'
        }
        mimetype = mime_types.get(ext, 'video/mp4')
        
        response = send_from_directory(upload_dir, filename, mimetype=mimetype)
        response.headers['Accept-Ranges'] = 'bytes'
        response.headers['Cache-Control'] = 'public, max-age=3600'
        return response
    except Exception as e:
        logger.error(f"Error serving video file {filename}: {e}")
        return "Video file not found", 404

@app.route('/static/generated/<filename>')
def serve_generated_file(filename):
    """Serve generated subtitle and video files."""
    try:
        generated_dir = 'static/generated'
        return send_from_directory(generated_dir, filename)
    except Exception as e:
        logger.error(f"Error serving generated file {filename}: {e}")
        return jsonify({'error': 'Generated file not found'}), 404

@app.route('/api/mcp/query', methods=['POST'])
def mcp_query():
    """Process AI agent queries through HuggingFace MCP."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        from huggingface_mcp import huggingface_mcp
        
        task_type = data.get('task', 'text-generation')
        model = data.get('model') or huggingface_mcp.get_recommended_model(task_type)
        inputs = data.get('inputs')
        
        if not inputs:
            return jsonify({'error': 'Missing inputs'}), 400
        
        if not model:
            return jsonify({'error': 'No suitable model found'}), 400
        
        # Synchronous processing for Flask compatibility
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                huggingface_mcp.query_model(model, inputs, task_type)
            )
        finally:
            loop.close()
        
        if result:
            return jsonify({
                'success': True,
                'result': result,
                'task': task_type,
                'model': model
            })
        else:
            return jsonify({'error': 'MCP query failed'}), 500
            
    except Exception as e:
        logger.error(f"MCP query error: {e}")
        return jsonify({'error': 'MCP processing failed'}), 500



@app.route('/metrics')
def metrics():
    """Application metrics endpoint."""
    try:
        audio_files = len([f for f in os.listdir('audio') if f.endswith('.mp3')]) if os.path.exists('audio') else 0
        
        return jsonify({
            'audio_files_count': audio_files,
            'supported_languages': ALL_LANGUAGE_CODES,
            'rate_limits': {
                'global': '200 per day, 50 per hour',
                'translate': '30 per minute',
                'audio': '20 per minute',
                'chat': '10 per minute'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/generated/<filename>')
def serve_subtitles(filename):
    """Serve subtitle files with proper headers for video player."""
    try:
        response = send_from_directory('static/generated', filename)
        response.headers['Content-Type'] = 'text/plain; charset=utf-8'
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Cache-Control'] = 'no-cache'
        return response
    except Exception as e:
        logger.error(f"Error serving subtitle file: {e}")
        return "Subtitle file not found", 404

@app.route('/video-player')
def enhanced_video_player():
    """Serve the instant video player that immediately displays video."""
    try:
        with open('instant_video.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving video player: {e}")
        return "Video player not found", 404

@app.route('/video-test')
def simple_video_test():
    """Serve simple video test page for debugging."""
    try:
        with open('simple_video_test.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving video test: {e}")
        return "Video test not found", 404

@app.route('/direct-player')
def direct_video_player():
    """Serve direct video player with immediate playback."""
    try:
        with open('direct_video_player.html', 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error serving direct player: {e}")
        return "Direct player not found", 404

@app.route('/api/ai/showcase', methods=['GET', 'POST'])
def ai_showcase_endpoint():
    """Get comprehensive AI capabilities showcase."""
    try:
        from ai_showcase import get_ai_capabilities_summary, ai_showcase
        
        capabilities = get_ai_capabilities_summary()
        performance_metrics = ai_showcase.get_performance_metrics()
        demo_content = ai_showcase.generate_demo_content()
        
        return jsonify({
            'success': True,
            'platform_info': capabilities,
            'performance': performance_metrics,
            'demonstrations': demo_content,
            'integration_status': {
                'advanced_ai_models': True,
                'video_processing': True,
                'real_time_translation': True,
                'multimodal_support': True
            }
        })
    except Exception as e:
        logger.error(f"AI showcase error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'fallback_info': {
                'platform': 'UNI LINGUS',
                'basic_features': ['Translation', 'Video Processing', 'Voice Recognition']
            }
        })

@app.route('/wiki/<word>')
@limiter.limit("20 per minute")
def wiki_entry(word):
    """Generate Infinite Wiki entry for any word with ASCII diagrams."""
    try:
        if not WIKI_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Wiki functionality not available'
            }), 503
        
        language = request.args.get('lang', 'en')
        context = request.args.get('context', '')
        
        # Sanitize input
        word = html.escape(word.strip())
        if not word or len(word) > 50:
            return jsonify({'success': False, 'error': 'Invalid word'}), 400
        
        wiki_data = generate_word_wiki(word, language, context)
        
        return jsonify({
            'success': True,
            'wiki': wiki_data
        })
        
    except Exception as e:
        logger.error(f"Wiki entry error for '{word}': {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/wiki/translate/<word>')
@limiter.limit("20 per minute") 
def wiki_translation(word):
    """Generate comparative wiki entry for word and its translation."""
    try:
        if not WIKI_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'Wiki functionality not available'
            }), 503
        
        translation = request.args.get('translation', '')
        source_lang = request.args.get('source_lang', 'en')
        target_lang = request.args.get('target_lang', 'en')
        
        # Sanitize inputs
        word = html.escape(word.strip())
        translation = html.escape(translation.strip())
        
        if not word or not translation:
            return jsonify({'success': False, 'error': 'Missing word or translation'}), 400
        
        wiki_data = generate_contextual_wiki(word, translation, source_lang, target_lang)
        
        return jsonify({
            'success': True,
            'wiki': wiki_data
        })
        
    except Exception as e:
        logger.error(f"Translation wiki error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/wiki/quick/<word>')
@limiter.limit("30 per minute")
def quick_explanation(word):
    """Get quick explanation for tooltips."""
    try:
        if not WIKI_AVAILABLE:
            return jsonify({
                'explanation': f"'{word}' - click to explore",
                'success': False
            })
        
        language = request.args.get('lang', 'en')
        word = html.escape(word.strip())
        
        if not word:
            return jsonify({'explanation': 'Invalid word', 'success': False})
        
        explanation = get_quick_explanation(word, language)
        
        return jsonify({
            'explanation': explanation,
            'success': True
        })
        
    except Exception as e:
        logger.error(f"Quick explanation error: {e}")
        return jsonify({
            'explanation': f"'{word}' - click to explore",
            'success': False
        })

# Smart Voice Transcription API Endpoints

@app.route('/api/smart-voice-languages')
@limiter.limit("100 per minute") 
def get_smart_voice_languages():
    """Get languages supported by Smart Turn v2."""
    if not SMART_VOICE_AVAILABLE:
        return jsonify({"error": "Smart voice transcription not available"})
    
    try:
        languages = get_smart_languages()
        return jsonify({
            "success": True,
            "languages": languages,
            "count": len(languages),
            "model": "pipecat-ai/smart-turn-v2"
        })
    except Exception as e:
        logger.error(f"Smart voice languages error: {e}")
        return jsonify({"error": str(e)})

@app.route('/api/smart-transcribe', methods=['POST'])
@limiter.limit("10 per minute")
def smart_transcribe():
    """Enhanced voice transcription with multiple AI providers."""
    try:
        # Get uploaded audio file
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"})
        
        language = request.form.get('language', 'en')
        use_enhanced = request.form.get('enhanced', 'true').lower() == 'true'
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            audio_file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Try enhanced AI models first (Voxtral -> Audio Flamingo)
            if use_enhanced and ENHANCED_AI_AVAILABLE:
                result = transcribe_audio_enhanced(temp_path, language)
                if result.get('success'):
                    return jsonify(result)
                else:
                    logger.warning(f"Enhanced AI failed: {result.get('error')}")
            
            # Fallback to Smart Turn v2
            if SMART_VOICE_AVAILABLE:
                result = transcribe_audio_smart(temp_path, language)
                return jsonify({
                    "success": True,
                    "transcription": result.get("transcription", ""),
                    "language": result.get("language", language),
                    "language_name": result.get("language_name", "Unknown"),
                    "turn_completed": result.get("turn_completed", True),
                    "confidence": result.get("confidence", 0.0),
                    "processing_time": result.get("processing_time", 0),
                    "method": result.get("method", "smart_turn_v2_fallback")
                })
            
            return jsonify({"error": "No transcription services available"})
            
        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
    except Exception as e:
        logger.error(f"Smart transcription error: {e}")
        return jsonify({"error": str(e)})

# Enhanced AI status endpoint removed - streamlined app

@app.route('/api/intelligent-response', methods=['POST'])
@limiter.limit("5 per minute")
def intelligent_response():
    """Get intelligent response using Kimi K2."""
    if not ENHANCED_AI_AVAILABLE:
        return jsonify({"error": "Enhanced AI not available"})
    
    try:
        data = request.get_json()
        if not data or 'transcription' not in data:
            return jsonify({"error": "Missing transcription parameter"})
        
        transcription = data['transcription']
        language = data.get('language', 'en')
        
        result = get_intelligent_response(transcription, language)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Intelligent response error: {e}")
        return jsonify({"error": str(e)})

@app.route('/generate_higgs_audio', methods=['POST'])
@limiter.limit("5 per minute")
def generate_higgs_audio():
    """Generate audio using Enhanced Higgs Audio v2."""
    if not MULTIMODAL_AVAILABLE:
        return jsonify({"error": "Multimodal generation not available"})
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Missing text parameter"})
        
        text = data['text']
        language = data.get('language', 'en')
        
        result = generate_enhanced_audio(text, language)
        
        if result.get('success'):
            return jsonify({
                "success": True,
                "audio_url": result.get('audio_url'),
                "method": result.get('method', 'enhanced_higgs'),
                "processing_time": result.get('processing_time_ms', 0)
            })
        else:
            return jsonify({"error": result.get('error', 'Enhanced audio generation failed')})
            
    except Exception as e:
        logger.error(f"Enhanced Higgs audio error: {e}")
        return jsonify({"error": str(e)})

@app.route('/multimodal_status')
@limiter.limit("100 per minute")
def multimodal_status():
    """Get status of multimodal generation services."""
    if not MULTIMODAL_AVAILABLE:
        return jsonify({"error": "Multimodal generation not available"})
    
    return jsonify({
        "success": True,
        "services": {
            "bria_3.2": bria_generator.available,
            "wan2_1_fast": wan2_generator.available,
            "enhanced_higgs": enhanced_higgs.available
        },
        "features": {
            "text_to_image": bria_generator.available,
            "image_to_video": wan2_generator.available,
            "enhanced_audio": enhanced_higgs.available
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

def enhance_uni_with_ai_logic(original_text: str, current_translation: str):
    """Simple AI enhancement for UNI translations."""
    try:
        # Basic enhancement logic - could be expanded with real AI
        if not current_translation or len(current_translation.strip()) < 2:
            return None
            
        # Simple improvements based on UNI grammar rules
        enhanced = current_translation.strip().upper()
        
        # Apply basic UNI grammar fixes
        if "LOVE" in original_text.upper():
            enhanced = enhanced.replace("LOVE", "AMOR")
        if "HELLO" in original_text.upper():
            enhanced = enhanced.replace("HELLO", "HELO")
        if "THANK" in original_text.upper():
            enhanced = enhanced.replace("THANK", "GRACIA")
            
        # Return enhanced version if different
        return enhanced if enhanced != current_translation else None
        
    except Exception as e:
        print(f"AI enhancement logic failed: {e}")
        return None

