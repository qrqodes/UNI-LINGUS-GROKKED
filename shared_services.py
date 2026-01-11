"""
Shared Services Module
Unified translation, TTS, and AI services for both Flask web app and Telegram bot
Ensures consistent quality across all platforms

FREE/OPEN-SOURCE STACK:
- Translation: Google Translate → MADLAD-400 → Aya 23 → Groq
- AI Chat: Groq → Cerebras → Mistral (NO Claude)
- TTS: ElevenLabs → Piper TTS → gTTS
- Language Detection: langdetect → lingua
"""

import os
import logging
import time
import tempfile
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# LANGUAGE CONFIGURATION
# ============================================================================

DEFAULT_LANGUAGES = ['en', 'es', 'pt', 'it', 'fr', 'zh-CN']
ALL_LANGUAGE_CODES = ['en', 'es', 'pt', 'it', 'fr', 'ru', 'de', 'zh-CN', 'ja', 'ko', 'ar', 'uni']
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'pt': 'Portuguese', 
    'it': 'Italian', 'fr': 'French', 'ru': 'Russian', 
    'zh-CN': 'Chinese', 'de': 'German', 'ja': 'Japanese',
    'ko': 'Korean', 'ar': 'Arabic', 'uni': 'UNI'
}
ALL_LANGUAGES = LANGUAGE_NAMES
FLAG_EMOJIS = {
    'en': '🇺🇸', 'es': '🇪🇸', 'pt': '🇵🇹', 'it': '🇮🇹',
    'fr': '🇫🇷', 'ru': '🇷🇺', 'zh-CN': '🇨🇳', 'de': '🇩🇪',
    'ja': '🇯🇵', 'ko': '🇰🇷', 'ar': '🇸🇦', 'uni': '🏳️'
}

# MADLAD-400 language codes mapping
MADLAD_LANG_CODES = {
    'en': 'en', 'es': 'es', 'pt': 'pt', 'it': 'it',
    'fr': 'fr', 'ru': 'ru', 'zh-CN': 'zh', 'de': 'de',
    'ja': 'ja', 'ko': 'ko', 'ar': 'ar'
}

def get_flag_emoji(lang_code: str) -> str:
    """Get flag emoji for language code."""
    return FLAG_EMOJIS.get(lang_code, '🌐')

# ============================================================================
# LANGUAGE DETECTION - langdetect + lingua (FREE)
# ============================================================================

def detect_language(text: str) -> str:
    """
    Detect language using free libraries:
    langdetect (fast) → lingua (accurate fallback)
    """
    if not text or not text.strip():
        return 'en'
    
    # Try langdetect first (faster)
    try:
        from langdetect import detect
        detected = detect(text)
        # Map to our language codes
        if detected == 'zh-cn' or detected == 'zh-tw':
            return 'zh-CN'
        return detected
    except Exception as e:
        logger.debug(f"langdetect failed: {e}")
    
    # Fallback to lingua (more accurate)
    try:
        from lingua import Language, LanguageDetectorBuilder
        detector = LanguageDetectorBuilder.from_all_languages().build()
        detected = detector.detect_language_of(text)
        if detected:
            lang_map = {
                Language.ENGLISH: 'en', Language.SPANISH: 'es',
                Language.PORTUGUESE: 'pt', Language.ITALIAN: 'it',
                Language.FRENCH: 'fr', Language.RUSSIAN: 'ru',
                Language.CHINESE: 'zh-CN', Language.GERMAN: 'de',
                Language.JAPANESE: 'ja', Language.KOREAN: 'ko',
                Language.ARABIC: 'ar'
            }
            return lang_map.get(detected, 'en')
    except Exception as e:
        logger.debug(f"lingua failed: {e}")
    
    return 'en'

# ============================================================================
# TRANSLATION SERVICE - Google → MADLAD-400 → Aya → Groq (ALL FREE)
# ============================================================================

import re

def normalize_punctuation_spacing(text: str) -> str:
    """
    Fix missing spaces after punctuation marks.
    Example: "¿Hola?¿Cómo" -> "¿Hola? ¿Cómo"
    """
    if not text:
        return text
    # Add space after sentence-ending punctuation if followed by a letter or opening punctuation
    text = re.sub(r'([.!?。！？])([^\s\d.!?。！？])', r'\1 \2', text)
    # Add space after closing punctuation followed by opening punctuation
    text = re.sub(r'(["\'\)\]])([¿¡\(\["\'])', r'\1 \2', text)
    return text

def translate_text(text: str, target_lang: str, source_lang: str = 'auto') -> str:
    """
    Translate text using free cascade (FASTEST FIRST):
    Groq (fastest AI) → Google Translate → MADLAD-400 → Aya 23 → Ollama
    """
    if not text or not text.strip():
        return ""
    
    # Handle UNI language translations
    if target_lang == 'uni':
        return translate_to_uni(text)
    if source_lang == 'uni':
        return translate_from_uni(text)
    
    # Try Groq FIRST (fastest - 14,400 requests/day free, ~200ms response)
    result = translate_with_groq(text, target_lang)
    if result:
        logger.info(f"Groq translation (fastest): {text[:30]}... -> {target_lang}")
        return normalize_punctuation_spacing(result)
    
    # Try Google Translate (free, usually reliable)
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, src=source_lang if source_lang != 'auto' else 'auto', dest=target_lang)
        if result and result.text and result.text != text:
            logger.info(f"Google Translate success: {text[:30]}... -> {target_lang}")
            return normalize_punctuation_spacing(result.text)
    except Exception as e:
        logger.warning(f"Google Translate failed: {e}")
    
    # Try MADLAD-400 via HuggingFace API (free tier, may have cold start)
    result = translate_with_madlad(text, target_lang)
    if result:
        return normalize_punctuation_spacing(result)
    
    # Try Aya 23 via HuggingFace API (free tier)
    result = translate_with_aya(text, target_lang)
    if result:
        return normalize_punctuation_spacing(result)
    
    # Try Ollama offline (no API needed)
    try:
        from ollama_translation import translate_with_ollama
        result = translate_with_ollama(text, source_lang, target_lang)
        if result and result != text:
            logger.info(f"Ollama offline translation success")
            return normalize_punctuation_spacing(result)
    except Exception as e:
        logger.warning(f"Ollama failed: {e}")
    
    return text

def translate_with_madlad(text: str, target_lang: str) -> Optional[str]:
    """Translate using MADLAD-400 via HuggingFace free API."""
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('hugg')
    
    try:
        import requests
        
        # Use MADLAD-400 3B model
        API_URL = "https://api-inference.huggingface.co/models/google/madlad400-3b-mt"
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        
        # MADLAD uses <2xx> prefix for target language
        madlad_code = MADLAD_LANG_CODES.get(target_lang, target_lang.split('-')[0])
        payload = {"inputs": f"<2{madlad_code}> {text}"}
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                translated = result[0].get('generated_text', '') or result[0].get('translation_text', '')
                if translated and translated != text:
                    logger.info(f"MADLAD-400 translation success")
                    return translated.strip()
        else:
            logger.warning(f"MADLAD-400 returned {response.status_code}")
            
    except Exception as e:
        logger.warning(f"MADLAD-400 failed: {e}")
    
    return None

def translate_with_aya(text: str, target_lang: str) -> Optional[str]:
    """Translate using Aya 23-8B via HuggingFace free API."""
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('hugg')
    
    try:
        import requests
        
        API_URL = "https://api-inference.huggingface.co/models/CohereForAI/aya-expanse-8b"
        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        
        target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        prompt = f"Translate to {target_name}: {text}"
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 500}}
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                translated = result[0].get('generated_text', '')
                # Remove the prompt from the response
                if translated and prompt in translated:
                    translated = translated.replace(prompt, '').strip()
                if translated and translated != text:
                    logger.info(f"Aya translation success")
                    return translated.strip()
                    
    except Exception as e:
        logger.warning(f"Aya failed: {e}")
    
    return None

def translate_with_groq(text: str, target_lang: str) -> Optional[str]:
    """Translate using Groq free tier (14,400 requests/day)."""
    groq_key = os.environ.get('GROQ_API_KEY')
    if not groq_key:
        return None
    
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": f"Translate to {LANGUAGE_NAMES.get(target_lang, target_lang)}. Return ONLY the translation:\n\n{text}"
            }],
            max_tokens=1024
        )
        result = response.choices[0].message.content.strip()
        if result:
            logger.info(f"Groq translation success")
            return result
    except Exception as e:
        logger.warning(f"Groq translation failed: {e}")
    
    return None

def translate_to_uni(text: str) -> str:
    """Translate to UNI using advanced grammar-aware translator."""
    try:
        from uni_advanced_translator import translate_to_uni_advanced
        return translate_to_uni_advanced(text)
    except Exception as e:
        logger.error(f"UNI translation error: {e}")
        return text.upper()

def translate_from_uni(text: str) -> str:
    """Translate from UNI to English."""
    try:
        from uni_advanced_translator import translate_from_uni as uni_reverse
        return uni_reverse(text)
    except Exception as e:
        logger.error(f"UNI reverse translation error: {e}")
        return text.lower()

def translate_to_all_languages(text: str, languages: List[str] = None, source_lang: str = 'auto') -> Dict[str, Any]:
    """
    Translate text to multiple languages - returns dict with translations.
    Language order follows ALL_LANGUAGE_CODES.
    """
    if languages is None:
        languages = DEFAULT_LANGUAGES
    
    # Order languages according to ALL_LANGUAGE_CODES order
    ordered_languages = []
    for lang in ALL_LANGUAGE_CODES:
        if lang in languages:
            ordered_languages.append(lang)
    
    translations = {}
    for lang in ordered_languages:
        try:
            translation = translate_text(text, lang, source_lang)
            translations[lang] = {
                'text': translation,
                'name': LANGUAGE_NAMES.get(lang, lang),
                'flag': get_flag_emoji(lang)
            }
            
            # Add transcriptions for specific languages
            if lang == 'ru':
                try:
                    from transcription import transliterate_russian
                    translations[lang]['latin'] = transliterate_russian(translation)
                except:
                    pass
            elif lang == 'zh-CN':
                try:
                    from transcription import get_pinyin
                    translations[lang]['pinyin'] = get_pinyin(translation)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Translation to {lang} failed: {e}")
            translations[lang] = {
                'text': text,
                'name': LANGUAGE_NAMES.get(lang, lang),
                'flag': get_flag_emoji(lang),
                'error': str(e)
            }
    
    return translations

# ============================================================================
# TEXT-TO-SPEECH SERVICE - ElevenLabs → Piper TTS → gTTS
# ============================================================================

ELEVENLABS_VOICES = {
    'en': '21m00Tcm4TlvDq8ikWAM',
    'es': 'JBFqnCBSD6RMkjVDRZzb',
    'pt': 'TxGEqnHWrfWFTfGW9XjX',
    'it': 'IKne3meq5aSn9XLyUdCD',
    'fr': 'pNInz6obpgDQGcFmaJgB',
    'ru': 'EXAVITQu4vr4xnSDxMaL',
    'zh-CN': 'XrExE9yKIg1WjnnlVkGX',
    'de': '21m00Tcm4TlvDq8ikWAM',
    'ja': '21m00Tcm4TlvDq8ikWAM',
    'ko': '21m00Tcm4TlvDq8ikWAM',
    'ar': '21m00Tcm4TlvDq8ikWAM',
    'uni': '21m00Tcm4TlvDq8ikWAM',
}

GTTS_LANG_MAP = {
    'en': 'en', 'es': 'es', 'pt': 'pt', 'it': 'it',
    'fr': 'fr', 'ru': 'ru', 'zh-CN': 'zh-CN', 'de': 'de',
    'ja': 'ja', 'ko': 'ko', 'ar': 'ar', 'uni': 'en'
}

PIPER_VOICES = {
    'en': 'en_US-lessac-medium',
    'es': 'es_ES-davefx-medium',
    'pt': 'pt_BR-faber-medium',
    'it': 'it_IT-riccardo-x_low',
    'fr': 'fr_FR-upmc-medium',
    'ru': 'ru_RU-ruslan-medium',
    'de': 'de_DE-thorsten-medium',
    'zh-CN': 'zh_CN-huayan-medium',
}

def generate_audio(text: str, language: str = 'en') -> Tuple[Optional[str], str]:
    """
    Generate audio using FREE cascade:
    Piper TTS (free, good) → gTTS (free, basic)
    
    Returns: (file_path, provider_used)
    """
    if not text or not text.strip():
        return None, 'none'
    
    # Try Piper TTS first (free, good quality)
    audio_path = generate_audio_piper(text, language)
    if audio_path:
        return audio_path, 'piper'
    
    # Fallback to Google gTTS (free, basic)
    audio_path = generate_audio_gtts(text, language)
    if audio_path:
        return audio_path, 'gtts'
    
    return None, 'failed'

def generate_audio_elevenlabs(text: str, language: str) -> Optional[str]:
    """Generate audio using ElevenLabs with native language voices."""
    api_key = os.environ.get('ELEVENLABS_API_KEY')
    if not api_key:
        logger.debug("ElevenLabs API key not available")
        return None
    
    try:
        import requests
        
        voice_id = ELEVENLABS_VOICES.get(language, ELEVENLABS_VOICES['en'])
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": api_key
        }
        
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.5,
                "use_speaker_boost": True
            }
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            os.makedirs('audio', exist_ok=True)
            filename = f"elevenlabs_{language}_{int(time.time())}.mp3"
            filepath = os.path.join('audio', filename)
            
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"ElevenLabs TTS generated: {filepath}")
            return filepath
        else:
            logger.warning(f"ElevenLabs returned {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"ElevenLabs error: {e}")
        return None

def generate_audio_piper(text: str, language: str) -> Optional[str]:
    """Generate audio using Piper TTS (free, local)."""
    try:
        import subprocess
        import wave
        
        voice = PIPER_VOICES.get(language, PIPER_VOICES.get('en'))
        if not voice:
            return None
        
        os.makedirs('audio', exist_ok=True)
        filename = f"piper_{language}_{int(time.time())}.wav"
        filepath = os.path.join('audio', filename)
        
        # Use piper command line
        process = subprocess.run(
            ['piper', '--model', voice, '--output_file', filepath],
            input=text.encode('utf-8'),
            capture_output=True,
            timeout=30
        )
        
        if process.returncode == 0 and os.path.exists(filepath):
            logger.info(f"Piper TTS generated: {filepath}")
            return filepath
        else:
            logger.warning(f"Piper TTS failed: {process.stderr.decode()}")
            return None
            
    except FileNotFoundError:
        logger.debug("Piper TTS not available (command not found)")
        return None
    except Exception as e:
        logger.warning(f"Piper TTS error: {e}")
        return None

def generate_audio_gtts(text: str, language: str) -> Optional[str]:
    """Generate audio using Google Text-to-Speech (free)."""
    try:
        from gtts import gTTS
        
        gtts_lang = GTTS_LANG_MAP.get(language, 'en')
        tts = gTTS(text=text, lang=gtts_lang, slow=False)
        
        os.makedirs('audio', exist_ok=True)
        filename = f"gtts_{language}_{int(time.time())}.mp3"
        filepath = os.path.join('audio', filename)
        
        tts.save(filepath)
        logger.info(f"gTTS generated: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"gTTS error: {e}")
        return None

# ============================================================================
# AI CHAT SERVICE - Groq → Cerebras → Mistral (ALL FREE, NO CLAUDE)
# ============================================================================

def chat_with_ai(message: str, context: str = None) -> Optional[str]:
    """
    Chat with AI using FREE providers cascade:
    Groq (14,400 req/day) → Cerebras (1M tokens/day) → Mistral (1B tokens/month)
    
    NO Claude - completely removed for cost savings.
    """
    system_prompt = """You are UNI LINGUS, a helpful multilingual AI assistant specializing in language learning and translation. 
You help users learn languages, understand grammar, and provide cultural insights.
Be friendly, educational, and encouraging. Answer in the user's language when possible."""
    
    if context:
        system_prompt += f"\n\nContext: {context}"
    
    # Try Groq first (fastest, 14,400 requests/day free)
    result = chat_with_groq(message, system_prompt)
    if result:
        return result
    
    # Try Cerebras (1M tokens/day free)
    result = chat_with_cerebras(message, system_prompt)
    if result:
        return result
    
    # Try Mistral (1B tokens/month free)
    result = chat_with_mistral(message, system_prompt)
    if result:
        return result
    
    logger.error("All AI chat providers failed")
    return None

def chat_with_groq(message: str, system_prompt: str) -> Optional[str]:
    """Chat using Groq free tier."""
    groq_key = os.environ.get('GROQ_API_KEY')
    if not groq_key:
        return None
    
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=2048
        )
        result = response.choices[0].message.content
        logger.info("Groq chat success")
        return result
    except Exception as e:
        logger.warning(f"Groq chat failed: {e}")
        return None

def chat_with_cerebras(message: str, system_prompt: str) -> Optional[str]:
    """Chat using Cerebras free tier (1M tokens/day)."""
    cerebras_key = os.environ.get('CEREBRAS_API_KEY')
    if not cerebras_key:
        return None
    
    try:
        from cerebras.cloud.sdk import Cerebras
        client = Cerebras(api_key=cerebras_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=2048
        )
        result = response.choices[0].message.content
        logger.info("Cerebras chat success")
        return result
    except Exception as e:
        logger.warning(f"Cerebras chat failed: {e}")
        return None

def chat_with_mistral(message: str, system_prompt: str) -> Optional[str]:
    """Chat using Mistral free tier (1B tokens/month)."""
    mistral_key = os.environ.get('MISTRAL_API_KEY')
    if not mistral_key:
        return None
    
    try:
        from mistralai import Mistral
        client = Mistral(api_key=mistral_key)
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ],
            max_tokens=2048
        )
        result = response.choices[0].message.content
        logger.info("Mistral chat success")
        return result
    except Exception as e:
        logger.warning(f"Mistral chat failed: {e}")
        return None

def is_ai_available() -> bool:
    """Check if any AI chat service is available."""
    return bool(
        os.environ.get('GROQ_API_KEY') or 
        os.environ.get('CEREBRAS_API_KEY') or 
        os.environ.get('MISTRAL_API_KEY')
    )

# ============================================================================
# VOCABULARY SERVICE
# ============================================================================

def get_vocabulary_by_level(level: str = 'beginner') -> List[Dict]:
    """Get vocabulary words by difficulty level."""
    try:
        from vocabulary_data import get_vocab_by_level
        return get_vocab_by_level(level)
    except Exception as e:
        logger.error(f"Vocabulary error: {e}")
        return [
            {'word': 'hello', 'translation': 'hola', 'language': 'es'},
            {'word': 'thank you', 'translation': 'gracias', 'language': 'es'},
            {'word': 'goodbye', 'translation': 'adiós', 'language': 'es'}
        ]

def get_learning_example(word: str, language: str) -> Optional[str]:
    """Get a learning example for a word using AI."""
    return chat_with_ai(f"Create a simple sentence example using the word '{word}' in {LANGUAGE_NAMES.get(language, language)}. Include pronunciation tips.")

# ============================================================================
# SERVICE STATUS
# ============================================================================

def get_service_status() -> Dict[str, bool]:
    """Get status of all services."""
    return {
        'google_translate': True,
        'madlad': True,  # HuggingFace free API
        'aya': True,  # HuggingFace free API
        'elevenlabs': bool(os.environ.get('ELEVENLABS_API_KEY')),
        'piper': True,  # Local, always available
        'gtts': True,
        'groq': bool(os.environ.get('GROQ_API_KEY')),
        'cerebras': bool(os.environ.get('CEREBRAS_API_KEY')),
        'mistral': bool(os.environ.get('MISTRAL_API_KEY')),
    }
