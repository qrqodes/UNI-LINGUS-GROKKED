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
from flask import Flask, render_template, request, jsonify, session, Response
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from langdetect import detect, LangDetectException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Language constants
ALL_LANGUAGE_CODES = ['en', 'es', 'pt', 'it', 'fr', 'ru', 'zh-CN']
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh-CN': 'Chinese',
    'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi'
}

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", secrets.token_hex(16))

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# CSRF Protection - disabled for API endpoints
# csrf = CSRFProtect(app)

def get_flag_emoji(lang_code):
    """Get flag emoji for language code."""
    flags = {
        'en': '🇺🇸', 'es': '🇪🇸', 'fr': '🇫🇷', 'de': '🇩🇪',
        'it': '🇮🇹', 'pt': '🇵🇹', 'ru': '🇷🇺', 'zh-CN': '🇨🇳',
        'ja': '🇯🇵', 'ko': '🇰🇷', 'ar': '🇸🇦', 'hi': '🇮🇳'
    }
    return flags.get(lang_code, '🌍')

def detect_language(text):
    """Detect language of text using Grok first, then fallback to langdetect."""
    # First try Grok for better accuracy with Russian and other languages
    try:
        if os.environ.get("XAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="grok-2-1212",
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
            
            response_text = message.content[0].text.strip()
            
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
                        
                        # Add transcriptions for Russian and Chinese
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

def translate_with_grok(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using optimized hierarchy with fast Google Translate first."""
    
    # Use Google Translate first for speed (it's fastest and most reliable)
    try:
        from googletrans import Translator
        translator = Translator()
        result = translator.translate(text, src=source_lang, dest=target_lang)
        if result and result.text and result.text != text:
            logger.info("Translation successful with Google Translate (fast)")
            translation = result.text
            
            # Add transcriptions for Russian and Chinese
            if target_lang == 'ru':
                return f"{translation}\n[{transliterate_russian(translation)}]"
            elif target_lang == 'zh-CN':
                return f"{translation}\n[{get_pinyin(translation)}]"
            
            return translation
    except Exception as e:
        logger.warning(f"Google Translate failed: {e}")
    
    # Fallback to Claude only if Google fails
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
            
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            translated = message.content[0].text.strip()
            if translated and translated != text:
                logger.info("Translation successful with Claude")
                return translated
    except Exception as e:
        logger.warning(f"Claude translation failed: {e}")
    
    try:
        # Final fallback to ElevenLabs (if available)
        if os.environ.get("ELEVENLABS_API_KEY"):
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
    """Add security headers to all responses."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

@app.route('/')
def index():
    """Render the main page of the translation app."""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
@limiter.limit("30 per minute")
def translate():
    """API endpoint to translate text."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
        
        text = data['text'].strip()
        target_languages = data.get('languages', ['en', 'es', 'pt', 'it', 'fr', 'ru', 'zh-CN'])
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if len(text) > 5000:
            return jsonify({'error': 'Text too long (max 5000 characters)'}), 400
        
        # Detect source language
        source_language = detect_language(text)
        
        # Fast batch translation with Google Translate for speed
        translations = []
        
        try:
            from googletrans import Translator
            translator = Translator()
            
            for target_lang in target_languages:
                if target_lang != source_language:
                    try:
                        # Use Google Translate for speed
                        result = translator.translate(text, src=source_language, dest=target_lang)
                        if result and result.text and result.text != text:
                            translated_text = result.text
                            
                            # Add transcriptions for Russian and Chinese
                            if target_lang == 'ru':
                                translated_text += f"\n[{transliterate_russian(result.text)}]"
                            elif target_lang == 'zh-CN':
                                translated_text += f"\n[{get_pinyin(result.text)}]"
                            
                            translations.append({
                                'code': target_lang,
                                'name': LANGUAGE_NAMES.get(target_lang, target_lang),
                                'text': translated_text,
                                'flag': get_flag_emoji(target_lang)
                            })
                        else:
                            # Fallback to other services if Google fails
                            fallback_text = translate_with_grok(text, source_language, target_lang)
                            translations.append({
                                'code': target_lang,
                                'name': LANGUAGE_NAMES.get(target_lang, target_lang),
                                'text': fallback_text,
                                'flag': get_flag_emoji(target_lang)
                            })
                    except Exception as e:
                        logger.warning(f"Translation failed for {target_lang}: {e}")
                        translations.append({
                            'code': target_lang,
                            'name': LANGUAGE_NAMES.get(target_lang, target_lang),
                            'text': "Translation failed",
                            'flag': get_flag_emoji(target_lang)
                        })
                        
        except Exception as e:
            logger.warning(f"Google Translate not available: {e}")
            # Fallback to individual translations
            for target_lang in target_languages:
                if target_lang != source_language:
                    translated_text = translate_with_grok(text, source_language, target_lang)
                    translations.append({
                        'code': target_lang,
                        'name': LANGUAGE_NAMES.get(target_lang, target_lang),
                        'text': translated_text,
                        'flag': get_flag_emoji(target_lang)
                    })
        
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
    """Generate audio pronunciation using ElevenLabs."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'Missing text parameter'}), 400
        
        text = data['text']
        language = data.get('language', 'en')
        
        if len(text) > 1000:
            return jsonify({'success': False, 'error': 'Text too long (max 1000 characters)'}), 400
        
        # Try ElevenLabs first (primary)
        if os.environ.get("ELEVENLABS_API_KEY"):
            try:
                voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default English voice
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
                
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": os.environ.get("ELEVENLABS_API_KEY")
                }
                
                data_payload = {
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.8
                    }
                }
                
                response = requests.post(url, json=data_payload, headers=headers)
                
                if response.status_code == 200:
                    filename = f"elevenlabs_{int(time.time())}_{language}.mp3"
                    filepath = os.path.join('audio', filename)
                    
                    os.makedirs('audio', exist_ok=True)
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    return jsonify({
                        'success': True,
                        'audio_url': f'/audio/{filename}',
                        'filename': filename,
                        'provider': 'ElevenLabs AI'
                    })
                else:
                    logger.error(f"ElevenLabs TTS failed: {response.status_code}")
            except Exception as e:
                logger.error(f"ElevenLabs TTS error: {e}")
        
        # Fallback to gTTS (Google Text-to-Speech)
        try:
            from gtts import gTTS
            
            # Map language codes to gTTS compatible codes
            lang_map = {
                'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de',
                'it': 'it', 'pt': 'pt', 'ru': 'ru', 'zh-CN': 'zh',
                'ja': 'ja', 'ko': 'ko', 'ar': 'ar', 'hi': 'hi'
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
        
        # Save temporary file
        timestamp = int(time.time())
        temp_filename = f"temp_audio_{timestamp}.webm"
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

        # Use new transcription system: ElevenLabs -> Claude -> Grok
        from new_voice_transcription import transcribe_audio_new_system
        
        transcription, detected_lang = transcribe_audio_new_system(wav_path)
        
        # Method 1: Try ElevenLabs Speech-to-Text first
        try:
            import requests
            import base64
            
            # Check if wav file exists and has content
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                logger.error(f"WAV file does not exist or is empty: {wav_path}")
                raise Exception("Invalid audio file")
            
            elevenlabs_api_key = os.environ.get('ELEVENLABS_API_KEY')
            if elevenlabs_api_key:
                logger.info("Attempting transcription with ElevenLabs")
                
                # Read audio file
                with open(wav_path, 'rb') as audio_file:
                    audio_data = audio_file.read()
                
                headers = {
                    'xi-api-key': elevenlabs_api_key
                }
                
                files = {
                    'audio': ('audio.wav', audio_data, 'audio/wav')
                }
                data = {
                    'model_id': 'eleven_multilingual_sts_v2'
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
                        detected_lang = detect_language(transcription) or 'en'
                        logger.info(f"ElevenLabs transcription successful: {transcription}")
                    else:
                        logger.warning("ElevenLabs returned empty transcription")
                else:
                    logger.warning(f"ElevenLabs failed: {response.status_code} - {response.text}")
            else:
                logger.warning("ElevenLabs API key not available")
                
        except Exception as el_error:
            logger.error(f"ElevenLabs transcription failed: {el_error}")
        
        # Method 2: If ElevenLabs failed, try Claude (Anthropic) as fallback
        if not transcription:
            try:
                hf_token = os.environ.get("HUGGINGFACE_API_TOKEN")
                pyannote_token = os.environ.get("PYANNOTE_API_TOKEN") 
                
                if hf_token or pyannote_token:
                    # Use OpenAI Whisper via HuggingFace as it's more reliable for transcription
                    api_url = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
                    headers = {"Authorization": f"Bearer {hf_token or pyannote_token}"}
                    
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
                            detected_lang = detect_language(transcription)
                            logger.info(f"Whisper transcription successful: {transcription}")
                        elif isinstance(result, list) and len(result) > 0 and 'text' in result[0]:
                            transcription = result[0]['text'].strip()
                            detected_lang = detect_language(transcription)
                            logger.info(f"Whisper transcription successful: {transcription}")
                    elif response.status_code == 503:
                        logger.warning("Whisper model is loading, trying again in a moment")
                    else:
                        logger.error(f"Whisper API failed: {response.status_code} - {response.text}")
                            
            except Exception as whisper_error:
                logger.error(f"Whisper fallback failed: {whisper_error}")
        
        # Check if we got transcription from any method
        if transcription:
            # Get target languages for translation (same parameter name as text translation)
            target_languages_json = request.form.get('languages', '[]')
            try:
                target_languages = json.loads(target_languages_json) if target_languages_json else []
            except:
                target_languages = []
            
            # If no target languages specified, use same default as text translation
            if not target_languages:
                target_languages = ['en', 'es', 'pt', 'it', 'fr', 'ru', 'zh-CN']
            
            # Generate translations
            translations = []
            for lang_code in target_languages:
                try:
                    translation = translate_with_grok(transcription, detected_lang, lang_code)
                    if translation and translation != transcription:
                        flag_emoji = get_flag_emoji(lang_code)
                        lang_name = {
                            'es': 'Spanish', 'fr': 'French', 'de': 'German', 
                            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
                            'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean',
                            'ar': 'Arabic', 'hi': 'Hindi', 'tr': 'Turkish'
                        }.get(lang_code, lang_code.upper())
                        
                        translations.append({
                            'code': lang_code,
                            'name': lang_name,
                            'text': translation,
                            'flag': flag_emoji
                        })
                except Exception as e:
                    logger.error(f"Translation failed for {lang_code}: {e}")
                    continue
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if wav_path != temp_path and os.path.exists(wav_path):
                os.remove(wav_path)
            
            return jsonify({
                'success': True,
                'text': transcription,
                'language': detected_lang,
                'model': 'Speech Recognition',
                'translations': translations
            })
        
        # If no transcription was successful, clean up and return error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_path != temp_path and os.path.exists(wav_path):
            os.remove(wav_path)
            
        return jsonify({'success': False, 'error': 'Could not understand the audio. Please try speaking more clearly.'}), 500

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
    """Serve generated audio files."""
    try:
        from flask import send_from_directory
        return send_from_directory('audio', filename)
    except FileNotFoundError:
        return "Audio file not found", 404

@app.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def chat_with_ai():
    """Chat with AI using available models."""
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        message = data['message'].strip()
        if not message:
            return jsonify({'success': False, 'error': 'Empty message'}), 400
        
        # Try to get AI response using Grok first for better Russian language support
        response = None
        
        # Try Grok first (primary - best for multilingual including Russian)
        try:
            if os.environ.get("XAI_API_KEY"):
                from openai import OpenAI
                client = OpenAI(base_url="https://api.x.ai/v1", api_key=os.environ.get("XAI_API_KEY"))
                
                response_obj = client.chat.completions.create(
                    model="grok-2-1212",
                    messages=[{"role": "user", "content": message}],
                    max_tokens=1000,
                    temperature=0.7
                )
                response = response_obj.choices[0].message.content
                logger.info("Chat response generated with Grok")
        except Exception as e:
            logger.warning(f"Grok chat failed: {e}")
        
        # Try Claude as fallback
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
                    response = message_obj.content[0].text
                    logger.info("Chat response generated with Claude")
            except Exception as e:
                logger.warning(f"Claude chat failed: {e}")
        
        # Try DeepSeek as third choice
        if not response:
            try:
                from deepseek_integration import chat_with_deepseek, is_deepseek_available
                
                if is_deepseek_available():
                    response = chat_with_deepseek(message)
                    if response:
                        logger.info("Chat response generated with DeepSeek")
            except Exception as e:
                logger.warning(f"DeepSeek chat failed: {e}")
        
        # Final fallback (if needed)
        if not response:
            try:
                import anthropic
                anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
                if anthropic_key:
                    client = anthropic.Anthropic(api_key=anthropic_key)
                    message_obj = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1000,
                        messages=[{"role": "user", "content": message}]
                    )
                    response = message_obj.content[0].text
                    logger.info("Chat response generated with Anthropic Claude")
            except Exception as e:
                logger.warning(f"Anthropic chat failed: {e}")
        
        if response:
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

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)