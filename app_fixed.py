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
import requests
from flask import Flask, render_template, request, jsonify, session, Response
from flask_wtf.csrf import CSRFProtect
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

# CSRF Protection
csrf = CSRFProtect(app)

def get_flag_emoji(lang_code):
    """Get flag emoji for language code."""
    flags = {
        'en': '🇺🇸', 'es': '🇪🇸', 'fr': '🇫🇷', 'de': '🇩🇪',
        'it': '🇮🇹', 'pt': '🇵🇹', 'ru': '🇷🇺', 'zh-CN': '🇨🇳',
        'ja': '🇯🇵', 'ko': '🇰🇷', 'ar': '🇸🇦', 'hi': '🇮🇳'
    }
    return flags.get(lang_code, '🌍')

def detect_language(text):
    """Detect language of text."""
    try:
        detected = detect(text)
        if detected == 'zh-cn':
            detected = 'zh-CN'
        return detected
    except:
        return 'en'

def translate_with_grok(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text using XAI Grok with fallbacks."""
    try:
        # Try XAI Grok first
        if os.environ.get("XAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=os.environ.get("XAI_API_KEY")
            )
            
            prompt = f"Translate this {LANGUAGE_NAMES.get(source_lang, source_lang)} text to {LANGUAGE_NAMES.get(target_lang, target_lang)}: {text}"
            
            response = client.chat.completions.create(
                model="grok-2-1212",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            
            return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Grok translation failed: {e}")
    
    # Fallback to Claude
    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            prompt = f"Translate this {LANGUAGE_NAMES.get(source_lang, source_lang)} text to {LANGUAGE_NAMES.get(target_lang, target_lang)}: {text}"
            
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Claude translation failed: {e}")
    
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
        target_languages = data.get('target_languages', ['en', 'es', 'fr'])
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if len(text) > 5000:
            return jsonify({'error': 'Text too long (max 5000 characters)'}), 400
        
        # Detect source language
        source_language = detect_language(text)
        
        # Translate to target languages
        translations = []
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
        
        # ElevenLabs TTS
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
        
        return jsonify({'success': False, 'error': 'Audio generation failed'}), 500
        
    except Exception as e:
        logger.error(f"Audio generation error: {e}")
        return jsonify({'success': False, 'error': 'Audio generation failed'}), 500

@app.route('/voice_transcribe', methods=['POST'])
@limiter.limit("10 per minute")
def voice_transcribe():
    """Transcribe voice using ElevenLabs."""
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
        
        # ElevenLabs Speech-to-Text
        if os.environ.get("ELEVENLABS_API_KEY"):
            try:
                url = "https://api.elevenlabs.io/v1/speech-to-text"
                
                headers = {
                    "xi-api-key": os.environ.get("ELEVENLABS_API_KEY")
                }
                
                with open(temp_path, 'rb') as f:
                    files = {"audio": f}
                    response = requests.post(url, headers=headers, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get('text', '').strip()
                    
                    if text:
                        detected_lang = detect_language(text)
                        
                        # Clean up
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                        return jsonify({
                            'success': True,
                            'text': text,
                            'language': detected_lang,
                            'model': 'ElevenLabs AI'
                        })
                else:
                    logger.error(f"ElevenLabs STT failed: {response.status_code}")
            except Exception as e:
                logger.error(f"ElevenLabs STT error: {e}")
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({'success': False, 'error': 'Transcription failed'}), 500
    
    except Exception as e:
        logger.error(f"Voice transcription error: {e}")
        return jsonify({'success': False, 'error': 'Transcription failed'}), 500

@app.route('/audio/<filename>')
def serve_audio(filename):
    """Serve generated audio files."""
    try:
        from flask import send_from_directory
        return send_from_directory('audio', filename)
    except FileNotFoundError:
        return "Audio file not found", 404

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': time.time()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)