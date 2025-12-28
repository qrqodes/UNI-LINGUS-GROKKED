"""
Voice recognition using NVIDIA Parakeet TDT model for better multilingual support.
This model is specifically designed for Portuguese, Russian, Chinese and other languages.
"""

import logging
import os
import uuid
import tempfile
import requests
from typing import Optional
from pydub import AudioSegment
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# Language mappings with flags
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese',
    'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi'
}

def get_flag_emoji(lang_code: str) -> str:
    """Get flag emoji for language code"""
    flags = {
        'en': '🇺🇸', 'es': '🇪🇸', 'fr': '🇫🇷', 'de': '🇩🇪',
        'it': '🇮🇹', 'pt': '🇵🇹', 'ru': '🇷🇺', 'zh': '🇨🇳',
        'ja': '🇯🇵', 'ko': '🇰🇷', 'ar': '🇸🇦', 'hi': '🇮🇳'
    }
    return flags.get(lang_code, '🌍')

def transcribe_with_parakeet(audio_path: str) -> Optional[dict]:
    """
    Transcribe audio using NVIDIA Parakeet TDT model via Hugging Face API.
    This model is excellent for multilingual transcription including Portuguese, Russian, Chinese.
    """
    try:
        # Convert to WAV format if needed
        audio = AudioSegment.from_file(audio_path)
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            audio.export(temp_file.name, format="wav", parameters=["-ar", "16000"])
            
            # Read the audio file
            with open(temp_file.name, 'rb') as f:
                audio_data = f.read()
            
            # Clean up temp file
            os.unlink(temp_file.name)
            
            # Use Hugging Face Inference API for NVIDIA Parakeet
            api_url = "https://api-inference.huggingface.co/models/nvidia/parakeet-tdt-0.6b"
            
            headers = {}
            if os.environ.get("HUGGINGFACE_API_TOKEN"):
                headers["Authorization"] = f"Bearer {os.environ.get('HUGGINGFACE_API_TOKEN')}"
            
            # Send request to Hugging Face API
            response = requests.post(
                api_url,
                headers=headers,
                data=audio_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and 'text' in result:
                    transcription = result['text'].strip()
                    if transcription:
                        logger.info(f"Parakeet transcription successful: {transcription[:50]}...")
                        
                        # Detect language using simple heuristics and Google Translate
                        detected_lang = detect_language_from_text(transcription)
                        
                        return {
                            'text': transcription,
                            'language': detected_lang,
                            'confidence': 0.9
                        }
                        
            elif response.status_code == 503:
                logger.warning("Parakeet model is loading, will try fallback")
                return None
            else:
                logger.error(f"Parakeet API error: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"Error with Parakeet transcription: {e}")
        return None

def transcribe_with_enhanced_parakeet(voice_path: str) -> Optional[dict]:
    """
    Enhanced Parakeet transcription with better language detection for Italian/Spanish.
    """
    try:
        # First try with language hints for Italian/Spanish
        for lang_hint in ["it", "es", "fr", "en"]:
            try:
                logger.info(f"Trying enhanced Parakeet with language hint: {lang_hint}")
                result = transcribe_with_parakeet_lang(voice_path, lang_hint)
                if result and result.get('text') and len(result['text'].strip()) > 2:
                    # Verify the language detection
                    detected_lang = enhanced_language_detection(result['text'])
                    if detected_lang in ["it", "es", "fr", "en"]:
                        result['language'] = detected_lang
                        result['method'] = 'enhanced_parakeet'
                        logger.info(f"Enhanced Parakeet detected {detected_lang}: {result['text'][:50]}...")
                        return result
            except Exception as e:
                logger.debug(f"Enhanced Parakeet with {lang_hint} failed: {e}")
                continue
        
        # Fallback to original method
        return transcribe_with_parakeet(voice_path)
        
    except Exception as e:
        logger.error(f"Enhanced Parakeet failed: {e}")
        return None

def enhanced_language_detection(text: str) -> str:
    """
    Enhanced language detection prioritizing Italian and Spanish.
    """
    if not text or len(text.strip()) < 3:
        return 'en'
    
    text_lower = text.lower().strip()
    
    # Italian patterns (highest priority)
    italian_score = 0
    italian_words = ['il', 'la', 'è', 'e', 'di', 'che', 'in', 'un', 'una', 'con', 'per', 'sono', 'ho', 'hai', 'della', 'degli', 'delle', 'questo', 'questa', 'come', 'quando', 'dove']
    italian_endings = ['zione', 'mente', 'ezza', 'ità', 'are', 'ere', 'ire']
    italian_chars = ['ò', 'ù', 'à', 'ì', 'é']
    
    for word in italian_words:
        if f' {word} ' in f' {text_lower} ' or text_lower.startswith(f'{word} ') or text_lower.endswith(f' {word}'):
            italian_score += 3
    
    for ending in italian_endings:
        if ending in text_lower:
            italian_score += 2
    
    for char in italian_chars:
        if char in text_lower:
            italian_score += 3
    
    # Spanish patterns (highest priority)
    spanish_score = 0
    spanish_words = ['el', 'la', 'es', 'y', 'de', 'que', 'en', 'un', 'una', 'con', 'por', 'para', 'soy', 'tengo', 'del', 'los', 'las', 'este', 'esta', 'como', 'cuando', 'donde']
    spanish_endings = ['ción', 'mente', 'dad', 'idad', 'ar', 'er', 'ir']
    spanish_chars = ['ñ', 'ü', '¿', '¡', 'á', 'é', 'í', 'ó', 'ú']
    
    for word in spanish_words:
        if f' {word} ' in f' {text_lower} ' or text_lower.startswith(f'{word} ') or text_lower.endswith(f' {word}'):
            spanish_score += 3
    
    for ending in spanish_endings:
        if ending in text_lower:
            spanish_score += 2
    
    for char in spanish_chars:
        if char in text_lower:
            spanish_score += 3
    
    # Other languages (lower priority)
    other_scores = {}
    other_patterns = {
        'fr': ['le', 'la', 'est', 'et', 'de', 'que', 'dans', 'un', 'une', 'avec', 'pour', 'du', 'les', 'des', 'ce', 'cette'],
        'en': ['the', 'and', 'is', 'are', 'this', 'that', 'with', 'have', 'will', 'from', 'i', 'you', 'we', 'they'],
        'pt': ['o', 'a', 'é', 'e', 'de', 'que', 'em', 'um', 'uma', 'com', 'para', 'do', 'da', 'os', 'as']
    }
    
    for lang, words in other_patterns.items():
        score = 0
        for word in words:
            if f' {word} ' in f' {text_lower} ' or text_lower.startswith(f'{word} ') or text_lower.endswith(f' {word}'):
                score += 1
        other_scores[lang] = score
    
    # Determine the best language
    all_scores = {
        'it': italian_score,
        'es': spanish_score,
        **other_scores
    }
    
    # Filter out zero scores
    non_zero_scores = {k: v for k, v in all_scores.items() if v > 0}
    
    if non_zero_scores:
        best_lang = max(non_zero_scores.items(), key=lambda x: x[1])[0]
        logger.info(f"Enhanced language detection scores: {all_scores}")
        logger.info(f"Selected language: {best_lang}")
        return best_lang
    
    return 'en'  # Default fallback

def transcribe_with_parakeet_lang(voice_path: str, language_hint: str) -> Optional[dict]:
    """
    Transcribe with Parakeet using a specific language hint.
    """
    try:
        from gradio_client import Client
        
        # Initialize Parakeet client
        parakeet_url = "https://nvidia-parakeet-tdt-0-6b-v2.hf.space"
        client = Client(parakeet_url)
        
        logger.info(f"Transcribing with Parakeet, language: {language_hint}")
        
        # Try different API endpoints
        possible_endpoints = ["/transcribe", "/predict", "/process_audio", "/"]
        
        for endpoint in possible_endpoints:
            try:
                result = client.predict(
                    voice_path,
                    language_hint,
                    api_name=endpoint
                )
                
                if result:
                    if isinstance(result, str) and len(result.strip()) > 2:
                        return {
                            'text': result.strip(),
                            'language': language_hint,
                            'confidence': 0.85,
                            'method': 'parakeet_lang_hint'
                        }
                    elif isinstance(result, dict) and result.get('text'):
                        return {
                            'text': result['text'].strip(),
                            'language': result.get('language', language_hint),
                            'confidence': result.get('confidence', 0.85),
                            'method': 'parakeet_lang_hint'
                        }
                        
            except Exception as endpoint_error:
                logger.debug(f"Parakeet endpoint {endpoint} failed: {endpoint_error}")
                continue
        
        return None
        
    except Exception as e:
        logger.error(f"Parakeet language transcription failed: {e}")
        return None

def detect_language_from_text(text: str) -> str:
    """
    Enhanced language detection prioritizing Italian and Spanish detection.
    Cross-checks multiple language indicators before deciding.
    """
    if not text or len(text.strip()) < 3:
        return 'en'
    
    text_lower = text.lower()
    confidence_scores = {}
    
    # Priority language patterns (Italian and Spanish first)
    language_patterns = {
        'it': ['il', 'la', 'è', 'e', 'di', 'che', 'in', 'un', 'una', 'con', 'per', 'del', 'gli', 'delle', 'sono', 'ho', 'hai', 'abbiamo'],
        'es': ['el', 'la', 'es', 'y', 'de', 'que', 'en', 'un', 'una', 'con', 'por', 'para', 'del', 'los', 'las', 'soy', 'tengo', 'tienes'],
        'fr': ['le', 'la', 'est', 'et', 'de', 'que', 'dans', 'un', 'une', 'avec', 'pour', 'du', 'les', 'des', 'je', 'tu', 'nous'],
        'en': ['the', 'and', 'is', 'are', 'this', 'that', 'with', 'have', 'will', 'from', 'i', 'you', 'we'],
        'de': ['der', 'die', 'das', 'und', 'ist', 'in', 'zu', 'ein', 'eine', 'mit', 'für', 'von', 'ich', 'du', 'wir'],
        'pt': ['o', 'a', 'é', 'e', 'de', 'que', 'em', 'um', 'uma', 'com', 'para', 'do', 'da', 'eu', 'você', 'nós'],
        'ru': ['и', 'в', 'на', 'что', 'с', 'по', 'не', 'он', 'она', 'это', 'я', 'ты', 'мы']
    }
    
    # Score each language based on word matches
    for lang, patterns in language_patterns.items():
        matches = sum(1 for pattern in patterns if pattern in text_lower)
        if matches > 0:
            confidence_scores[lang] = matches / len(patterns)
    
    # Check for language-specific characters (higher weight for Italian/Spanish)
    char_indicators = {
        'it': (['ò', 'ù', 'à', 'ì'], 0.8),  # Italian gets higher weight
        'es': (['ñ', 'ü', '¿', '¡'], 0.8),  # Spanish gets higher weight
        'fr': (['é', 'è', 'ê', 'ë', 'ç', 'à', 'ù', 'î', 'ô', 'û'], 0.6),
        'de': (['ä', 'ö', 'ü', 'ß'], 0.6),
        'pt': (['ã', 'õ', 'ç'], 0.4),  # Portuguese gets lower priority
        'ru': (['й', 'ё', 'ъ', 'ь'], 0.6)
    }
    
    for lang, (chars, weight) in char_indicators.items():
        char_matches = sum(1 for char in chars if char in text)
        if char_matches > 0:
            confidence_scores[lang] = confidence_scores.get(lang, 0) + (char_matches * weight)
    
    # Try Google Translate as additional verification (but with lower weight)
    try:
        from deep_translator import GoogleTranslator
        detected = GoogleTranslator.detect(text)
        if detected and detected != 'pt':  # Don't boost Portuguese detection
            if detected == 'zh':
                detected = 'zh-CN'
            confidence_scores[detected] = confidence_scores.get(detected, 0) + 0.2
    except Exception as e:
        logger.debug(f"Google Translate detection failed: {e}")
    
    # Special boost for Italian and Spanish if they have any indicators
    if 'it' in confidence_scores:
        confidence_scores['it'] *= 1.5  # Boost Italian
    if 'es' in confidence_scores:
        confidence_scores['es'] *= 1.5  # Boost Spanish
    
    # Find the language with highest confidence
    if confidence_scores:
        best_lang = max(confidence_scores.items(), key=lambda x: x[1])
        logger.info(f"Language detection scores: {confidence_scores}")
        logger.info(f"Selected language: {best_lang[0]} (confidence: {best_lang[1]:.2f})")
        return best_lang[0]
    
    # Character-based fallback
    if any('\u4e00' <= char <= '\u9fff' for char in text):
        return 'zh-CN'
    elif any('\u0400' <= char <= '\u04ff' for char in text):
        return 'ru'
    
    # Default to English only if no other indicators found
    return 'en'

async def handle_voice_message_parakeet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle voice messages using NVIDIA Parakeet model for better multilingual support.
    """
    if not update.message or not update.message.voice:
        return
    
    chat_id = update.message.chat_id
    voice = update.message.voice
    
    try:
        # Download voice file
        voice_file = await context.bot.get_file(voice.file_id)
        unique_id = uuid.uuid4().hex[:8]
        voice_path = f"audio/voice_{unique_id}_{voice.file_id[-8:]}.ogg"
        await voice_file.download_to_drive(voice_path)
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            "🎙️ Processing voice with NVIDIA Parakeet multilingual model..."
        )
        
        # SMART MODEL ROUTING based on language strengths:
        # Google excels at: Italian, Spanish, French
        # NVIDIA excels at: Chinese, Russian, Portuguese, English
        
        result = None
        
        # STEP 1: Let Grok/Claude help NVIDIA with ALL languages
        try:
            from ai_services_simplified import query_claude
            from xai import query_grok
            
            # First try basic transcription with NVIDIA
            nvidia_result = transcribe_with_enhanced_parakeet(voice_path)
            
            if nvidia_result and nvidia_result.get('text'):
                nvidia_text = nvidia_result['text']
                logger.info(f"🤖 NVIDIA transcribed: {nvidia_text[:50]}...")
                
                # Now let AI improve and verify the transcription
                ai_prompt = f"""
                Analyze this voice transcription and improve it if needed:
                "{nvidia_text}"
                
                Please:
                1. Fix any obvious transcription errors
                2. Detect the most likely language (it, es, fr, en, zh-CN, ru, pt)
                3. Provide the corrected text
                
                Respond in JSON format:
                {{"corrected_text": "improved text", "language": "detected_language", "confidence": 0.95}}
                """
                
                # Try Claude first, then Grok as fallback
                ai_response = query_claude(ai_prompt, max_tokens=200, temperature=0.3)
                if not ai_response:
                    ai_response = query_grok(ai_prompt, max_tokens=200, temperature=0.3)
                
                if ai_response:
                    try:
                        import json
                        ai_data = json.loads(ai_response)
                        result = {
                            'text': ai_data.get('corrected_text', nvidia_text),
                            'language': ai_data.get('language', 'en'),
                            'confidence': ai_data.get('confidence', 0.9),
                            'method': 'nvidia_ai_enhanced'
                        }
                        logger.info(f"🎯 AI enhanced: {result['text'][:50]}...")
                    except:
                        result = nvidia_result
                else:
                    result = nvidia_result
            else:
                # Fallback to Google for IT/ES/FR if NVIDIA fails
                import speech_recognition as sr
                logger.info("🎯 NVIDIA failed, trying Google for IT/ES/FR...")
                r = sr.Recognizer()
                
                with sr.AudioFile(voice_path) as source:
                    audio = r.record(source)
                
                for lang_hint in ["it", "es", "fr"]:
                    try:
                        google_text = r.recognize_google(audio, language=lang_hint)
                        if google_text and len(google_text.strip()) > 2:
                            result = {
                                'text': google_text.strip(),
                                'language': lang_hint,
                                'confidence': 0.95,
                                'method': 'google_fallback'
                            }
                            logger.info(f"✅ Google fallback caught {lang_hint.upper()}: {google_text[:50]}...")
                            break
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.debug(f"AI-enhanced transcription failed: {e}")
            result = transcribe_with_enhanced_parakeet(voice_path)
        
        if not result:
            # Fallback to basic speech recognition with focus on target languages
            result = fallback_speech_recognition(voice_path)
        
        if not result or not result.get('text'):
            await processing_msg.edit_text(
                "❌ Could not transcribe the voice message. Please try:\n"
                "• Speaking more clearly\n"
                "• Reducing background noise\n"
                "• Trying again"
            )
            os.unlink(voice_path)
            return
        
        transcription = result['text']
        detected_language = result.get('language', 'en')
        
        # Get user's selected languages
        from database import db
        user_data = db.get_user(chat_id)
        selected_languages = ['es', 'fr', 'de', 'it', 'pt', 'ru', 'zh-CN', 'ja', 'ko', 'ar', 'hi']
        
        if user_data and user_data.get('selected_languages'):
            try:
                import json
                selected_languages = json.loads(user_data['selected_languages'])
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Translate using enhanced translator
        from enhanced_translator import translate_to_all
        translation_result = translate_to_all(transcription, detected_language, selected_languages)
        
        if translation_result and 'translations' in translation_result:
            # Format response
            source_lang_name = LANGUAGE_NAMES.get(detected_language, detected_language)
            source_flag = get_flag_emoji(detected_language)
            
            response_parts = [
                f"🎙️ **Voice Message Transcribed (Parakeet AI):**",
                f"📝 {source_flag} *{source_lang_name}*: \"{transcription}\"",
                "",
                "🌍 **Translations:**"
            ]
            
            # Add translations
            for lang_code, translation_data in translation_result['translations'].items():
                lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
                flag = get_flag_emoji(lang_code)
                
                if isinstance(translation_data, dict):
                    translation_text = translation_data.get('text', 'Translation failed')
                else:
                    translation_text = str(translation_data)
                
                response_parts.append(f"{flag} **{lang_name}**: {translation_text}")
            
            response_text = "\n".join(response_parts)
            
            # Create audio buttons
            keyboard = []
            audio_row = []
            
            for lang_code in translation_result['translations'].keys():
                flag = get_flag_emoji(lang_code)
                audio_row.append(InlineKeyboardButton(f"🎧 {flag}", callback_data=f"audio_{lang_code}"))
                
                if len(audio_row) >= 3:
                    keyboard.append(audio_row)
                    audio_row = []
            
            if audio_row:
                keyboard.append(audio_row)
            
            reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
            
            # Store for audio generation
            context.user_data['last_translation'] = translation_result
            
            # Update message
            await processing_msg.edit_text(
                response_text,
                parse_mode='Markdown',
                reply_markup=reply_markup
            )
        else:
            await processing_msg.edit_text(
                f"🎙️ **Voice Transcribed:** {transcription}\n\n"
                f"❌ Translation failed. Please try again."
            )
        
        # Clean up voice file
        try:
            os.unlink(voice_path)
            logger.info(f"Cleaned up voice file: {voice_path}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up voice file: {cleanup_error}")
    
    except Exception as e:
        logger.error(f"Parakeet voice message processing failed: {e}")
        await update.message.reply_text(f"❌ Voice processing failed: {str(e)}")

def fallback_speech_recognition(voice_path: str) -> Optional[dict]:
    """
    Fallback speech recognition focusing on Portuguese, Russian, Chinese first.
    """
    try:
        import speech_recognition as sr
        from pydub import AudioSegment
        
        # Convert to WAV
        audio = AudioSegment.from_file(voice_path)
        wav_path = voice_path.replace('.ogg', '.wav')
        audio.export(wav_path, format="wav")
        
        r = sr.Recognizer()
        
        with sr.AudioFile(wav_path) as source:
            audio_data = r.record(source)
            
            # Priority order: Portuguese, Russian, Chinese first
            priority_languages = [
                ('pt-PT', 'pt'), ('pt-BR', 'pt'),
                ('ru-RU', 'ru'),
                ('zh-CN', 'zh-CN'), ('zh-TW', 'zh-CN'),
                ('es-ES', 'es'), ('fr-FR', 'fr'), ('it-IT', 'it'),
                ('en-US', 'en'), ('ja-JP', 'ja')
            ]
            
            for google_lang, base_lang in priority_languages:
                try:
                    result = r.recognize_google(audio_data, language=google_lang)
                    if result and result.strip():
                        # Quick language verification
                        detected = detect_language_from_text(result)
                        if detected == base_lang or (detected == 'zh-CN' and base_lang == 'zh-CN'):
                            os.unlink(wav_path)
                            return {
                                'text': result.strip(),
                                'language': detected,
                                'confidence': 0.7
                            }
                except Exception:
                    continue
        
        os.unlink(wav_path)
        return None
        
    except Exception as e:
        logger.error(f"Fallback recognition failed: {e}")
        return None