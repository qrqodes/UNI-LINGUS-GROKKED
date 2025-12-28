"""
Fast Voice Transcription Handler - Optimized for speed and accuracy.
Uses the best available transcription service with minimal fallbacks.
"""

import os
import logging
import tempfile
from typing import Optional, Tuple
from pydub import AudioSegment

logger = logging.getLogger(__name__)

def transcribe_audio_fast(audio_path: str, language_hint: str = 'en') -> Optional[str]:
    """
    Fast audio transcription using the best available service.
    Priority: ElevenLabs -> Google Speech Recognition -> Whisper
    
    Args:
        audio_path: Path to audio file
        language_hint: Language hint for better accuracy
        
    Returns:
        Transcribed text or None if all methods fail
    """
    
    # Method 1: Try ElevenLabs first (most accurate for multiple languages)
    try:
        if os.environ.get("ELEVENLABS_API_KEY"):
            from elevenlabs_voice import ElevenLabsVoice
            voice_service = ElevenLabsVoice()
            if voice_service.is_available():
                result = voice_service.speech_to_text(audio_path)
                if result and result.get('text'):
                    logger.info(f"ElevenLabs transcription successful: {result['text'][:50]}...")
                    return result['text']
    except Exception as e:
        logger.warning(f"ElevenLabs transcription failed: {e}")
    
    # Method 2: Google Speech Recognition (fast and reliable)
    try:
        import speech_recognition as sr
        
        # Convert to WAV if needed
        wav_path = audio_path
        if not audio_path.endswith('.wav'):
            audio = AudioSegment.from_file(audio_path)
            wav_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            audio.export(wav_path, format="wav")
        
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
        
        # Map language hints to Google's format
        lang_map = {
            'en': 'en-US',
            'es': 'es-ES', 
            'fr': 'fr-FR',
            'it': 'it-IT',
            'pt': 'pt-PT',
            'ru': 'ru-RU',
            'zh-CN': 'zh-CN',
            'de': 'de-DE',
            'ja': 'ja-JP'
        }
        
        google_lang = lang_map.get(language_hint, 'en-US')
        
        try:
            text = recognizer.recognize_google(audio_data, language=google_lang)
            if text:
                logger.info(f"Google Speech Recognition successful: {text[:50]}...")
                # Clean up temp file
                if wav_path != audio_path:
                    try:
                        os.unlink(wav_path)
                    except:
                        pass
                return text
        except sr.UnknownValueError:
            # Try with English as fallback
            if google_lang != 'en-US':
                text = recognizer.recognize_google(audio_data, language='en-US')
                if text:
                    logger.info(f"Google Speech Recognition successful with English fallback: {text[:50]}...")
                    if wav_path != audio_path:
                        try:
                            os.unlink(wav_path)
                        except:
                            pass
                    return text
        
        # Clean up temp file
        if wav_path != audio_path:
            try:
                os.unlink(wav_path)
            except:
                pass
                
    except Exception as e:
        logger.warning(f"Google Speech Recognition failed: {e}")
    
    # Method 3: Try Whisper via OpenAI API (if available)
    try:
        if os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI()
            
            with open(audio_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language_hint if language_hint != 'zh-CN' else 'zh'
                )
                
            if transcript and transcript.text:
                logger.info(f"OpenAI Whisper transcription successful: {transcript.text[:50]}...")
                return transcript.text
                
    except Exception as e:
        logger.warning(f"OpenAI Whisper transcription failed: {e}")
    
    logger.error("All transcription methods failed")
    return None

async def handle_voice_message_fast(update, context):
    """
    Fast voice message handler optimized for speed.
    """
    from telegram import Update
    from telegram.ext import ContextTypes
    import json
    from database import db
    from enhanced_translator import translate_to_all
    
    voice = update.message.voice
    chat_id = update.effective_chat.id
    
    # Send processing message
    status_message = await update.message.reply_text("🎤 Processing voice...")
    
    voice_path = None
    try:
        # Download voice file
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        voice_path = f"audio/voice_{unique_id}_{voice.file_id[-8:]}.ogg"
        
        os.makedirs("audio", exist_ok=True)
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive(voice_path)
        
        # Get user's language preferences
        user_data = db.get_user(chat_id)
        language_hint = 'en'  # Default
        selected_languages = ['en', 'es', 'pt', 'it', 'fr', 'ru', 'zh-CN']  # Default languages
        
        if user_data and user_data.get('selected_languages'):
            try:
                selected_langs = json.loads(user_data['selected_languages'])
                if selected_langs:
                    language_hint = selected_langs[0]  # Use first selected language as hint
                    selected_languages = selected_langs
            except:
                pass
        
        # Fast transcription
        transcription = transcribe_audio_fast(voice_path, language_hint)
        
        if not transcription:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text="❌ Could not transcribe voice message. Please try speaking more clearly or type your message."
            )
            return
        
        # Update status
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message.message_id,
            text=f"🎤 Transcribed: {transcription[:50]}...\n🔄 Translating..."
        )
        
        # Check if in chat mode
        if context.user_data.get('in_chat_mode', False):
            # Handle as AI chat
            chat_history = context.user_data.get('chat_history', [])
            chat_history.append({"role": "user", "content": transcription})
            
            # Get AI response using Grok (fastest)
            try:
                from xai import chat_with_grok, GROK_TEXT_MODEL
                response = chat_with_grok(chat_history, model=GROK_TEXT_MODEL)
                
                if response:
                    chat_history.append({"role": "assistant", "content": response})
                    context.user_data['chat_history'] = chat_history
                    context.user_data['last_ai_response'] = response
                    
                    # Update message with AI response
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message.message_id,
                        text=f"🎤 You said: {transcription}\n\n🤖 AI: {response}"
                    )
                else:
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message.message_id,
                        text=f"🎤 You said: {transcription}\n\n❌ AI response failed"
                    )
            except Exception as e:
                logger.error(f"AI chat error: {e}")
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message.message_id,
                    text=f"🎤 You said: {transcription}\n\n❌ AI error: {str(e)[:50]}"
                )
        else:
            # Handle as translation
            from langdetect import detect
            try:
                detected_lang = detect(transcription)
                if detected_lang == 'zh':
                    detected_lang = 'zh-CN'
            except:
                detected_lang = language_hint
            
            # Fast translation
            translation_result = translate_to_all(transcription, detected_lang, selected_languages)
            
            if translation_result and 'translations' in translation_result:
                # Delete status message and send translations
                await context.bot.delete_message(chat_id=chat_id, message_id=status_message.message_id)
                
                # Send translations
                for lang_code, trans_data in translation_result['translations'].items():
                    if isinstance(trans_data, dict):
                        text = trans_data.get('text', str(trans_data))
                    else:
                        text = str(trans_data)
                    
                    from enhanced_translator import get_flag_emoji, LANGUAGE_NAMES
                    flag = get_flag_emoji(lang_code)
                    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
                    
                    await update.message.reply_text(f"{flag} {lang_name}: {text}")
            else:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message.message_id,
                    text=f"🎤 You said: {transcription}\n\n❌ Translation failed"
                )
    
    except Exception as e:
        logger.error(f"Voice processing error: {e}")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text="❌ Voice processing failed. Please try again."
            )
        except:
            pass
    
    finally:
        # Clean up
        if voice_path and os.path.exists(voice_path):
            try:
                os.unlink(voice_path)
            except:
                pass