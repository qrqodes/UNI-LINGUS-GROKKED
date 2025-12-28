"""
Improved voice recognition focused on Portuguese, Russian, and Chinese.
Uses Google Speech Recognition with Google Translate detection for better accuracy.
"""

import logging
import os
import uuid
import speech_recognition as sr
from pydub import AudioSegment
from deep_translator import GoogleTranslator
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)

# Language mappings with flags
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh-CN': 'Chinese',
    'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi'
}

def get_flag_emoji(lang_code: str) -> str:
    """Get flag emoji for language code"""
    flags = {
        'en': '🇺🇸', 'es': '🇪🇸', 'fr': '🇫🇷', 'de': '🇩🇪',
        'it': '🇮🇹', 'pt': '🇵🇹', 'ru': '🇷🇺', 'zh-CN': '🇨🇳',
        'ja': '🇯🇵', 'ko': '🇰🇷', 'ar': '🇸🇦', 'hi': '🇮🇳'
    }
    return flags.get(lang_code, '🌍')

async def handle_voice_message_improved(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Improved voice message handler with better Portuguese, Russian, and Chinese support.
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
            "🎙️ Analyzing voice message with improved multilingual detection..."
        )
        
        # Convert to WAV
        audio = AudioSegment.from_ogg(voice_path)
        wav_path = voice_path.replace('.ogg', '.wav')
        audio.export(wav_path, format="wav")
        
        # Try speech recognition with focused language order
        r = sr.Recognizer()
        transcription = None
        detected_language = None
        
        with sr.AudioFile(wav_path) as source:
            audio_data = r.record(source)
            
            # Focused language attempts - prioritize Portuguese, Russian, Chinese
            languages_to_try = [
                ('pt-PT', 'pt'),    # Portuguese (Portugal)
                ('pt-BR', 'pt'),    # Portuguese (Brazil)
                ('ru-RU', 'ru'),    # Russian
                ('zh-CN', 'zh'),    # Chinese (Simplified)
                ('zh-TW', 'zh'),    # Chinese (Traditional)
                ('es-ES', 'es'),    # Spanish
                ('fr-FR', 'fr'),    # French
                ('it-IT', 'it'),    # Italian
                ('de-DE', 'de'),    # German
                ('en-US', 'en'),    # English
                ('ja-JP', 'ja'),    # Japanese
                ('ko-KR', 'ko')     # Korean
            ]
            
            recognition_results = {}
            
            for google_lang, base_lang in languages_to_try:
                try:
                    result = r.recognize_google(audio_data, language=google_lang)
                    if result and result.strip():
                        # Use Google Translate to verify the language
                        try:
                            detected = GoogleTranslator.detect(result)
                            confidence_score = 1.0 if detected == base_lang else 0.1
                            
                            # Special handling for Chinese variants
                            if detected == 'zh' and base_lang == 'zh':
                                confidence_score = 1.0
                            
                            recognition_results[google_lang] = {
                                'text': result.strip(),
                                'detected_lang': detected,
                                'confidence': confidence_score,
                                'base_lang': base_lang
                            }
                            
                            logger.info(f"Recognition {google_lang}: {result[:50]}... (detected: {detected}, confidence: {confidence_score})")
                            
                        except Exception as e:
                            logger.debug(f"Language detection failed for {result}: {e}")
                            recognition_results[google_lang] = {
                                'text': result.strip(),
                                'detected_lang': base_lang,
                                'confidence': 0.3,
                                'base_lang': base_lang
                            }
                            
                except Exception as e:
                    logger.debug(f"Recognition failed for {google_lang}: {e}")
                    continue
        
        # Clean up WAV file
        os.unlink(wav_path)
        
        # Select best result
        if recognition_results:
            # Sort by confidence and prefer non-English when confidence is high
            best_result = max(recognition_results.items(), 
                            key=lambda x: (x[1]['confidence'], 
                                         1.5 if x[1]['base_lang'] != 'en' else 1.0,
                                         len(x[1]['text'])))
            
            transcription = best_result[1]['text']
            detected_language = best_result[1]['detected_lang']
            
            logger.info(f"Selected best result: {best_result[0]} - {transcription[:50]}... (detected: {detected_language})")
        
        if not transcription:
            await processing_msg.edit_text(
                "❌ Could not transcribe the voice message. Please try:\n"
                "• Speaking more clearly\n"
                "• Reducing background noise\n"
                "• Using a supported language"
            )
            os.unlink(voice_path)
            return
        
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
        
        # Normalize detected language for translation
        if detected_language == 'zh':
            detected_language = 'zh-CN'
        
        # Translate using enhanced translator
        from enhanced_translator import translate_to_all
        translation_result = translate_to_all(transcription, detected_language, selected_languages)
        
        if translation_result and 'translations' in translation_result:
            # Format response
            source_lang_name = LANGUAGE_NAMES.get(detected_language, detected_language)
            source_flag = get_flag_emoji(detected_language)
            
            response_parts = [
                f"🎙️ **Voice Message Transcribed:**",
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
        logger.error(f"Improved voice message processing failed: {e}")
        await update.message.reply_text(f"❌ Voice processing failed: {str(e)}")