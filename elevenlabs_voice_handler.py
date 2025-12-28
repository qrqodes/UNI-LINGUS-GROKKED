"""
ElevenLabs voice message handler for Telegram bot.
This matches the web version's voice transcription functionality using ElevenLabs.
"""

import os
import logging
import asyncio
import tempfile
import uuid
import time
from typing import Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
import requests
from ai_services_simplified import translate_text_with_fallback

logger = logging.getLogger(__name__)

# Import database function
try:
    from database import get_user
except ImportError:
    logger.warning("Database module not available")
    def get_user(chat_id):
        return None

# Language names mapping
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish', 
    'fr': 'French',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh-cn': 'Chinese',
    'zh': 'Chinese'
}


def get_flag_emoji(lang_code):
    """Get flag emoji for language code."""
    flag_map = {
        'en': '🇺🇸', 'es': '🇪🇸', 'pt': '🇵🇹', 'it': '🇮🇹', 
        'fr': '🇫🇷', 'ru': '🇷🇺', 'zh-cn': '🇨🇳', 'zh': '🇨🇳'
    }
    return flag_map.get(lang_code.lower(), '🌐')


def transcribe_with_elevenlabs(audio_path: str) -> Optional[str]:
    """
    Transcribe audio using ElevenLabs API to match web version functionality.
    """
    try:
        api_key = os.environ.get('ELEVENLABS_API_KEY')
        if not api_key:
            logger.error("ElevenLabs API key not found")
            return None
            
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        
        with open(audio_path, 'rb') as audio_file:
            files = {'file': audio_file}
            headers = {'xi-api-key': api_key}
            data = {
                'model_id': 'scribe_v1',
                'language': 'auto'
            }
            
            response = requests.post(url, files=files, headers=headers, data=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('text', '')
            else:
                logger.error(f"ElevenLabs transcription failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"Error in ElevenLabs transcription: {e}")
        return None


async def handle_voice_message_elevenlabs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle voice messages using ElevenLabs transcription to match web version functionality.
    Translates to all supported languages like the web version.
    """
    if not update.message or not update.message.voice:
        return

    voice_file_path = None
    try:
        # Generate unique filename for the voice file
        unique_id = str(uuid.uuid4())[:8]
        voice_file_path = f"audio/voice_{unique_id}_{update.message.voice.file_unique_id}.ogg"
        
        # Ensure audio directory exists
        os.makedirs("audio", exist_ok=True)
        
        # Download the voice file
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        await voice_file.download_to_drive(voice_file_path)
        
        logger.info(f"Voice file downloaded: {voice_file_path}")
        
        # Transcribe using multiple fallbacks
        transcribed_text = None
        
        # Try ElevenLabs first
        try:
            transcribed_text = transcribe_with_elevenlabs(voice_file_path)
            logger.info(f"ElevenLabs transcription: {transcribed_text}")
        except Exception as e:
            logger.warning(f"ElevenLabs transcription failed: {e}")
        
        # Try AgenticSeek fallback if ElevenLabs failed
        if not transcribed_text:
            try:
                from agenticseek_transcription import transcribe_with_multiple_fallbacks
                transcribed_text = transcribe_with_multiple_fallbacks(voice_file_path)
                logger.info(f"Fallback transcription: {transcribed_text}")
            except Exception as e:
                logger.warning(f"Fallback transcription failed: {e}")
        
        if not transcribed_text:
            await update.message.reply_text("❌ Could not transcribe voice message")
            return
            
        logger.info(f"Transcribed text: {transcribed_text}")
        
        # Check if user is in chat mode
        user_data = context.user_data or {}
        
        # Check chat mode first
        if user_data.get('in_chat_mode'):
            # In chat mode, generate AI response instead of translations
            from ai_services_simplified import query_claude
            
            # Get chat history
            if 'chat_history' not in user_data:
                user_data['chat_history'] = []
            
            # Add user message to history
            user_data['chat_history'].append({
                'role': 'user',
                'content': transcribed_text
            })
            
            # Generate AI response
            prompt = f"User said via voice: {transcribed_text}\n\nPlease respond naturally as a helpful AI assistant."
            ai_response = query_claude(prompt)
            
            if ai_response:
                # Add AI response to history
                user_data['chat_history'].append({
                    'role': 'assistant', 
                    'content': ai_response
                })
                
                # Send AI response
                await update.message.reply_text(f"🤖 {ai_response}")
                
                # Try to generate voice response
                try:
                    from enhanced_audio import generate_audio
                    audio_file = generate_audio(ai_response, 'en')
                    if audio_file and os.path.exists(audio_file):
                        with open(audio_file, 'rb') as audio:
                            await context.bot.send_voice(
                                chat_id=update.message.chat_id,
                                voice=audio,
                                caption="🎧 AI Voice Response"
                            )
                        os.remove(audio_file)
                except Exception as e:
                    logger.warning(f"Could not generate voice response: {e}")
            else:
                await update.message.reply_text("🎤 " + transcribed_text)
            return
            
        # Not in chat mode, proceed with translation
        # Get user's selected languages from database
        chat_id = update.message.chat_id
        target_languages = ['en', 'es', 'fr', 'it', 'pt']  # Default languages
        
        try:
            user_data_db = get_user(chat_id)
            if user_data_db and user_data_db.get('selected_languages'):
                import json
                target_languages = json.loads(user_data_db['selected_languages'])
        except Exception as e:
            logger.warning(f"Could not get user languages: {e}")
            
        translations = translate_text_with_fallback(transcribed_text, target_languages)
        
        if not translations:
            await update.message.reply_text(f"🎤 {transcribed_text}")
            return
            
        logger.info(f"Voice translations generated for {len(translations)} languages")
        
        # Send original transcription first
        await update.message.reply_text(f"🎤 Original: {transcribed_text}")
        
        # Prepare all translations into a single message with flag emoji buttons
        translation_lines = []
        flag_buttons = []
        
        for lang_code, translation_info in translations.items():
            if isinstance(translation_info, dict):
                translated_text = translation_info.get('text', 'Translation failed')
            else:
                translated_text = str(translation_info)
                
            if not translated_text or translated_text.strip() == 'Translation failed':
                continue
                
            flag = get_flag_emoji(lang_code)
            lang_name = LANGUAGE_NAMES.get(lang_code, lang_code.upper())
            translation_lines.append(f"{flag} **{lang_name}**: {translated_text}")
            
            # Add flag button for audio generation with headphone emoji
            flag_buttons.append(InlineKeyboardButton(f"🎧 {flag}", callback_data=f"audio_{lang_code}"))
        
        # Send single message with all translations and flag buttons at the bottom
        if translation_lines and flag_buttons:
            message_text = "\n\n".join(translation_lines)
            
            # Arrange flag buttons in rows of 3
            keyboard = []
            for i in range(0, len(flag_buttons), 3):
                keyboard.append(flag_buttons[i:i+3])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await context.bot.send_message(
                chat_id=update.message.chat_id,
                text=message_text,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        
        # Store translation data for audio generation in the correct format
        if context.user_data is not None:
            # Convert translation format to match what audio handler expects
            formatted_translations = {}
            for lang_code, translation_info in translations.items():
                if isinstance(translation_info, dict):
                    formatted_translations[lang_code] = {
                        'text': translation_info.get('text', 'Translation failed')
                    }
                else:
                    formatted_translations[lang_code] = {
                        'text': str(translation_info)
                    }
            
            context.user_data['last_translation'] = {
                'original_text': transcribed_text,
                'translations': formatted_translations,
                'timestamp': time.time()
            }
        
    except Exception as e:
        logger.error(f"Error processing voice message: {e}", exc_info=True)
        await update.message.reply_text(f"❌ Error processing voice message: {str(e)[:100]}")
    finally:
        # Clean up voice file
        if voice_file_path and os.path.exists(voice_file_path):
            try:
                os.remove(voice_file_path)
                logger.info(f"Cleaned up temporary voice file: {voice_file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up voice file: {e}")