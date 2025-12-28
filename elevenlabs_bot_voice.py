"""
ElevenLabs voice message handler for Telegram bot.
This matches the web version's voice transcription functionality using ElevenLabs.
"""

import os
import logging
import uuid
from typing import Optional
from telegram import Update
from telegram.ext import ContextTypes
import requests
import json
from ai_services_simplified import translate_text_with_fallback

logger = logging.getLogger(__name__)


def transcribe_with_elevenlabs(audio_path: str) -> Optional[str]:
    """
    Transcribe audio using ElevenLabs API with proper parameters.
    """
    try:
        api_key = os.environ.get('ELEVENLABS_API_KEY')
        if not api_key:
            logger.error("ElevenLabs API key not found")
            return None
            
        url = "https://api.elevenlabs.io/v1/speech-to-text"
        
        # Check if file exists and has content
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            logger.error(f"Audio file does not exist or is empty: {audio_path}")
            return None
        
        with open(audio_path, 'rb') as audio_file:
            files = {
                'audio': audio_file,
                'model_id': (None, 'eleven_multilingual_sts_v2'),
                'language': (None, 'auto')
            }
            headers = {'xi-api-key': api_key}
            
            logger.info(f"Sending transcription request to ElevenLabs for file: {audio_path}")
            response = requests.post(url, files=files, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('text', '').strip()
                if text:
                    logger.info(f"ElevenLabs transcription successful: {text[:50]}...")
                    return text
                else:
                    logger.warning("ElevenLabs returned empty transcription")
                    return None
            else:
                logger.error(f"ElevenLabs transcription failed: {response.status_code} - {response.text}")
                return None
                
    except Exception as e:
        logger.error(f"Error in ElevenLabs transcription: {e}")
        return None


async def handle_voice_message_elevenlabs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle voice messages using ElevenLabs transcription to match web version functionality.
    Only translates to user-selected languages like the web version.
    """
    if not update.message or not update.message.voice:
        return
    
    chat_id = update.message.chat_id
    voice = update.message.voice
    voice_path = None
    
    # Send processing message
    status_message = await update.message.reply_text(
        "🎤 Processing voice message with ElevenLabs..."
    )
    
    try:
        # Create audio directory if it doesn't exist
        os.makedirs('audio', exist_ok=True)
        
        # Download voice file
        voice_file = await context.bot.get_file(voice.file_id)
        unique_id = uuid.uuid4().hex[:8]
        voice_path = f"audio/voice_{unique_id}_{voice.file_id[-8:]}.ogg"
        await voice_file.download_to_drive(voice_path)
        
        logger.info(f"Downloaded voice file: {voice_path}")
        
        # Update status
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message.message_id,
            text="🎧 Transcribing with ElevenLabs..."
        )
        
        # Transcribe with ElevenLabs
        transcription = await transcribe_with_elevenlabs(voice_path)
        
        if transcription and transcription.strip():
            logger.info(f"Transcribed: {transcription}")
            
            # Get user's selected languages (like web version)
            selected_languages = context.user_data.get('selected_languages', ['en', 'es', 'pt', 'it', 'fr', 'ru'])
            
            # Update status for translation
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text="🌍 Translating to selected languages..."
            )
            
            # Translate to selected languages like web version
            translations = translate_text_with_fallback(transcription, selected_languages)
            
            # Format response like web version
            response_text = f"🎤 **Transcribed:** {transcription}\n\n"
            
            if translations:
                response_text += "🌍 **Translations:**\n"
                
                # Language flags mapping
                flags = {
                    'en': '🇺🇸', 'es': '🇪🇸', 'pt': '🇵🇹', 
                    'it': '🇮🇹', 'fr': '🇫🇷', 'ru': '🇷🇺',
                    'zh': '🇨🇳', 'de': '🇩🇪', 'ja': '🇯🇵'
                }
                
                for lang_code, translation in translations.items():
                    if translation and translation.strip() and translation != transcription:
                        flag = flags.get(lang_code, '🌍')
                        response_text += f"{flag} **{lang_code.upper()}:** {translation}\n"
            
            # Update final message
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text=response_text,
                parse_mode='Markdown'
            )
            
            # Store transcription for potential use
            context.user_data['last_transcription'] = transcription
            
        else:
            # Transcription failed
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text="❌ Could not transcribe voice message. Please try speaking more clearly or use text instead."
            )
            
    except Exception as e:
        logger.error(f"Error processing voice message with ElevenLabs: {e}")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text="❌ Error processing voice message. Please try again or use text instead."
            )
        except:
            pass
    
    finally:
        # Clean up voice file
        if voice_path and os.path.exists(voice_path):
            try:
                os.remove(voice_path)
                logger.info(f"Cleaned up voice file: {voice_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up voice file: {cleanup_error}")