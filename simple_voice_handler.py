"""
Simple voice message handler that matches the web version behavior.
Uses ElevenLabs for transcription and translates only to user-selected languages.
"""

import logging
import os
import tempfile
from typing import Optional, List, Dict, Any
import json

logger = logging.getLogger(__name__)

# Default languages to use if none selected
DEFAULT_LANGUAGES = ['en', 'es', 'pt', 'it', 'fr', 'ru', 'zh-CN']

def get_user_selected_languages(user_data: Dict[str, Any]) -> List[str]:
    """Get user's selected languages, fallback to defaults if none set."""
    if not user_data or not user_data.get('selected_languages'):
        return DEFAULT_LANGUAGES
    try:
        return json.loads(user_data['selected_languages'])
    except (json.JSONDecodeError, TypeError):
        return DEFAULT_LANGUAGES

async def process_voice_message_simple(update, context, voice_file_path: str) -> bool:
    """
    Process voice message using simple approach that matches web version.
    
    Args:
        update: Telegram update object
        context: Telegram context object
        voice_file_path: Path to the voice file
        
    Returns:
        bool: True if processing was successful
    """
    chat_id = update.effective_chat.id
    
    try:
        # Import database and get user settings
        from database import DatabaseManager
        db = DatabaseManager()
        user_data = db.get_user(str(chat_id))
        selected_languages = get_user_selected_languages(user_data)
        
        # Send initial processing message
        status_message = await context.bot.send_message(
            chat_id=chat_id,
            text="🎤 Processing voice message..."
        )
        
        # Step 1: Transcribe using ElevenLabs (primary method)
        transcription = None
        try:
            from elevenlabs_transcription import transcribe_with_elevenlabs
            logger.info("Attempting transcription with ElevenLabs")
            transcription = transcribe_with_elevenlabs(voice_file_path)
            if transcription:
                logger.info(f"ElevenLabs transcription successful: {transcription[:100]}...")
        except Exception as e:
            logger.warning(f"ElevenLabs transcription failed: {e}")
        
        # Step 2: Fallback to Whisper if ElevenLabs fails
        if not transcription:
            try:
                from new_voice_transcription import transcribe_audio_with_whisper
                logger.info("Falling back to Whisper transcription")
                transcription = transcribe_audio_with_whisper(voice_file_path)
                if transcription:
                    logger.info(f"Whisper transcription successful: {transcription[:100]}...")
            except Exception as e:
                logger.warning(f"Whisper transcription failed: {e}")
        
        # If transcription failed completely
        if not transcription:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text="❌ Could not transcribe voice message. Please try again or type your message."
            )
            return False
        
        # Update status with transcription
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message.message_id,
            text=f"🎤 Transcription: {transcription}\n\n🔄 Translating to selected languages..."
        )
        
        # Step 3: Detect source language
        from enhanced_translator import detect_language
        detected_lang = detect_language(transcription)
        logger.info(f"Detected language: {detected_lang}")
        
        # Step 4: Translate to selected languages (excluding detected language)
        target_languages = [lang for lang in selected_languages if lang != detected_lang]
        
        if not target_languages:
            # If no target languages (user only selected the detected language)
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text=f"🎤 Transcription: {transcription}\n\n✅ No translation needed - already in selected language!"
            )
            return True
        
        # Translate using AI services
        from ai_services_simplified import translate_text_with_fallback
        translations = translate_text_with_fallback(transcription, target_languages)
        
        # Step 5: Store translation data for future reference
        context.user_data['last_translation'] = {
            'original_text': transcription,
            'detected_language': detected_lang,
            'translations': {}
        }
        
        # Build translations dict with proper structure
        for lang_code, translation in translations.items():
            if translation:
                context.user_data['last_translation']['translations'][lang_code] = {
                    'text': translation,
                    'language': lang_code
                }
        
        # Step 6: Display results (like web version)
        result_text = f"🎤 **Voice Message Processed**\n\n"
        result_text += f"📝 **Original ({detected_lang.upper()}):** {transcription}\n\n"
        
        if translations:
            result_text += "🌍 **Translations:**\n"
            for lang_code, translation in translations.items():
                if translation:
                    flag = get_flag_emoji(lang_code)
                    result_text += f"{flag} **{lang_code.upper()}:** {translation}\n"
        
        # Update final message
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message.message_id,
            text=result_text,
            parse_mode='Markdown'
        )
        
        return True
        
    except Exception as e:
        logger.error(f"Error in voice processing: {e}")
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text="❌ Error processing voice message. Please try again."
            )
        except:
            pass
        return False

def get_flag_emoji(lang_code: str) -> str:
    """Get flag emoji for language code."""
    flag_map = {
        'en': '🇺🇸',
        'es': '🇪🇸', 
        'pt': '🇵🇹',
        'it': '🇮🇹',
        'fr': '🇫🇷',
        'ru': '🇷🇺',
        'zh-CN': '🇨🇳',
        'zh': '🇨🇳',
        'de': '🇩🇪',
        'ja': '🇯🇵',
        'ko': '🇰🇷'
    }
    return flag_map.get(lang_code, '🌍')