"""
Enhanced version of the Telegram translation bot with advanced learning features.
Includes pinyin for Chinese, Latin transcription for Russian, and AI fallback.
"""
import os
import sys
import json
import time
import logging
import random
import tempfile
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, 
    ReplyKeyboardMarkup, ReplyKeyboardRemove, BotCommand, Message
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, ConversationHandler, filters, CallbackContext, PicklePersistence
)

# Import shared services (unified with web app for consistent quality)
from shared_services import (
    translate_text, translate_to_all_languages, generate_audio,
    chat_with_ai, is_ai_available, get_service_status,
    DEFAULT_LANGUAGES, LANGUAGE_NAMES, ALL_LANGUAGE_CODES, ALL_LANGUAGES,
    get_flag_emoji, translate_to_uni, translate_from_uni
)
from transcription import get_transcription
# Use shared_services for TTS (ElevenLabs → gTTS cascade)
from audio_generation import clean_old_audio_files
from database import db
from ai_services_simplified import (
    generate_learning_example, generate_thematic_vocabulary,
    is_ai_service_available, query_claude
)
# Import advanced AI modules (music/video generation removed - not using paid APIs)
from autoregressive_speech import (
    generate_streaming_speech, transcribe_streaming_audio,
    get_autoregressive_capabilities
)
from advanced_models import (
    enhanced_model_translation, enhanced_tts_with_parakeet,
    get_advanced_model_capabilities, initialize_memory,
    update_memory, get_enhanced_context, cleanup_old_memories
)

# Import existing modules
from vocabulary_data import (
    get_vocab_by_level, get_word_options, get_translation_challenge
)
from language_facts import (
    get_language_fact, get_cultural_trivia
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mask sensitive information in httpx logs
httpx_logger = logging.getLogger('httpx')
httpx_logger.setLevel(logging.WARNING)  # Only show warnings/errors, not INFO requests

# Get the Telegram token from environment variables
TOKEN = os.environ.get('TELEGRAM_TOKEN') or os.environ.get('TELEGRAM_BOT_TOKEN')
if not TOKEN:
    logger.error("No TELEGRAM_TOKEN found in environment variables.")
    sys.exit(1)  # Exit if no token is provided

# Conversation states
(
    AWAITING_TEXT, AWAITING_LANGUAGE_SELECTION, AWAITING_LEVEL_SELECTION,
    AWAITING_GAME_PAIR, AWAITING_GAME_ANSWER, AWAITING_THEME, 
    AWAITING_CUSTOM_NOTIFICATION, AWAITING_TIME_SELECTION
) = range(8)

# Helper functions
def get_language_keyboard(selected_langs=None):
    """Generate a keyboard for language selection."""
    if selected_langs is None:
        selected_langs = []
    
    keyboard = []
    row = []
    
    for i, lang_code in enumerate(ALL_LANGUAGE_CODES):
        flag = get_flag_emoji(lang_code)
        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
        status = "✅" if lang_code in selected_langs else "⬜"
        button_text = f"{status} {flag} {lang_name}"
        
        # Create rows of 2 buttons each
        row.append(InlineKeyboardButton(button_text, callback_data=f"toggle_{lang_code}"))
        
        if i % 2 == 1 or i == len(ALL_LANGUAGE_CODES) - 1:
            keyboard.append(row)
            row = []
    
    # Note: "Done" button removed as requested by user
    # Each toggle action will automatically update the languages
    
    return InlineKeyboardMarkup(keyboard)

def get_level_keyboard():
    """Generate a keyboard for level selection."""
    keyboard = [
        [InlineKeyboardButton("📘 Beginner", callback_data="level_beginner")],
        [InlineKeyboardButton("📗 Intermediate", callback_data="level_intermediate")],
        [InlineKeyboardButton("📙 Advanced", callback_data="level_advanced")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_notification_keyboard():
    """Generate a keyboard for notification settings."""
    keyboard = [
        [InlineKeyboardButton("🔔 Daily Word", callback_data="notify_daily_word")],
        [InlineKeyboardButton("🔄 Learning Review", callback_data="notify_review")],
        [InlineKeyboardButton("📚 Language Facts", callback_data="notify_facts")],
        [InlineKeyboardButton("📅 Custom Schedule", callback_data="notify_custom")],
        [InlineKeyboardButton("❌ Disable All", callback_data="notify_disable")],
        [InlineKeyboardButton("✅ Done", callback_data="notify_done")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_time_slot_keyboard():
    """Generate a keyboard for time slot selection."""
    keyboard = []
    
    # Morning slots
    row1 = []
    for hour in range(6, 12):
        time_str = f"{hour:02d}:00"
        row1.append(InlineKeyboardButton(f"🌅 {time_str}", callback_data=f"time_{time_str}"))
    keyboard.append(row1)
    
    # Afternoon slots
    row2 = []
    for hour in range(12, 18):
        time_str = f"{hour:02d}:00"
        row2.append(InlineKeyboardButton(f"☀️ {time_str}", callback_data=f"time_{time_str}"))
    keyboard.append(row2)
    
    # Evening slots
    row3 = []
    for hour in range(18, 24):
        time_str = f"{hour:02d}:00"
        row3.append(InlineKeyboardButton(f"🌙 {time_str}", callback_data=f"time_{time_str}"))
    keyboard.append(row3)
    
    # Cancel button
    keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="time_cancel")])
    
    return InlineKeyboardMarkup(keyboard)

def get_post_translation_keyboard(ai_available=True):
    """Generate a keyboard for actions after translation."""
    keyboard = [
        [InlineKeyboardButton("🎧 Hear Audio", callback_data="action_audio")]
    ]
    
    # We've removed the Save button as requested
        
    return InlineKeyboardMarkup(keyboard)

    


def format_translation_message(translation_data: Dict[str, Any]) -> str:
    """Format the translation message. Transcriptions are already included in the text."""
    source_lang = translation_data['source_language']
    source_lang_name = translation_data['source_language_name']
    translations = translation_data['translations']
    source_text = translation_data.get('original_text', '')
    
    # Start message with just the source text, without the language name prefix
    message = f"{source_text}\n\n"
    
    # Add each translation (transcriptions are already included in the text by enhanced_translator)
    for lang_code, trans_data in translations.items():
        text = trans_data['text']
        
        flag = get_flag_emoji(lang_code)
        lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
        
        message += f"{flag} *{lang_name}:* {text}\n"
        
        # Add pinyin for Chinese translations
        if lang_code == 'zh-CN' and 'pinyin' in trans_data:
            message += f"    📖 _{trans_data['pinyin']}_\n"
        
        message += "\n"
    
    return message

async def audio_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /audio command - generate audio from last available text.
    This command will look for the last message/translation in the chat
    and generate audio for ALL selected languages.
    """
    chat_id = update.effective_chat.id
    
    # Send a loading message
    loading_message = await update.message.reply_text("🔍 Looking for text to convert to audio...")
    
    try:
        # Get user's selected languages
        user_data = db.get_user(chat_id)
        if not user_data or not user_data.get('selected_languages'):
            selected_langs = DEFAULT_LANGUAGES
        else:
            selected_langs = json.loads(user_data['selected_languages'])
        
        # First check for last translation
        translation_data = context.user_data.get('last_translation')
        
        if translation_data and translation_data.get('translations'):
            # Update loading message
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_message.message_id,
                text=f"🔊 Generating audio in all selected languages..."
            )
            
            # Generate and send audio for each language in the translation
            success_count = 0
            error_count = 0
            
            for lang_code, trans_data in translation_data.get('translations', {}).items():
                if lang_code in selected_langs:  # Only generate for selected languages
                    text = trans_data.get('text', '')
                    if not text:
                        error_count += 1
                        continue
                        
                    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
                    flag = get_flag_emoji(lang_code)
                    
                    try:
                        # Clean text for audio generation (remove transliterations)

                        from transliteration_utils import clean_text_for_audio

                        clean_text = clean_text_for_audio(text)

                        

                        # Generate audio file (returns tuple: path, provider)
                        audio_result = generate_audio(clean_text, lang_code)
                        audio_file = audio_result[0] if isinstance(audio_result, tuple) else audio_result
                        
                        if audio_file:
                            # Send the audio
                            with open(audio_file, 'rb') as audio:
                                await context.bot.send_voice(
                                    chat_id=chat_id,
                                    voice=audio,
                                    caption=f"🎧 {flag} {lang_name}: {text[:50]}{'...' if len(text) > 50 else ''}"
                                )
                            
                            # Delete the temporary file
                            import os
                            if os.path.exists(audio_file):
                                os.remove(audio_file)
                            success_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        logger.error(f"Error generating audio for {lang_code}: {e}")
                        error_count += 1
            
            # Delete or update status message
            if success_count > 0:
                # Delete status message if at least one audio was sent
                await context.bot.delete_message(
                    chat_id=chat_id,
                    message_id=loading_message.message_id
                )
                
                # Add button for chat with AI after audio translations
                keyboard = [
                    [InlineKeyboardButton("💬 Chat with AI", callback_data="chat_ai")]
                ]
                await context.bot.send_message(
                    chat_id=chat_id,
                    text="What would you like to do next?",
                    reply_markup=InlineKeyboardMarkup(keyboard)
                )
                
                if error_count > 0:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"⚠️ Note: Failed to generate audio for {error_count} language(s)."
                    )
            else:
                # Update status message with error if all failed
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading_message.message_id,
                    text=f"😓 Sorry, I couldn't generate audio in any language. Please try again later."
                )
            return
            
        # If no translation found, check for last message
        last_message = context.user_data.get('last_message')
        if last_message:
            # Try to detect the language using shared services
            from langdetect import detect
            detected_lang = detect(last_message)
            
            # Update loading message
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_message.message_id,
                text=f"🔊 Translating and generating audio in all selected languages..."
            )
            
            try:
                # Translate to all selected languages using shared services (same as web app)
                translation_result = {'translations': translate_to_all_languages(last_message, selected_langs)}
                
                # Store the translation for future use
                context.user_data['last_translation'] = translation_result
                
                # Generate and send audio for each language
                success_count = 0
                error_count = 0
                
                for lang_code, trans_data in translation_result.get('translations', {}).items():
                    text = trans_data.get('text', '')
                    if not text:
                        error_count += 1
                        continue
                        
                    lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
                    flag = get_flag_emoji(lang_code)
                    
                    try:
                        # Clean text for audio generation (remove transliterations)
                        from transliteration_utils import clean_text_for_audio
                        clean_text = clean_text_for_audio(text)

                        # Generate audio file (returns tuple: path, provider)
                        audio_result = generate_audio(clean_text, lang_code)
                        audio_file = audio_result[0] if isinstance(audio_result, tuple) else audio_result
                        
                        if audio_file:
                            # Send the audio
                            with open(audio_file, 'rb') as audio:
                                await context.bot.send_voice(
                                    chat_id=chat_id,
                                    voice=audio,
                                    caption=f"🎧 {flag} {lang_name}: {text[:50]}{'...' if len(text) > 50 else ''}"
                                )
                            
                            # Delete the temporary file
                            import os
                            if os.path.exists(audio_file):
                                os.remove(audio_file)
                            success_count += 1
                        else:
                            error_count += 1
                    except Exception as e:
                        logger.error(f"Error generating audio for {lang_code}: {e}")
                        error_count += 1
                
                # Delete or update status message
                if success_count > 0:
                    # Delete status message if at least one audio was sent
                    await context.bot.delete_message(
                        chat_id=chat_id,
                        message_id=loading_message.message_id
                    )
                    
                    # Add button for chat with AI after audio translations
                    keyboard = [
                        [InlineKeyboardButton("💬 Chat with AI", callback_data="chat_ai")]
                    ]
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="What would you like to do next?",
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )
                    
                    if error_count > 0:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=f"⚠️ Note: Failed to generate audio for {error_count} language(s)."
                        )
                else:
                    # Update status message with error if all failed
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=loading_message.message_id,
                        text=f"😓 Sorry, I couldn't generate audio in any language. Please try again later."
                    )
                return
                
            except Exception as e:
                logger.error(f"Error translating before audio generation: {e}")
                # Try with just the original language
                try:
                    # Update loading message
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=loading_message.message_id,
                        text=f"🔊 Generating audio for original text in {LANGUAGE_NAMES.get(detected_lang, detected_lang)}..."
                    )
                    
                    # Generate audio for original text only (returns tuple: path, provider)
                    audio_result = generate_audio(last_message, detected_lang)
                    audio_file = audio_result[0] if isinstance(audio_result, tuple) else audio_result
                    
                    if audio_file:
                        # Send the audio file
                        with open(audio_file, 'rb') as audio:
                            await context.bot.send_voice(
                                chat_id=chat_id,
                                voice=audio,
                                caption=f"🎧 Audio in {LANGUAGE_NAMES.get(detected_lang, detected_lang)}: \"{last_message[:50]}{'...' if len(last_message) > 50 else ''}\""
                            )
                        
                        # Clean up the audio file
                        import os
                        if os.path.exists(audio_file):
                            os.remove(audio_file)
                        
                        # Delete the loading message
                        await context.bot.delete_message(
                            chat_id=chat_id,
                            message_id=loading_message.message_id
                        )
                        
                        # Add button for chat with AI after audio translations
                        keyboard = [
                            [InlineKeyboardButton("💬 Chat with AI", callback_data="chat_ai")]
                        ]
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text="What would you like to do next?",
                            reply_markup=InlineKeyboardMarkup(keyboard)
                        )
                        return
                    else:
                        raise Exception("Failed to generate audio for original language")
                        
                except Exception as inner_e:
                    logger.error(f"Error in fallback audio generation: {inner_e}")
                    raise e  # Re-raise the original error for consistent error handling
        
        # If we get here, no suitable text was found
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_message.message_id,
            text="❌ No recent text found to convert to audio. Please send a message first, then use /audio."
        )
    except Exception as e:
        logger.error(f"Error in audio command: {e}")
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_message.message_id,
            text=f"❌ Error generating audio: {str(e)[:100]}"
        )

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    user = update.effective_user
    chat_id = update.effective_chat.id
    
    # Create user in database if not exists
    db.create_or_update_user(
        chat_id=chat_id,
        username=user.username,
        selected_languages=DEFAULT_LANGUAGES,  # Use default languages (multi-language)
        learning_level='advanced'  # Default to advanced
    )
    
    # Set translation mode as default
    context.user_data['in_chat_mode'] = False
    
    # Update the database with the user's preference
    db.set_chat_mode(chat_id, False)
    
    # Create plain keyboard buttons with forward slash for commands
    keyboard = [
        ["/languages", "/flags"],
        ["/chat", "/translator"],
        ["/audio", "/shop"],
        ["/webapp"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    # Concise English welcome message
    welcome_text = (
        f"*Type any message to translate it*\n\n"
        f"*Commands:*\n"
        f"🇨🇳 /languages - Select languages\n"
        f"🚩 /flags - Toggle language flags display\n"
        f"💬 /chat - Switch to chatting with AI mode\n"
        f"🔄 /translator - Switch to translation mode\n"
        f"🔊 /audio - Generate speech\n"
        f"🛍️ /shop - Buy clothes 👕 and language classes 👩‍🏫\n"
        f"🌐 /webapp - Open web version"
    )
    
    # Send English welcome message with keyboard
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN, reply_markup=reply_markup)
    
    logger.info(f"User {user.id} started the bot")

# Help command removed as requested

async def languages_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle the /languages command."""
    chat_id = update.effective_chat.id
    
    # Get user data from database
    user_data = db.get_user(chat_id)
    if not user_data or not user_data.get('selected_languages'):
        selected_langs = DEFAULT_LANGUAGES  # Default to default languages
    else:
        # Parse JSON string from database
        selected_langs = json.loads(user_data['selected_languages'])
    
    await update.message.reply_text(
        "🇨🇳 Select languages to translate to:",
        reply_markup=get_language_keyboard(selected_langs)
    )
    
    return AWAITING_LANGUAGE_SELECTION

async def handle_language_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle language selection."""
    try:
        query = update.callback_query
        await query.answer()
        chat_id = query.message.chat_id
        
        # Extract the language code
        lang_code = query.data.replace("toggle_", "")
        
        # Get user data from database with error handling
        try:
            user_data = db.get_user(chat_id)
            if not user_data or not user_data.get('selected_languages'):
                selected_langs = DEFAULT_LANGUAGES.copy()  # Default to default languages
            else:
                # Parse JSON string from database
                selected_langs = json.loads(user_data['selected_languages'])
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            selected_langs = DEFAULT_LANGUAGES.copy()  # Use defaults if database error
        
        # Toggle language selection
        if lang_code in selected_langs:
            # Don't remove if it's the last language
            if len(selected_langs) > 1:
                selected_langs.remove(lang_code)
        else:
            selected_langs.append(lang_code)
        
        # Update database with error handling
        try:
            db.update_user_preferences(chat_id, {'selected_languages': selected_langs})
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            # Continue anyway, as we'll use the in-memory changes
        
        # Format language names for the success message
        lang_names = [f"{get_flag_emoji(lang)} {LANGUAGE_NAMES.get(lang, lang)}" for lang in selected_langs]
        
        # Show both updated language selection and success message
        success_message = (
            f"✅ Languages updated! Type any text to translate it.\n\n"
            f"Selected languages: {', '.join(lang_names[:5])}"
            f"{'...' if len(lang_names) > 5 else ''}\n\n"
            f"You can continue to adjust your languages:"
        )
        
        # Try to edit the message, if it fails, send a new one
        try:
            await query.edit_message_text(
                success_message,
                reply_markup=get_language_keyboard(selected_langs)
            )
        except Exception as e:
            logger.error(f"Error editing language selection message: {e}")
            await context.bot.send_message(
                chat_id=chat_id,
                text=success_message,
                reply_markup=get_language_keyboard(selected_langs)
            )
        
        return AWAITING_LANGUAGE_SELECTION
        
    except Exception as e:
        logger.error(f"Unexpected error in language selection: {e}")
        # Try to send a fallback message
        try:
            chat_id = update.effective_chat.id
            await context.bot.send_message(
                chat_id=chat_id,
                text="Sorry, there was an error updating your language preferences. Please try the /languages command again."
            )
        except Exception as fallback_error:
            logger.error(f"Failed to send fallback message: {fallback_error}")
        
        return ConversationHandler.END

async def level_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle the /level command."""
    chat_id = update.effective_chat.id
    
    # Get user data from database
    user_data = db.get_user(chat_id)
    current_level = user_data.get('learning_level', 'advanced') if user_data else 'advanced'
    
    await update.message.reply_text(
        f"📚 Your current level is: *{current_level.capitalize()}*\n\n"
        f"Select your vocabulary level for daily words, games, and challenges:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_level_keyboard()
    )
    
    return AWAITING_LEVEL_SELECTION

async def handle_level_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle level selection."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    # Extract the level
    level = query.data.split("_")[1]
    
    # Update database
    db.update_user_preferences(chat_id, {'learning_level': level})
    
    # Confirm level selection without prompting for language selection
    await query.edit_message_text(
        f"📚 Your level has been set to *{level.capitalize()}*.\n\n"
        f"The difficulty of daily words, games, and challenges will be adjusted accordingly.",
        parse_mode=ParseMode.MARKDOWN
    )
    
    # End the conversation without starting another one
    return ConversationHandler.END

async def random_word_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /random command - now disabled."""
    chat_id = update.effective_chat.id
    await context.bot.send_message(
        chat_id=chat_id,
        text="The random word feature has been disabled in this version. Please use translation features instead.",
        parse_mode=ParseMode.MARKDOWN
    )

async def send_daily_word(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the daily vocabulary word - now disabled."""
    # Send a simple message that this feature is disabled
    await context.bot.send_message(
        chat_id=chat_id,
        text="The daily word feature has been disabled in this version. Please use translation features instead.",
        parse_mode=ParseMode.MARKDOWN
    )

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /settings command for notification preferences."""
    chat_id = update.effective_chat.id
    
    # Get current notification settings from database
    prefs = db.get_notification_preferences(chat_id)
    
    # Build current settings message
    if prefs:
        notify_daily = "✅" if prefs.get('notify_daily') else "❌"
        notify_review = "✅" if prefs.get('notify_review') else "❌"
        notify_facts = "✅" if prefs.get('notify_facts') else "❌"
        time_str = prefs.get('notification_time', 'Not set')
        
        settings_message = (
            f"🔔 *Your Notification Settings*\n\n"
            f"Daily Word: {notify_daily}\n"
            f"Learning Review: {notify_review}\n"
            f"Language Facts: {notify_facts}\n"
            f"Custom Time: {time_str}\n\n"
            f"Select which notifications to toggle:"
        )
    else:
        settings_message = (
            f"🔔 *Notification Settings*\n\n"
            f"You don't have any notifications set up yet.\n"
            f"Select which notifications to enable:"
        )
    
    await update.message.reply_text(
        settings_message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_notification_keyboard()
    )

async def handle_notifications_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle notification settings selection."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    if query.data == "notify_done":
        await query.edit_message_text(
            "✅ Notification settings updated!",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    if query.data == "notify_disable":
        # Disable all notifications
        db.set_notification_preferences(
            chat_id,
            notify_daily=False,
            notify_review=False,
            notify_facts=False
        )
        
        await query.edit_message_text(
            "🔕 All notifications have been disabled.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_notification_keyboard()
        )
        return
    
    if query.data == "notify_custom":
        await query.edit_message_text(
            "⏰ Select a time for your daily notifications:",
            reply_markup=get_time_slot_keyboard()
        )
        return
    
    # Handle individual notification toggles
    if query.data == "notify_daily_word":
        # Get current setting
        prefs = db.get_notification_preferences(chat_id)
        current = prefs.get('notify_daily', False) if prefs else False
        
        # Toggle setting
        db.set_notification_preferences(chat_id, notify_daily=not current)
        
        if not current:
            await query.message.reply_text(
                "✅ Daily word notifications enabled!"
            )
        else:
            await query.message.reply_text(
                "❌ Daily word notifications disabled."
            )
    
    elif query.data == "notify_review":
        # Get current setting
        prefs = db.get_notification_preferences(chat_id)
        current = prefs.get('notify_review', False) if prefs else False
        
        # Toggle setting
        db.set_notification_preferences(chat_id, notify_review=not current)
        
        if not current:
            await query.message.reply_text(
                "✅ Learning review notifications enabled!"
            )
        else:
            await query.message.reply_text(
                "❌ Learning review notifications disabled."
            )
    
    elif query.data == "notify_facts":
        # Get current setting
        prefs = db.get_notification_preferences(chat_id)
        current = prefs.get('notify_facts', False) if prefs else False
        
        # Toggle setting
        db.set_notification_preferences(chat_id, notify_facts=not current)
        
        if not current:
            await query.message.reply_text(
                "✅ Language facts notifications enabled!"
            )
        else:
            await query.message.reply_text(
                "❌ Language facts notifications disabled."
            )
    
    # Get updated settings for display
    prefs = db.get_notification_preferences(chat_id)
    
    if prefs:
        notify_daily = "✅" if prefs.get('notify_daily') else "❌"
        notify_review = "✅" if prefs.get('notify_review') else "❌"
        notify_facts = "✅" if prefs.get('notify_facts') else "❌"
        time_str = prefs.get('notification_time', 'Not set')
        
        settings_message = (
            f"🔔 *Your Notification Settings*\n\n"
            f"Daily Word: {notify_daily}\n"
            f"Learning Review: {notify_review}\n"
            f"Language Facts: {notify_facts}\n"
            f"Custom Time: {time_str}\n\n"
            f"Select which notifications to toggle:"
        )
    else:
        settings_message = (
            f"🔔 *Notification Settings*\n\n"
            f"You don't have any notifications set up yet.\n"
            f"Select which notifications to enable:"
        )
    
    await query.edit_message_text(
        settings_message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_notification_keyboard()
    )

async def handle_time_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle time slot selection."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    if query.data == "time_cancel":
        await query.edit_message_text(
            "⏰ Time selection cancelled.",
            reply_markup=get_notification_keyboard()
        )
        return
    
    # Extract time from callback data
    time_str = query.data.replace("time_", "")
    
    # Update database
    db.set_notification_preferences(chat_id, notification_time=time_str)
    
    await query.edit_message_text(
        f"⏰ Notification time set to {time_str}!\n\n"
        f"You'll receive your selected notifications at this time daily.",
        reply_markup=get_notification_keyboard()
    )

def generate_streak_badges(streak_days: int) -> str:
    """Generate visual streak badges based on streak days.
    
    Args:
        streak_days: Number of days in the current streak
        
    Returns:
        Formatted string with streak badges and achievements
    """
    if streak_days <= 0:
        return "No active streak yet. Start translating today!"
    
    # Base badges - always show
    badges = []
    
    # Milestone badge types with corresponding emoji
    milestones = [
        (1, "🔥"),        # 1 day (fire)
        (3, "🔥🔥"),      # 3 days (double fire)
        (7, "🔥🔥🔥"),    # 7 days (triple fire)
        (14, "🌟"),       # 14 days (star)
        (30, "🌟🌟"),     # 30 days (double star)
        (60, "🌟🌟🌟"),   # 60 days (triple star)
        (100, "💎"),      # 100 days (diamond)
        (180, "💎💎"),    # 180 days (double diamond)
        (365, "👑")       # 365 days (crown)
    ]
    
    # Add milestone badges for achieved levels
    for days, badge in milestones:
        if streak_days >= days:
            badges.append(f"{badge} {days}-day streak")
    
    # Special achievements for longer streaks
    if streak_days >= 7:
        badges.append("🏆 Weekly Warrior")
    if streak_days >= 30:
        badges.append("🏅 Monthly Master")
    if streak_days >= 100:
        badges.append("🎖️ Century Club")
    if streak_days >= 365:
        badges.append("👑 Language Royalty")
    
    # Add current streak
    current_streak_display = f"🔥 Current streak: *{streak_days}* days"
    
    # Format everything
    if badges:
        badge_display = "\n".join([f"• {badge}" for badge in badges])
        return f"{current_streak_display}\n\n*Streak Achievements:*\n{badge_display}"
    else:
        return current_streak_display

async def vocabulary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /vocabulary command to extract vocabulary from text."""
    chat_id = update.effective_chat.id
    
    # Check if any AI service is available
    if not is_ai_service_available():
        await update.message.reply_text(
            "📚 *Vocabulary Extraction Unavailable*\n\n"
            "Sorry, AI services are currently unavailable. Please try again later.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Determine the message to analyze
    text_to_analyze = None
    message_origin = "unknown"
    
    # Check if this command was triggered from a button (action_ask_ai)
    if hasattr(update, 'callback_query') and update.callback_query:
        # Get the original message text from the message the button was attached to
        # This should be a translation result
        if update.callback_query.message and update.callback_query.message.text:
            text_to_analyze = update.callback_query.message.text
            message_origin = "bot's translation"

    # If the command was used as a reply to a message
    elif update.message and update.message.reply_to_message:
        # Use the text of the replied-to message
        text_to_analyze = update.message.reply_to_message.text
        message_origin = "replied message"
    
    # If it was triggered via a command and there's a recent message
    elif context.user_data.get('last_translation'):
        # Use the source text from the last translation
        text_to_analyze = context.user_data['last_translation'].get('source_text', '')
        if not text_to_analyze and context.user_data['last_translation'].get('translations'):
            # If source text is empty, use the first available translation
            for lang, trans_data in context.user_data['last_translation'].get('translations', {}).items():
                if trans_data and 'text' in trans_data and trans_data['text']:
                    text_to_analyze = trans_data['text']
                    message_origin = f"translation to {LANGUAGE_NAMES.get(lang, lang)}"
                    break
        else:
            message_origin = "original message"
    
    # If no message found
    if not text_to_analyze:
        await update.message.reply_text(
            "🤔 I couldn't find any text to analyze. Please reply to a message with /askai or use the 'Ask AI' button after a translation."
        )
        return
        
    # Send typing action
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    # Send a temporary message with a loading indicator
    loading_message = await context.bot.send_message(
        chat_id=chat_id,
        text="📚 Extracting vocabulary..."
    )
    
    try:
        # First try using amurex for vocabulary extraction
        try:
            import amurex_ai
            if amurex_ai.is_available():
                # Use amurex to extract vocabulary
                response = amurex_ai.extract_vocabulary(text_to_analyze)
                if response:
                    logger.info("Successfully extracted vocabulary using amurex")
                else:
                    logger.warning("Amurex vocabulary extraction returned empty result")
            else:
                logger.info("Amurex not available for vocabulary extraction, falling back to Claude")
                response = None
        except ImportError:
            logger.warning("Amurex module not found, falling back to standard Claude query")
            response = None
        except Exception as e:
            logger.error(f"Error using amurex for vocabulary extraction: {e}")
            response = None
            
        # Fallback to standard Claude query if amurex failed or isn't available
        if not response:
            # Prepare the prompt for Claude to extract vocabulary
            prompt = f"""
            Please extract one interesting or useful word from the following text. 
            Then provide a thorough definition, synonyms, antonyms (if any), and an explanatory example sentence using the word.
            Format your response like this:
            
            **Word:** [selected word]
            
            **Definition:** [thorough definition]
            
            **Synonyms:** [list of synonyms]
            
            **Antonyms:** [list of antonyms, or "None" if none exist]
            
            **Example:** [clear example sentence using the word]
            
            **Usage Notes:** [any special notes about usage, context, register, etc.]
            
            Text to analyze:
            {text_to_analyze}
            """
            
            # Use Claude for vocabulary extraction
            from ai_services_simplified import query_claude
            response = query_claude(prompt)
        
        if response:
            # Delete the loading message
            await context.bot.delete_message(
                chat_id=chat_id,
                message_id=loading_message.message_id
            )
            
            # Save the AI response for potential translation
            context.user_data['last_ai_response'] = response
            
            # Send the vocabulary analysis
            await update.message.reply_text(
                f"📚 *Vocabulary from {message_origin}:*\n\n{response}",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("💬 Chat with AI", callback_data="chat_ai")]
                ])
            )
        else:
            # Update loading message with error
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_message.message_id,
                text="😓 Sorry, I couldn't extract vocabulary from the message. The AI service might be temporarily unavailable."
            )
    except Exception as e:
        logger.error(f"Error in vocabulary extraction: {e}")
        
        # Update loading message with error
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_message.message_id,
            text=f"😓 Sorry, there was an error extracting vocabulary: {str(e)[:100]}"
        )

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /stats command for viewing user statistics."""
    chat_id = update.effective_chat.id
    
    # Get user statistics from database
    stats = db.get_user_statistics(chat_id)
    
    if not stats:
        await update.message.reply_text(
            "📊 You don't have any learning statistics yet.\n\n"
            "Translate messages and get random words to start building your stats!"
        )
        return
    
    # Format achievement badges - only keep translation and vocabulary related ones
    achievements = []
    for achievement in stats.get('achievements', []):
        achievement_type = achievement.get('achievement_type', '')
        if achievement_type.startswith('vocabulary_'):
            achievements.append("📚")
        elif achievement_type.startswith('translations_'):
            achievements.append("🇨🇳")
        elif achievement_type.startswith('streak_'):
            achievements.append("🔥")
    
    # Get streak days
    streak_days = stats.get('streak', 0)
    
    # Get the translation count
    translations = stats.get('translations_requested', 0)
    
    # Format simplified stats message
    message = (
        f"📊 *Your Learning Statistics*\n\n"
        f"Translations Requested: {translations}\n"
        f"Days Active: {streak_days}\n\n"
    )
    
    # Add streak badges
    streak_message = generate_streak_badges(streak_days)
    message += f"{streak_message}\n\n"
    
    if achievements:
        message += f"*Achievements:* {''.join(achievements[:10])}"
        if len(achievements) > 10:
            message += f" +{len(achievements) - 10} more"
    else:
        message += "*Achievements:* None yet - keep learning!"
    
    # Add a tip about using translation
    message += "\n\n💡 *Tip:* Type any text to translate it to your selected languages instantly"
    
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)

# Removing translate_command function as it's no longer needed
# Translation now happens automatically in handle_text_message

# Updated endchat command that doesn't rely on translate_command
async def translator_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /translator command to switch to translation-only mode."""
    chat_id = update.effective_chat.id
    
    # Set translation mode
    context.user_data['in_chat_mode'] = False
    
    # Update database to translator mode (False = translator mode)
    db.set_chat_mode(chat_id, False)
    
    # Clean up chat history
    if 'chat_history' in context.user_data:
        del context.user_data['chat_history']
    if 'chat_model' in context.user_data:
        del context.user_data['chat_model']
    
    # Get user's selected languages
    user_data = db.get_user(chat_id)
    if not user_data or not user_data.get('selected_languages'):
        selected_langs = DEFAULT_LANGUAGES  # Default to default languages
    else:
        # Parse JSON string from database
        selected_langs = json.loads(user_data['selected_languages'])
    
    # Format language names for display
    lang_names = [f"{get_flag_emoji(lang)} {LANGUAGE_NAMES.get(lang, lang)}" for lang in selected_langs]
    
    # Inform user they are now in translation mode - simplified message
    await update.message.reply_text(
        f"🔄 *Translation Mode*\n\n"
        f"Messages will be translated to: {', '.join(lang_names[:3])}{' and more...' if len(lang_names) > 3 else ''}\n\n"
        f"Type any message to translate it.",
        parse_mode=ParseMode.MARKDOWN
    )
    
async def chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /chat command to switch to direct chat mode with AI."""
    chat_id = update.effective_chat.id
    
    # Check if any AI service is available using shared services
    if not is_ai_available():
        await update.message.reply_text(
            "💬 *Chat Mode Unavailable*\n\n"
            "Sorry, AI services are currently unavailable. Please try again later.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Set chat mode - the system will automatically choose the available service
    context.user_data['in_chat_mode'] = True
    
    # Update the database with the user's preference
    db.set_chat_mode(chat_id, True)
    
    # Create plain keyboard buttons with forward slash for commands
    keyboard = [
        ["/languages", "/flags"],
        ["/chat", "/translator"],
        ["/audio", "/shop"]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    # Try to check if Grok is available first
    try:
        from xai import is_available
        if is_available():
            context.user_data['chat_model'] = 'grok'  # Default to Grok
        else:
            # Use Claude if Grok is not available
            if os.environ.get("ANTHROPIC_API_KEY"):
                context.user_data['chat_model'] = 'claude'
            else:
                context.user_data['chat_model'] = 'grok'  # Default fallback
    except ImportError:
        context.user_data['chat_model'] = 'grok'  # Default fallback
    
    # Initialize chat history if not present
    if 'chat_history' not in context.user_data:
        context.user_data['chat_history'] = []
    
    await update.message.reply_text(
        f"💬 *Chat Mode*\n\n"
        f"Type a message to chat with AI.\n\n"
        f"Press the translator button anytime to return to translation mode.",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /history command for viewing translation history."""
    chat_id = update.effective_chat.id
    
    # Get command arguments if any
    args = context.args
    limit = 5  # Default number of history items to show
    saved_only = False
    
    # Parse arguments if provided
    if args:
        for arg in args:
            if arg.lower() == "saved":
                saved_only = True
            elif arg.isdigit():
                limit = min(int(arg), 10)  # Limit to max 10 entries
    
    # Get translation history from database
    history = db.get_translation_history(chat_id, limit=limit, saved_only=saved_only)
    
    if not history:
        message = "📚 No translation history found."
        if saved_only:
            message += " You haven't saved any translations yet."
        
        await update.message.reply_text(message)
        return
    
    # Create message with history
    message = f"📚 *Your Translation History*{' (Saved Only)' if saved_only else ''}\n\n"
    
    # Format each history entry
    for i, entry in enumerate(history, 1):
        source_text = entry['source_text']
        source_lang = entry['source_language']
        source_lang_name = LANGUAGE_NAMES.get(source_lang, source_lang)
        flag = get_flag_emoji(source_lang)
        created_at = entry['created_at']
        saved = "⭐" if entry['saved'] else ""
        
        # Format date
        try:
            date_str = created_at.strftime("%b %d, %Y %H:%M")
        except:
            date_str = str(created_at)
        
        # Add header for this entry
        message += f"{i}. {saved} {flag} *{source_lang_name}* ({date_str})\n"
        
        # Add original text (shortened if needed)
        if len(source_text) > 50:
            message += f"_{source_text[:50]}..._\n"
        else:
            message += f"_{source_text}_\n"
        
        # Add a few translations
        translations = entry['translations']
        count = 0
        for lang_code, text in translations.items():
            if count >= 2:  # Show only first 2 translations to keep message compact
                break
                
            lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
            flag = get_flag_emoji(lang_code)
            
            if len(text) > 40:
                message += f"  {flag} {lang_name}: {text[:40]}...\n"
            else:
                message += f"  {flag} {lang_name}: {text}\n"
                
            count += 1
            
        # Add a separator between entries
        message += "\n"
    
    # Add footer with instructions
    message += "\nUse `/history saved` to see only saved translations.\n"
    message += "Use `/history 10` to see more items."
    
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    
    # Add buttons to interact with history
    keyboard = [
        [InlineKeyboardButton("🗑️ Clear History", callback_data="action_clear_history")],

    ]
    
    await update.message.reply_text(
        "What would you like to do with your translation history?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages for translation or AI chat."""
    chat_id = update.effective_chat.id
    text = update.message.text
    
    # Skip processing commands (they are handled by other handlers)
    if text.startswith('/'):
        # Skip other commands - they are handled by other handlers
        return
    
    # Check if message starts with ? to force AI chat mode regardless of current setting
    if text.startswith('?'):
        # Force AI chat mode and remove the ? prefix
        context.user_data['in_chat_mode'] = True
        if 'chat_model' not in context.user_data:
            context.user_data['chat_model'] = 'grok'  # Default to Grok for /chat mode
        text = text[1:].strip()
        
        if not text:  # If message was just "?" with nothing else
            await update.message.reply_text(
                "I'm in AI chat mode now. Ask me any language-related question!"
            )
            return
    
    # Default to translation mode unless explicitly set to AI chat mode
    if context.user_data.get('in_chat_mode', False):
        # Check if this is an audio/pronunciation request
        is_audio_request = False
        audio_text = None
        
        # Check if message contains audio-related keywords
        audio_keywords = [
            "pronounce", "pronunciation", "say", "speak", "audio", 
            "listen", "sound", "hear", "pronouncing", "speech",
            "generate audio", "text to speech", "tts", "voice"
        ]
        
        if any(keyword in text.lower() for keyword in audio_keywords):
            is_audio_request = True
            
            # Try to extract the text they want to hear
            import re
            # Look for quoted text
            quoted = re.findall(r'["\']([^"\']+)["\']', text)
            
            if quoted:
                # Take the first quoted string
                audio_text = quoted[0]
            else:
                # Look for text after common phrases
                patterns = [
                    r'(?:pronounce|say|speak|audio for|generate audio for|voice for|speech for|hear) [""]?([^.!?]+)[.!?]?',
                    r'(?:the word|the phrase|the sentence) [""]?([^.!?]+)[.!?]?',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    if matches:
                        audio_text = matches[0].strip()
                        break
                        
            # If we couldn't extract text, use the whole message as the text to pronounce
            if not audio_text and is_audio_request:
                # Remove the audio keywords and use the rest
                cleaned_text = text
                for keyword in audio_keywords:
                    cleaned_text = cleaned_text.replace(keyword, "").strip()
                
                # If there's still text after removing keywords, use it
                if cleaned_text:
                    audio_text = cleaned_text
        
        # Ensure chat_model is set (default to Grok for /chat mode)
        if 'chat_model' not in context.user_data:
            context.user_data['chat_model'] = 'grok'
        model = context.user_data['chat_model']
        chat_history = context.user_data.get('chat_history', [])
        
        # Send typing action
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        
        # Send a temporary message with a loading indicator
        loading_message = await context.bot.send_message(
            chat_id=chat_id,
            text="🤔 Thinking..."
        )
        
        # FULL AI CHAT MODE - Handle as intelligent conversation (using shared services, same as web app)
        try:
            # Add user message to chat history
            chat_history.append({"role": "user", "content": text})
            
            # Respond using shared services (Claude 3.5 Sonnet → Groq fallback, same as web app)
            conversation = "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in chat_history[-10:]])
            ai_response = chat_with_ai(f"Continue this conversation naturally and helpfully:\n\n{conversation}")
            
            if ai_response:
                # Add AI response to history
                chat_history.append({"role": "assistant", "content": ai_response})
                context.user_data['chat_history'] = chat_history[-20:]  # Keep last 20 messages
                
                # Delete loading message and send AI response
                await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
                await update.message.reply_text(ai_response)
                return
            else:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=loading_message.message_id,
                    text="Sorry, I couldn't process your message right now. Please try again."
                )
                return
                
        except Exception as e:
            logger.error(f"AI chat failed: {e}")
            # Fallback handling
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_message.message_id,
                text="I'm having trouble processing your message. Let me try translation mode instead."
            )
        
        # For audio requests in chat mode, we'll handle them specially
        if is_audio_request and audio_text:
            try:
                # First add the user message to history
                chat_history.append({"role": "user", "content": text})
                context.user_data['chat_history'] = chat_history
                
                # Generate audio using shared services (ElevenLabs → gTTS cascade, same as web app)
                audio_file, provider = generate_audio(audio_text, 'en')
                logger.info(f"Audio generated via {provider}")
                
                if audio_file:
                    # Create AI response
                    ai_response = f"Here's the pronunciation of '{audio_text}'. I've generated an audio file for you to listen to."
                    
                    # Add AI response to chat history
                    chat_history.append({"role": "assistant", "content": ai_response})
                    context.user_data['chat_history'] = chat_history
                    
                    # Edit the loading message with the AI response
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=loading_message.message_id,
                        text=ai_response
                    )
                    
                    # Send the audio file as voice message (better for pronunciation)
                    with open(audio_file, 'rb') as audio:
                        await context.bot.send_voice(
                            chat_id=chat_id,
                            voice=audio,
                            caption=f"Pronunciation of: {audio_text}"
                        )
                    
                    # Add switch to translator mode button
                    keyboard = [
                        [InlineKeyboardButton("🔄 Switch to Translation Mode", callback_data="action_switch_translation")]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    # Send a follow-up message with the button
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="Need to translate something else?",
                        reply_markup=reply_markup
                    )
                    return
                
            except Exception as e:
                logger.error(f"Error generating audio: {e}")
                # Continue with normal processing if audio fails
        
        try:
            response = None
            
            # Call appropriate AI service based on selected model
            if model == 'grok':
                # Use Grok
                try:
                    from xai import chat_with_grok
                    
                    # Add message to history for context
                    chat_history.append({"role": "user", "content": text})
                    
                    # Call Grok with the chat history
                    response = chat_with_grok(
                        messages=chat_history,
                        temperature=0.7,
                        max_tokens=2000
                    )
                    
                    if response:
                        # Add response to chat history
                        chat_history.append({"role": "assistant", "content": response})
                    else:
                        response = "Sorry, I couldn't get a response from Grok AI. Please try again later."
                except ImportError:
                    logger.error("Grok module not found")
                    response = "Sorry, Grok AI is not available. Please try using Claude instead."
                except Exception as e:
                    logger.error(f"Error using Grok AI: {e}")
                    response = f"Sorry, there was an error with Grok AI: {str(e)[:100]}"
            
            elif model == 'deepseek':
                # Use DeepSeek
                import os
                import requests
                from urllib.parse import urljoin
                
                server_url = os.environ.get("DEEPSEEK_SERVER_URL")
                if server_url:
                    # Add message to history for context
                    chat_history.append({"role": "user", "content": text})
                    
                    try:
                        response = requests.post(
                            urljoin(server_url, "v1/chat/completions"),
                            json={
                                "model": "deepseek-coder", 
                                "messages": chat_history,
                                "temperature": 0.7
                            },
                            timeout=60
                        ).json()
                        
                        # Extract the response text
                        if response and "choices" in response and len(response["choices"]) > 0:
                            ai_message = response["choices"][0]["message"]["content"]
                            # Add to history
                            chat_history.append({"role": "assistant", "content": ai_message})
                            response = ai_message
                    except Exception as e:
                        logger.error(f"Error with DeepSeek API: {e}")
                        response = f"Sorry, there was an error with the DeepSeek API: {str(e)[:100]}"
                else:
                    response = "DeepSeek API URL is not configured. Please set the DEEPSEEK_SERVER_URL environment variable."
            
            elif model == 'llama':
                # Use Llama
                import os
                import requests
                from urllib.parse import urljoin
                
                server_url = os.environ.get("LLAMA_SERVER_URL")
                if server_url:
                    # Add message to history for context
                    chat_history.append({"role": "user", "content": text})
                    
                    try:
                        response = requests.post(
                            urljoin(server_url, "v1/chat/completions"),
                            json={
                                "model": "llama", 
                                "messages": chat_history,
                                "temperature": 0.7
                            },
                            timeout=60
                        ).json()
                        
                        # Extract the response text
                        if response and "choices" in response and len(response["choices"]) > 0:
                            ai_message = response["choices"][0]["message"]["content"]
                            # Add to history
                            chat_history.append({"role": "assistant", "content": ai_message})
                            response = ai_message
                    except Exception as e:
                        logger.error(f"Error with Llama API: {e}")
                        response = f"Sorry, there was an error with the Llama API: {str(e)[:100]}"
                else:
                    response = "Llama API URL is not configured. Please set the LLAMA_SERVER_URL environment variable."
            
            elif model == 'gpt':
                # Use OpenAI GPT
                import os
                from openai import OpenAI
                
                api_key = os.environ.get("OPENAI_API_KEY")
                if api_key:
                    # Add message to history for context
                    chat_history.append({"role": "user", "content": text})
                    
                    try:
                        client = OpenAI(api_key=api_key)
                        openai_response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=chat_history,
                            temperature=0.7
                        )
                        
                        ai_message = openai_response.choices[0].message.content
                        # Add to history
                        chat_history.append({"role": "assistant", "content": ai_message})
                        response = ai_message
                    except Exception as e:
                        logger.error(f"Error with OpenAI API: {e}")
                        response = f"Sorry, there was an error with the OpenAI API: {str(e)[:100]}"
                else:
                    response = "OpenAI API key is not configured. Please set the OPENAI_API_KEY environment variable."
            
            elif model == 'claude':
                # Use Anthropic Claude with amurex if available
                import os
                
                # First, try to use amurex which provides a simpler interface
                try:
                    # Import amurex integration
                    import amurex_ai
                    
                    # Check if amurex is available
                    if amurex_ai.is_available():
                        # Add message to history for context
                        chat_history.append({"role": "user", "content": text})
                        
                        # Use amurex to chat with Claude
                        ai_message = amurex_ai.chat_with_claude(
                            messages=chat_history,
                            temperature=0.7,
                            max_tokens=1000
                        )
                        
                        if ai_message:
                            # Add to history
                            chat_history.append({"role": "assistant", "content": ai_message})
                            response = ai_message
                        else:
                            logger.warning("Failed to get response from Claude using amurex")
                            response = "Sorry, there was an error communicating with Claude. Please try again."
                            
                        # Skip the fallback to regular Anthropic API if amurex was used successfully
                        if response:
                            # We have a response, no need to try fallback
                            return
                    else:
                        logger.info("Amurex integration not available, falling back to standard Anthropic API")
                except ImportError:
                    logger.warning("Amurex module not available, falling back to standard Anthropic API")
                except Exception as e:
                    logger.error(f"Error using amurex for Claude: {e}")
                
                # Fallback to standard Anthropic API
                try:
                    import anthropic
                    
                    api_key = os.environ.get("ANTHROPIC_API_KEY")
                    if api_key:
                        # Add message to history for context if not already added
                        if not chat_history or chat_history[-1]["role"] != "user" or chat_history[-1]["content"] != text:
                            chat_history.append({"role": "user", "content": text})
                        
                        try:
                            client = anthropic.Anthropic(api_key=api_key)
                            anthropic_messages = []
                            
                            # Convert history to Anthropic format
                            for msg in chat_history:
                                if msg["role"] == "user":
                                    anthropic_messages.append({"role": "user", "content": msg["content"]})
                                elif msg["role"] == "assistant":
                                    anthropic_messages.append({"role": "assistant", "content": msg["content"]})
                            
                            anthropic_response = client.messages.create(
                                model="claude-3-5-sonnet-20241022",
                                messages=anthropic_messages,
                                temperature=0.7,
                                max_tokens=1000
                            )
                            
                            ai_message = anthropic_response.content[0].text
                            # Add to history
                            chat_history.append({"role": "assistant", "content": ai_message})
                            response = ai_message
                        except Exception as e:
                            logger.error(f"Error with Anthropic API: {e}")
                            
                            # Check if this is an overload error (HTTP 529) and try Grok instead
                            if "529" in str(e) or "overloaded" in str(e).lower():
                                logger.info("Claude AI is overloaded, automatically switching to Grok AI")
                                
                                try:
                                    # Try using Grok AI instead
                                    from xai import chat_with_grok
                                    
                                    # Call Grok with the chat history
                                    grok_response = chat_with_grok(
                                        messages=chat_history,
                                        temperature=0.7,
                                        max_tokens=2000
                                    )
                                    
                                    if grok_response:
                                        # Add response to chat history
                                        chat_history.append({"role": "assistant", "content": grok_response})
                                        response = grok_response
                                        
                                        # No need to notify user about the switch
                                    else:
                                        response = "Sorry, the AI service is unavailable. Please try again later."
                                except Exception as grok_err:
                                    logger.error(f"Error using Grok AI as fallback: {grok_err}")
                                    response = "Sorry, the AI service is experiencing issues. Please try again later."
                            else:
                                response = f"Sorry, there was an error with the Anthropic API: {str(e)[:100]}"
                    else:
                        response = "Anthropic API key is not configured. Please set the ANTHROPIC_API_KEY environment variable."
                except ImportError:
                    response = "Anthropic library not installed. Please install it or use amurex."
            
            elif model == 'grok':
                # Use xAI's Grok-2
                import os
                from xai import chat_with_grok, GROK_TEXT_MODEL
                
                api_key = os.environ.get("XAI_API_KEY")
                if api_key:
                    # Add message to history for context
                    chat_history.append({"role": "user", "content": text})
                    
                    try:
                        # Send message to xAI's API
                        ai_message = chat_with_grok(chat_history, model=GROK_TEXT_MODEL)
                        
                        if ai_message:
                            # Add to history
                            chat_history.append({"role": "assistant", "content": ai_message})
                            response = ai_message
                            
                            # Generate audio for the response
                            try:
                                audio_file = generate_audio_for_ai_message(ai_message)
                                if audio_file:
                                    # Need to send audio separately
                                    context.user_data['last_ai_audio'] = audio_file
                                    context.user_data['send_audio_after_message'] = True
                            except Exception as audio_err:
                                logger.error(f"Error generating audio for AI response: {audio_err}")
                        else:
                            response = "Sorry, I couldn't get a response from Grok-2. Please try again or use a different model."
                    except Exception as e:
                        logger.error(f"Error with xAI API: {e}")
                        response = f"Sorry, there was an error with the xAI API: {str(e)[:100]}"
                else:
                    response = "xAI API key is not configured. Please set the XAI_API_KEY environment variable."
            
            elif model == 'huggingface':
                # Use Hugging Face models
                from huggingface_bot import chat_with_model, SUPPORTED_MODELS
                
                # Get the selected HF model ID
                hf_model_id = context.user_data.get('hf_model_id')
                if not hf_model_id or hf_model_id not in SUPPORTED_MODELS:
                    response = "Invalid Hugging Face model selected. Please try again with /chat command."
                else:
                    # Add message to history for context
                    chat_history.append({"role": "user", "content": text})
                    
                    try:
                        # Get the full model ID from the supported models list
                        model_info = SUPPORTED_MODELS[hf_model_id]
                        full_model_id = model_info["id"]
                        model_name = model_info["name"]
                        
                        # Send message to Hugging Face API
                        ai_message = chat_with_model(chat_history, model_id=full_model_id)
                        
                        if ai_message:
                            # Add to history
                            chat_history.append({"role": "assistant", "content": ai_message})
                            response = ai_message
                            
                            # Generate audio for the response
                            try:
                                audio_file = generate_audio_for_ai_message(ai_message)
                                if audio_file:
                                    # Need to send audio separately
                                    context.user_data['last_ai_audio'] = audio_file
                                    context.user_data['send_audio_after_message'] = True
                            except Exception as audio_err:
                                logger.error(f"Error generating audio for AI response: {audio_err}")
                        else:
                            response = f"Sorry, I couldn't get a response from {model_name}. Please try again or use a different model."
                    except Exception as e:
                        logger.error(f"Error with Hugging Face API: {e}")
                        response = f"Sorry, there was an error with the Hugging Face API: {str(e)[:100]}"
            
            else:
                response = "Unknown AI model selected. Please try again with a valid model."
            
            # Update chat history in user data
            context.user_data['chat_history'] = chat_history
            
            # Delete loading message and send the AI response
            await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
            
            if response:
                # Handle message too long
                if len(response) > 4000:
                    # Split message into chunks of max 4000 characters
                    chunks = [response[i:i+4000] for i in range(0, len(response), 4000)]
                    for i, chunk in enumerate(chunks):
                        # For the last chunk, add translator button
                        if i == len(chunks) - 1:
                            # Create keyboard with translator button and rephrase button
                            keyboard = [
                                [InlineKeyboardButton("🔁 Rephrase", callback_data="action_rephrase")],
                                [InlineKeyboardButton("🔄 Switch to Translation Mode", callback_data="action_switch_translation")]
                            ]
                            reply_markup = InlineKeyboardMarkup(keyboard)
                            
                            # Send with translator button
                            sent_message = await update.message.reply_text(
                                chunk,
                                reply_markup=reply_markup
                            )
                            
                            # Save last AI response for translation/audio
                            context.user_data['last_ai_response'] = response
                        else:
                            # Send other chunks without button
                            sent_message = await update.message.reply_text(chunk)
                else:
                    # Create keyboard with translator button, audio button and rephrase button
                    keyboard = [
                        [InlineKeyboardButton("🔄 Switch to Translation Mode", callback_data="action_switch_translation")],
                        [InlineKeyboardButton("🔊 Generate Audio", callback_data="action_audio_ai")],
                        [InlineKeyboardButton("🔁 Rephrase", callback_data="action_rephrase")]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    # Create a message with the response
                    sent_message = await update.message.reply_text(
                        response,
                        reply_markup=reply_markup
                    )
                    
                    # Save last AI response for translation/audio
                    context.user_data['last_ai_response'] = response
                    
                    # Check if we need to send audio for the response
                    if context.user_data.get('send_audio_after_message', False) and 'last_ai_audio' in context.user_data:
                        audio_file = context.user_data['last_ai_audio']
                        if os.path.exists(audio_file):
                            try:
                                with open(audio_file, 'rb') as audio:
                                    await context.bot.send_voice(
                                        chat_id=chat_id,
                                        voice=audio,
                                        caption=f"🎧 AI Response Audio",
                                        reply_to_message_id=sent_message.message_id
                                    )
                                # Delete the temporary file after sending
                                os.remove(audio_file)
                            except Exception as audio_err:
                                logger.error(f"Error sending AI response audio: {audio_err}")
                        
                        # Clear the flags and audio path
                        context.user_data['send_audio_after_message'] = False
                        context.user_data.pop('last_ai_audio', None)
            else:
                await update.message.reply_text("Sorry, I couldn't generate a response. Please try again.")
            
        except Exception as e:
            logger.error(f"Error in chat mode: {e}")
            
            # Delete loading message and send error
            await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
            
            await update.message.reply_text(
                "😓 Sorry, there was an error communicating with the AI service. Please try again later."
            )
        
        return  # Skip the translation part
    
    # Normal translation mode - using shared services (same quality as web app)
    # Get user data from database
    user_data = db.get_user(chat_id)
    
    # Get selected languages
    if user_data and user_data.get('selected_languages'):
        selected_langs = json.loads(user_data['selected_languages'])
    else:
        selected_langs = DEFAULT_LANGUAGES
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    # Send a temporary message with a loading indicator
    loading_message = await context.bot.send_message(
        chat_id=chat_id,
        text="🔄 Translating your message..."
    )
    
    try:
        # Check if AI services are available using shared services
        ai_available = is_ai_available()
        
        # Translation with shared services (same cascade as web app)
        try:
            # Detect the source language
            from langdetect import detect, detect_langs
            source_lang = detect(text)
            confidence = 0.9  # langdetect doesn't provide confidence easily
            
            # Translate to all selected languages using shared services (English first, proper ordering)
            translations_result = translate_to_all_languages(text, selected_langs, source_lang)
            translations_dict = {'translations': translations_result, 'source_lang': source_lang}
            
            # Log the translation structure for debugging
            logger.info(f"Translation structure: {type(translations_dict)}")
            
            # Process the translations
            if isinstance(translations_dict, dict):
                translation_data = translations_dict
            else:
                # If we got an unexpected format, create a simple structure
                logger.warning(f"Unexpected translation format: {type(translations_dict)}")
                translation_data = {
                    "source_text": text,
                    "source_lang": source_lang,
                    "source_lang_name": ALL_LANGUAGES.get(source_lang, source_lang),
                    "translations": {}
                }
                # Add default translations
                for lang in selected_langs:
                    translation_data["translations"][lang] = {
                        "text": f"[Translation error for {lang}]",
                        "transcription": None
                    }
        except Exception as e:
            logger.error(f"Translation failed with error: {e}")
            
            # Display error message to user
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_message.message_id,
                text="⚠️ Translation failed. Please check your internet connection and try again later.\n\n"
                     "💡 You can still use /random to get vocabulary words with Claude AI."
            )
            return
            
        if not translation_data:
            # If translation data is empty or None
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_message.message_id,
                text="⚠️ Translation failed. Please try again with a different text.\n\n"
                     "💡 You can use /random to get vocabulary words with Claude AI."
            )
            return
        
        # Create keyboard with multiple "Hear Audio" buttons for each language
        keyboard = []
        audio_row = []
        
        # Add audio buttons for each translated language
        if 'translations' in translation_data:
            for lang_code in translation_data['translations'].keys():
                flag = get_flag_emoji(lang_code)
                lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
                
                # Add button to current row
                audio_row.append(InlineKeyboardButton(f"🎧 {flag}", callback_data=f"audio_{lang_code}"))
                
                # If we have 3 buttons in a row, start a new row
                if len(audio_row) >= 3:
                    keyboard.append(audio_row)
                    audio_row = []
            
            # Add any remaining buttons in the last row
            if audio_row:
                keyboard.append(audio_row)
        
        # We've removed the AI chat option button as requested
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Store translation data for later use
        if 'translations' not in context.user_data:
            context.user_data['translations'] = {}
        
        context.user_data['last_translation'] = translation_data
        
        # Save translation to database history
        try:
            translations_dict = {}
            # Check if translations key exists in the data
            if 'translations' in translation_data:
                for lang_code, trans_data in translation_data['translations'].items():
                    if isinstance(trans_data, dict) and 'text' in trans_data:
                        translations_dict[lang_code] = trans_data['text']
                    elif isinstance(trans_data, str):
                        translations_dict[lang_code] = trans_data
                    else:
                        translations_dict[lang_code] = str(trans_data)
                
                # Get the source language key
                source_lang_key = 'source_lang'
                if 'source_language' in translation_data:
                    source_lang_key = 'source_language'
                elif 'source_lang' in translation_data:
                    source_lang_key = 'source_lang'
                
                history_id = db.save_translation_history(
                    chat_id=chat_id, 
                    source_text=text, 
                    source_language=translation_data.get(source_lang_key, 'auto'),
                    translations=translations_dict
                )
            else:
                logger.error(f"Missing 'translations' key in translation_data: {translation_data.keys()}")
                history_id = None
        except Exception as e:
            logger.error(f"Error saving translation history: {e}")
            history_id = None
        
        # Store history ID for save action
        if history_id:
            context.user_data['last_history_id'] = history_id
        
        # Delete loading message
        await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
        
        # Only send source text as first message if it's not the same language as one of the selected languages
        # Get source text and language with proper key handling
        source_text = ''
        if 'source_text' in translation_data:
            source_text = translation_data['source_text']
        elif 'original_text' in translation_data:
            source_text = translation_data['original_text']
        
        # Handle source language
        source_lang = None
        if 'source_language' in translation_data:
            source_lang = translation_data['source_language']
        elif 'source_lang' in translation_data:
            source_lang = translation_data['source_lang']
        else:
            # Default to English if no source language is found
            source_lang = 'en'
            logger.warning("No source language found in translation data, defaulting to 'en'")
        
        # Don't send source message separately - it will be handled with other translations
        
        try:
            # Get translations, with error handling
            if 'translations' in translation_data:
                translations = translation_data['translations']
            else:
                logger.error(f"Missing translations key in data: {translation_data.keys()}")
                # Create empty translations to avoid errors
                translations = {}
                await update.message.reply_text("Sorry, I couldn't translate your message. Please try again.")
                return
            
            # Send each translation individually
            for lang_code, trans_data in translations.items():
                # Handle different formats of translation data
                if isinstance(trans_data, dict) and 'text' in trans_data:
                    text = trans_data['text']
                    transcription = trans_data.get('transcription')
                elif isinstance(trans_data, str):
                    text = trans_data
                    transcription = None
                else:
                    text = str(trans_data)
                    transcription = None
                
                flag = get_flag_emoji(lang_code)
                lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
                
                # Check user's flag display preference (default to True if not set)
                show_flags = context.user_data.get('show_flags', True)
        except Exception as e:
            logger.error(f"Error processing translations: {e}")
            await update.message.reply_text("Sorry, there was an error displaying the translations. Please try again.")
            return
            
        # Send each translation
        for lang_code, trans_data in translations.items():
            try:
                # Handle different formats of translation data
                if isinstance(trans_data, dict) and 'text' in trans_data:
                    text = trans_data['text']
                    transcription = trans_data.get('transcription')
                elif isinstance(trans_data, str):
                    text = trans_data
                    transcription = None
                else:
                    text = str(trans_data)
                    transcription = None
                    
                flag = get_flag_emoji(lang_code)
                lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
                
                # Check user's flag display preference
                show_flags = context.user_data.get('show_flags', True)
                
                # Format message (transcriptions already included in text from enhanced_translator)
                if lang_code == source_lang and text == source_text:
                    # For source language, just send the text without language identification
                    trans_message = text
                elif show_flags:
                    # Show flags and language names when enabled
                    trans_message = f"{flag} *{lang_name}:* {text}"
                else:
                    # Just show the translation text without flags/names
                    trans_message = text
                
                # Add pinyin for Chinese translations
                if lang_code == 'zh-CN' and isinstance(trans_data, dict):
                    pinyin = trans_data.get('pinyin')
                    if pinyin:
                        trans_message += f"\n📖 _{pinyin}_"
                
                # Add Latin transcription for Russian
                if lang_code == 'ru' and isinstance(trans_data, dict):
                    latin = trans_data.get('latin')
                    if latin:
                        trans_message += f"\n📖 _{latin}_"
                
                # Only add reply markup to the last translation message
                if list(translations.keys()) and lang_code == list(translations.keys())[-1]:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=trans_message,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=reply_markup
                    )
                else:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=trans_message,
                        parse_mode=ParseMode.MARKDOWN
                    )
            except Exception as e:
                logger.error(f"Error sending translation for {lang_code}: {e}")
                continue
                
        # Update database statistics
        db.update_translation_count(chat_id)
        
        # Add action buttons after all translations are sent
        action_keyboard = [
            [InlineKeyboardButton("🎧 Hear Audio", callback_data="action_audio")],
            [InlineKeyboardButton("💬 Chat with AI", callback_data="chat_ai")]
        ]
        await context.bot.send_message(
            chat_id=chat_id,
            text="💡 Want to do more with this text?",
            reply_markup=InlineKeyboardMarkup(action_keyboard)
        )
        
    except Exception as e:
        logger.error(f"Error translating message: {e}")
        
        # Delete loading message and send error
        await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
        
        await update.message.reply_text(
            "😓 Sorry, I couldn't translate your message. Please try again later."
        )

async def handle_action_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle buttons after translation or other actions."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    # Handle audio_ai action by calling the correct handler
    if query.data == "action_audio_ai":
        await handle_audio_ai_response(update, context)
        return
    
    
    # Extract action type for regular translation actions
    action = query.data.replace("action_", "")
    
    if action == "rephrase":
        # Handle rephrase request by regenerating the AI response
        # First, check if we're in chat mode
        if not context.user_data.get('in_chat_mode', True):
            # Not in chat mode, so we don't have chat history
            await query.message.reply_text(
                "😓 Sorry, I can't rephrase in translation mode. Please use chat mode."
            )
            return
            
        # Check if we have chat history
        chat_history = context.user_data.get('chat_history', [])
        if not chat_history or len(chat_history) < 2:
            await query.message.reply_text(
                "😓 Sorry, I don't have enough context to rephrase. Please start a new conversation."
            )
            return
            
        # Send typing action
        await context.bot.send_chat_action(chat_id=chat_id, action='typing')
        
        # Send a temporary message with a loading indicator
        loading_message = await context.bot.send_message(
            chat_id=chat_id,
            text="🔁 Rephrasing my response..."
        )
        
        try:
            # Get the last user message
            user_messages = [msg for msg in chat_history if msg.get('role') == 'user']
            if not user_messages:
                # No user messages, we can't rephrase
                await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
                await query.message.reply_text(
                    "😓 Sorry, I couldn't find your original message to rephrase my response."
                )
                return
                
            last_user_message = user_messages[-1]
            
            # Remove the last assistant message (the one we're rephrasing)
            if chat_history[-1].get('role') == 'assistant':
                chat_history.pop()
                
            # Get the chat model (default to grok)
            model = context.user_data.get('chat_model', 'grok')
            
            # Generate a new response
            response = None
            
            # Use Grok by default since it's most reliable
            try:
                from xai import chat_with_grok
                
                # Call Grok with the chat history
                response = chat_with_grok(
                    messages=chat_history,
                    temperature=0.9,  # Slightly higher temperature for more variation
                    max_tokens=2000
                )
                
                if response:
                    # Add response to chat history
                    chat_history.append({"role": "assistant", "content": response})
                else:
                    response = "Sorry, I couldn't generate a rephrased response. Please try again."
            except Exception as e:
                logger.error(f"Error rephrasing with Grok AI: {e}")
                response = f"Sorry, there was an error rephrasing: {str(e)[:100]}"
                
            # Update chat history in user data
            context.user_data['chat_history'] = chat_history
            
            # Delete loading message
            await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
            
            if response:
                # Create keyboard with translator button and rephrase button
                keyboard = [
                    [InlineKeyboardButton("🔄 Switch to Translation Mode", callback_data="action_switch_translation")],
                    [InlineKeyboardButton("🔁 /rephrase", callback_data="action_rephrase")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Send the rephrased response
                sent_message = await query.message.reply_text(
                    response,
                    reply_markup=reply_markup
                )
                
                # Save last AI response for translation/audio
                context.user_data['last_ai_response'] = response
            else:
                await query.message.reply_text(
                    "😓 Sorry, I couldn't generate a rephrased response. Please try again."
                )
        except Exception as e:
            logger.error(f"Error in rephrase handler: {e}")
            
            # Delete loading message and send error
            await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
            
            await query.message.reply_text(
                "😓 Sorry, there was an error rephrasing my response. Please try again later."
            )
    
    elif action == "switch_translation":
        # Switch to translation mode (same as /translator command)
        # Set translation mode
        context.user_data['in_chat_mode'] = False
        
        # Clean up chat history
        if 'chat_history' in context.user_data:
            del context.user_data['chat_history']
        if 'chat_model' in context.user_data:
            del context.user_data['chat_model']
        
        # Get user's selected languages
        user_data = db.get_user(chat_id)
        if not user_data or not user_data.get('selected_languages'):
            selected_langs = DEFAULT_LANGUAGES  # Default to default languages
        else:
            # Parse JSON string from database
            selected_langs = json.loads(user_data['selected_languages'])
        
        # Format language names for display
        lang_names = [f"{get_flag_emoji(lang)} {LANGUAGE_NAMES.get(lang, lang)}" for lang in selected_langs]
        
        # Inform user they are now in translation mode
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"🔄 *Switched to Translation Mode*\n\n"
                f"Now I'll automatically translate your messages to your selected languages:\n"
                f"{', '.join(lang_names[:5])}{' and more...' if len(lang_names) > 5 else ''}\n\n"
                f"Type any message to translate it, or use /chat to talk with AI.",
            parse_mode=ParseMode.MARKDOWN
        )
        
    elif action == "audio":
        # Handle audio generation for all languages at once
        if 'last_translation' not in context.user_data:
            await query.message.reply_text(
                "😓 Sorry, I don't have the translation data anymore. Please try translating again."
            )
            return
        
        translation_data = context.user_data['last_translation']
        
        # Send a status message
        status_message = await context.bot.send_message(
            chat_id=chat_id,
            text=f"🎧 Generating audio in all selected languages..."
        )
        
        try:
            success_count = 0
            error_count = 0
            
            # Generate and send audio for all languages in the translation
            for lang_code, trans_data in translation_data['translations'].items():
                text = trans_data['text']
                lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
                flag = get_flag_emoji(lang_code)
                
                try:
                    # Clean text for audio generation (remove transliterations)

                    from transliteration_utils import clean_text_for_audio

                    clean_text = clean_text_for_audio(text)

                    # Generate audio file (returns tuple: path, provider)
                    audio_result = generate_audio(clean_text, lang_code)
                    audio_file = audio_result[0] if isinstance(audio_result, tuple) else audio_result
                    
                    if audio_file:
                        # Send the audio
                        with open(audio_file, 'rb') as audio:
                            await context.bot.send_voice(
                                chat_id=chat_id,
                                voice=audio,
                                caption=f"🎧 {flag} {lang_name}: {text[:50]}{'...' if len(text) > 50 else ''}"
                            )
                        
                        # Delete the temporary file
                        import os
                        if os.path.exists(audio_file):
                            os.remove(audio_file)
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    logger.error(f"Error generating audio for {lang_code}: {e}")
                    error_count += 1
            
            # Delete or update status message
            if success_count > 0:
                # Delete status message if at least one audio was sent
                await context.bot.delete_message(
                    chat_id=chat_id,
                    message_id=status_message.message_id
                )
                
                if error_count > 0:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"⚠️ Note: Failed to generate audio for {error_count} language(s)."
                    )
            else:
                # Update status message with error if all failed
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message.message_id,
                    text=f"😓 Sorry, I couldn't generate audio in any language. Please try again later."
                )
        except Exception as e:
            logger.error(f"Error in audio generation process: {e}")
            
            # Update status message with error
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text=f"😓 Sorry, there was an error generating audio: {str(e)[:50]}"
            )
    
    
    elif action == "learn":
        # Handle adding to learning
        if 'last_translation' not in context.user_data:
            await query.message.reply_text(
                "😓 Sorry, I don't have the translation data anymore. Please try translating again."
            )
            return
        
        translation_data = context.user_data['last_translation']
        source_lang = translation_data['source_language']
        
        # Get original text
        original_text = query.message.text.split('\n')[0]
        if '*From' in original_text:
            original_text = original_text.split('*From')[0].strip()
        
        # Create keyboard with language options
        keyboard = []
        
        for lang_code, trans_data in translation_data['translations'].items():
            if lang_code == source_lang:
                continue
                
            flag = get_flag_emoji(lang_code)
            lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
            
            # Get the translation text
            text = trans_data['text']
            
            # Add button for this language
            keyboard.append([InlineKeyboardButton(
                f"📝 {flag} {lang_name}: {text[:20]}{'...' if len(text) > 20 else ''}",
                callback_data=f"learn_{source_lang}_{lang_code}_{text}"
            )])
        
        await query.message.reply_text(
            "📝 Choose which translation to add to your vocabulary:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif action == "save":
        # Handle saving/favoriting the translation
        if 'last_history_id' not in context.user_data:
            await query.message.reply_text(
                "😓 Sorry, I don't have the translation data anymore. Please try translating again."
            )
            return
            
        history_id = context.user_data['last_history_id']
        
        # Mark as saved in database
        if db.mark_translation_saved(history_id, True):
            # Update the keyboard to show it's saved
            current_keyboard = query.message.reply_markup.inline_keyboard
            
            # Find the save button and replace with "Saved"
            new_keyboard = []
            for row in current_keyboard:
                new_row = []
                for button in row:
                    if button.callback_data == "action_save":
                        new_row.append(InlineKeyboardButton("⭐ Saved", callback_data="action_unsave"))
                    else:
                        new_row.append(button)
                new_keyboard.append(new_row)
                
            # Update the message with new keyboard
            await query.edit_message_reply_markup(
                reply_markup=InlineKeyboardMarkup(new_keyboard)
            )
            
            # Notify user
            await query.answer("Translation saved! Use /history saved to view.")
        else:
            await query.answer("Failed to save translation. Please try again.")
    
    elif action == "unsave":
        # Handle unsaving/unfavoriting the translation
        if 'last_history_id' not in context.user_data:
            await query.message.reply_text(
                "😓 Sorry, I don't have the translation data anymore. Please try translating again."
            )
            return
            
        history_id = context.user_data['last_history_id']
        
        # Mark as unsaved in database
        if db.mark_translation_saved(history_id, False):
            # Update the keyboard to show save option again
            current_keyboard = query.message.reply_markup.inline_keyboard
            
            # Find the saved button and remove it (we no longer have save functionality)
            new_keyboard = []
            for row in current_keyboard:
                new_row = []
                for button in row:
                    if button.callback_data != "action_unsave" and button.callback_data != "action_save":
                        new_row.append(button)
                if new_row:  # Only add the row if it's not empty
                    new_keyboard.append(new_row)
                
            # Update the message with new keyboard only if we have buttons left
            if new_keyboard:
                await query.edit_message_reply_markup(
                    reply_markup=InlineKeyboardMarkup(new_keyboard))
            
            # Notify user
            await query.answer("Translation removed from saved items.")
        else:
            await query.answer("Failed to update translation. Please try again.")
            
    elif action == "view_saved":
        # Create a context with args for saved history
        context.args = ["saved"]
        await history_command(update, context)
        
    elif action == "clear_history":
        # Handle clearing history
        keyboard = [
            [InlineKeyboardButton("✅ Yes, delete all", callback_data="action_confirm_clear")],
            [InlineKeyboardButton("❌ No, keep history", callback_data="action_cancel_clear")]
        ]
        
        await query.message.reply_text(
            "⚠️ Are you sure you want to delete your translation history? This cannot be undone.",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        
    elif action == "confirm_clear":
        # Confirm and clear history
        if db.delete_translation_history(chat_id=chat_id):
            await query.message.reply_text("🗑️ Translation history cleared successfully.")
        else:
            await query.message.reply_text("😓 Error clearing history. Please try again later.")
            
    elif action == "cancel_clear":
        # Cancel clearing history
        await query.message.reply_text("👍 Your translation history is safe.")
    
    elif action == "game":
        # Handle starting a game
        await query.message.reply_text(
            "🎮 Which game would you like to play?",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🎯 Translation Challenge", callback_data="game_challenge")],
                [InlineKeyboardButton("🎲 Vocabulary Match", callback_data="game_vocab_match")],
                [InlineKeyboardButton("📝 Sentence Practice", callback_data="game_sentence")]
            ])
        )


async def handle_audio_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle audio pronunciation buttons."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    # Extract language code or word
    data = query.data.replace("audio_", "")
    
    # Check if this is a daily word audio request
    if data.startswith("daily_"):
        word = data.replace("daily_", "")
        
        # Get user data to get selected languages
        user_data = db.get_user(chat_id)
        
        # Get user's selected languages
        if user_data and user_data.get('selected_languages'):
            selected_langs = json.loads(user_data['selected_languages'])
        else:
            selected_langs = DEFAULT_LANGUAGES
            
        # Send a status message
        status_message = await context.bot.send_message(
            chat_id=chat_id,
            text=f"🎧 Generating audio in all selected languages..."
        )
        
        try:
            success_count = 0
            error_count = 0
            
            # First translate the word to all selected languages
            translations = {}
            
            # Use the translate function from the simplified AI services
            from ai_services_simplified import translate_text_with_fallback
            try:
                # Translate to all selected languages
                translations = translate_text_with_fallback(word, selected_langs)
            except Exception as e:
                logger.error(f"Error translating word for audio: {e}")
                
            # Generate and send audio for all selected languages
            for lang_code in selected_langs:
                lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
                flag = get_flag_emoji(lang_code)
                
                # Use the translated text if available, otherwise use original
                text = translations.get(lang_code, word)
                
                try:
                    # Clean text for audio generation (remove transliterations)

                    from transliteration_utils import clean_text_for_audio

                    clean_text = clean_text_for_audio(text)

                    # Generate audio file (returns tuple: path, provider)
                    audio_result = generate_audio(clean_text, lang_code)
                    audio_file = audio_result[0] if isinstance(audio_result, tuple) else audio_result
                    
                    if audio_file:
                        # Send the audio
                        with open(audio_file, 'rb') as audio:
                            await context.bot.send_voice(
                                chat_id=chat_id,
                                voice=audio,
                                caption=f"🎧 {flag} {lang_name}: {text[:50]}{'...' if len(text) > 50 else ''}"
                            )
                        
                        # Delete the temporary file
                        os.remove(audio_file)
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    logger.error(f"Error generating audio for {lang_code}: {e}")
                    error_count += 1
            
            # Delete or update status message
            if success_count > 0:
                # Delete status message if at least one audio was sent
                await context.bot.delete_message(
                    chat_id=chat_id,
                    message_id=status_message.message_id
                )
                
                if error_count > 0:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=f"⚠️ Note: Failed to generate audio for {error_count} language(s)."
                    )
            else:
                # Update status message with error if all failed
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message.message_id,
                    text=f"😓 Sorry, I couldn't generate audio in any language. Please try again later."
                )
        except Exception as e:
            logger.error(f"Error in audio generation process: {e}")
            
            # Update status message with error
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text=f"😓 Sorry, there was an error generating audio: {str(e)[:50]}"
            )
        return
    
    # Regular translation audio request
    # Check if we have translation data
    if 'last_translation' not in context.user_data:
        await query.message.reply_text(
            "😓 Sorry, I don't have the translation data anymore. Please try translating again."
        )
        return
    
    translation_data = context.user_data['last_translation']
    
    # Get the text for this language
    if data in translation_data['translations']:
        text = translation_data['translations'][data]['text']
        
        # Clean text for audio generation (remove transliterations)
        from transliteration_utils import clean_text_for_audio
        clean_text = clean_text_for_audio(text)
        
        lang_name = LANGUAGE_NAMES.get(data, data)
        flag = get_flag_emoji(data)
        
        # Send a status message
        status_message = await context.bot.send_message(
            chat_id=chat_id,
            text=f"🎧 Generating audio for {flag} {lang_name}..."
        )
        
        try:
            # Generate audio file using clean text (returns tuple: path, provider)
            audio_result = generate_audio(clean_text, data)
            audio_file = audio_result[0] if isinstance(audio_result, tuple) else audio_result
            
            if audio_file and os.path.exists(audio_file):
                # Send the audio
                with open(audio_file, 'rb') as audio:
                    await context.bot.send_voice(
                        chat_id=chat_id,
                        voice=audio,
                        caption=f"🎧 {flag} {lang_name}: {text[:50]}{'...' if len(text) > 50 else ''}"
                    )
                
                # Delete the temporary file
                try:
                    os.remove(audio_file)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup audio file {audio_file}: {cleanup_error}")
            elif audio_file:
                logger.error(f"Audio file generated but not found: {audio_file}")
                raise FileNotFoundError(f"Generated audio file not accessible: {audio_file}")
            else:
                # No audio file generated
                raise Exception("Audio generation returned None")
                
            # Delete status message on success
            await context.bot.delete_message(
                chat_id=chat_id,
                message_id=status_message.message_id
            )
        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            
            # Update status message with error
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text=f"😓 Sorry, there was an error generating audio: {str(e)[:50]}"
            )
    else:
        await query.message.reply_text(
            f"😓 Sorry, I couldn't find the translation for {data}."
        )

async def handle_learn_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle adding words to vocabulary."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    # Get user level from database
    user_data = db.get_user(chat_id)
    level = user_data.get('learning_level', 'advanced') if user_data else 'advanced'
    
    # Parse callback data
    parts = query.data.split('_')
    
    # Check if this is a daily word addition
    if len(parts) == 2 and parts[0] == "learn":
        # Format: learn_word (from daily word feature)
        word = parts[1]
        source_lang = "en"  # Default source language
        target_lang = "en"  # Default target language
        
        # Try to determine better language values from user settings
        if user_data and user_data.get('selected_languages'):
            try:
                selected_langs = json.loads(user_data['selected_languages'])
                if selected_langs and len(selected_langs) > 0:
                    target_lang = selected_langs[0]  # Use first selected language
            except Exception as e:
                logger.error(f"Error parsing selected languages: {e}")
        
        # Add word to database
        vocab_id = db.add_vocabulary(
            chat_id=chat_id,
            source_language=source_lang,
            target_language=target_lang,
            word=word,
            translation='',  # Will be filled by AI
            context=''
        )
        
        # Try to generate example with AI
        if vocab_id and is_ai_service_available():
            try:
                example_data = generate_learning_example(word, target_lang, level)
                
                if example_data:
                    # Update vocabulary entry with AI-generated example
                    translation = example_data.get('translation', '')
                    context = json.dumps({
                        'example': example_data.get('example', ''),
                        'explanation': example_data.get('explanation', '')
                    })
                    db.update_vocabulary_example(vocab_id, translation, context)
            except Exception as e:
                logger.error(f"Error generating learning example: {e}")
        
        # Inform user
        await query.message.reply_text(
            f"📝 Added '{word}' to your vocabulary for study!\n\n"
            f"Use /review to practice your vocabulary with spaced repetition."
        )
        return
    elif len(parts) >= 4:
        # Format: learn_source-lang_target-lang_word
        source_lang = parts[1]
        target_lang = parts[2]
        word = '_'.join(parts[3:])  # Join in case word contains underscores
        
        # Add word to database
        vocab_id = db.add_vocabulary(
            chat_id=chat_id,
            source_language=source_lang,
            target_language=target_lang,
            word=word,
            translation='',  # Will be filled by AI
            context=''
        )
        
        if vocab_id:
            # Try to generate example with AI
            if is_ai_service_available():
                try:
                    example_data = generate_learning_example(word, target_lang, level)
                    
                    if example_data:
                        # Update vocabulary entry with AI-generated example
                        translation = example_data.get('translation', '')
                        context = json.dumps({
                            'example': example_data.get('example', ''),
                            'explanation': example_data.get('explanation', '')
                        })
                        db.update_vocabulary_example(vocab_id, translation, context)
                except Exception as e:
                    logger.error(f"Error generating learning example: {e}")
            
            await query.message.reply_text(
                f"📝 Added to your vocabulary for study!\n\n"
                f"Word: {word}\n"
                f"Use /review to practice your vocabulary with spaced repetition."
            )
        else:
            await query.message.reply_text(
                "😓 Sorry, I couldn't add that word to your vocabulary. Please try again."
            )
    else:
        await query.message.reply_text(
            "😓 Sorry, there was an error processing your request. Please try again."
        )

async def handle_pivot_language_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle pivot language selection (stub function - feature removed)."""
    query = update.callback_query
    await query.answer()
    
    # Function stub - feature removed
    await query.edit_message_text("Back-translation feature has been removed.")


async def handle_chat_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle AI chat model selection callbacks."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    # Extract action from callback data
    data = query.data.replace("chat_", "")
    
    # Only handle cancel action since we simplify to just "AI"
    if data == "cancel":
        await query.edit_message_text("Chat canceled. You can translate messages anytime by typing them.")
        # Ensure we're not in chat mode
        if "in_chat_mode" in context.user_data:
            del context.user_data["in_chat_mode"]
        return
    
    # Handle the "chat_ai" callback without prompting for languages
    if data == "ai":
        # Update chat mode in database
        db.set_chat_mode(chat_id, "ai")
        
        # Default to grok for best performance but this is hidden from user
        context.user_data["chat_model"] = "grok"
        
        await query.edit_message_text(
            f"💬 *AI Chat Mode Activated*\n\n"
            f"Send me any message and I'll respond intelligently. "
            f"You can ask questions, request language learning tips, "
            f"or just have a conversation.\n\n"
            f"Type /translator to go back to translation mode.",
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Change chat state to indicate we're in chat mode
        context.user_data["in_chat_mode"] = True
        
        # Initialize chat history for context
        context.user_data["chat_history"] = []
        return
    
    # For any other chat commands (though we're simplifying now)
    # Default to grok for best performance but this is hidden from user
    context.user_data["chat_model"] = "grok"
    
    await query.edit_message_text(
        f"💬 *Chat Mode*\n\n"
        f"🇺🇸 Type a message to chat with AI\n"
        f"🇪🇸 Escribe un mensaje para chatear con IA\n"
        f"🇵🇹 Digite uma mensagem para conversar com IA\n"
        f"🇮🇹 Scrivi un messaggio per chattare con l'AI\n"
        f"🇫🇷 Écrivez un message pour discuter avec l'IA\n"
        f"🇷🇺 Напишите сообщение для общения с ИИ\n"
        f"🇩🇪 Schreiben Sie eine Nachricht, um mit KI zu chatten\n"
        f"🇨🇳 写一条消息与人工智能聊天\n"
        f"🇰🇷 AI와 채팅할 메시지를 입력하세요\n"
        f"🇯🇵 AIとチャットするメッセージを入力",
        parse_mode=ParseMode.MARKDOWN
    )
    
    # Change chat state to indicate we're in chat mode
    context.user_data["in_chat_mode"] = True
    
    # Initialize chat history for context
    context.user_data["chat_history"] = []

async def handle_game_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle game selection buttons."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    # Get user level from database
    user_data = db.get_user(chat_id)
    level = user_data.get('learning_level', 'advanced') if user_data else 'advanced'
    
    if query.data == "game_challenge":
        # Start translation challenge
        challenge = get_translation_challenge(level)
        
        if not challenge:
            await query.message.reply_text(
                "😓 Sorry, I couldn't create a challenge right now. Please try again later."
            )
            return
        
        # Store challenge data
        context.user_data['current_challenge'] = challenge
        
        # Create message with challenge
        message = (
            f"🎯 *Translation Challenge ({level.capitalize()})*\n\n"
            f"Translate this {challenge['source_language']} phrase to {challenge['target_language']}:\n\n"
            f"*{challenge['phrase']}*\n\n"
            f"Type your answer or use /solution to see the correct translation."
        )
        
        await query.message.reply_text(
            message,
            parse_mode=ParseMode.MARKDOWN
        )
    
    elif query.data == "game_vocab_match":
        # Start vocabulary matching game with user's saved vocabulary
        user_vocab = db.get_vocabulary(chat_id)
        
        if not user_vocab or len(user_vocab) < 4:
            await query.message.reply_text(
                "😓 Sorry, you don't have enough saved vocabulary words for a matching game. "
                "Try using the 'Learn Word' button after translations first."
            )
            return
        
        # Select random words for the game
        game_words = random.sample(user_vocab, 4)
        
        # Extract source and target languages from the first word
        if game_words and 'language' in game_words[0] and 'native_language' in game_words[0]:
            source_lang = game_words[0].get('native_language', 'en')
            target_lang = game_words[0].get('language', 'en')
        else:
            # Default languages if not found
            source_lang = 'en'
            target_lang = 'es'
        
        # Start the matching game with new function
        await start_match_game(chat_id, context, game_words, source_lang, target_lang)
    
    elif query.data == "game_next":
        # Send next game round
        await send_game_round(chat_id, context)
    
    elif query.data == "game_sentence":
        # Start sentence practice
        await query.message.reply_text(
            "📝 Sentence practice is coming soon! Please try one of the other games."
        )

async def start_match_game(chat_id: int, context: ContextTypes.DEFAULT_TYPE, words: List[Dict], source_lang: str, target_lang: str) -> None:
    """Start a vocabulary matching game with words between source and target languages."""
    # Initialize game data
    context.user_data['game_words'] = words
    context.user_data['game_round'] = 0
    context.user_data['game_score'] = 0
    context.user_data['game_source_lang'] = source_lang
    context.user_data['game_target_lang'] = target_lang
    
    # Start the first round
    await send_game_round(chat_id, context)

async def send_game_round(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a single round of the matching game."""
    # Get game data
    words = context.user_data.get('game_words', [])
    round_number = context.user_data.get('game_round', 0)
    source_lang = context.user_data.get('game_source_lang', 'en')
    target_lang = context.user_data.get('game_target_lang', 'en')
    
    if not words or round_number >= len(words):
        # Game is over
        score = context.user_data.get('game_score', 0)
        total = len(words) if words else 0
        
        # Clear game data
        for key in ['game_words', 'game_round', 'game_score', 'game_options', 
                   'game_correct', 'game_source_lang', 'game_target_lang']:
            context.user_data.pop(key, None)
        
        # Update stats in database
        db.update_game_stats(chat_id, correct=(score > 0))
        
        # Create message
        message = (
            f"🎮 *Game Over!*\n\n"
            f"Your score: {score}/{total}\n\n"
        )
        
        if score == total and total > 0:
            message += "🏆 Perfect score! Amazing job!\n\n"
        elif total > 0 and score >= total * 0.8:
            message += "🥇 Excellent work!\n\n"
        elif total > 0 and score >= total * 0.6:
            message += "🥈 Good effort!\n\n"
        else:
            message += "Keep practicing! You'll improve with time.\n\n"
        
        message += "Type /game to play again with different words."
        
        await context.bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Get current word
    word_item = words[round_number]
    
    # Handle different word formats
    if isinstance(word_item, dict):
        if 'word' in word_item:
            question_word = word_item['word']
            correct_answer = word_item.get('translation', '')
        else:
            # Skip invalid items
            context.user_data['game_round'] = round_number + 1
            await send_game_round(chat_id, context)
            return
    else:
        # Skip invalid items
        context.user_data['game_round'] = round_number + 1
        await send_game_round(chat_id, context)
        return
    
    if not question_word or not correct_answer:
        # Skip invalid items
        context.user_data['game_round'] = round_number + 1
        await send_game_round(chat_id, context)
        return
    
    # Create options (1 correct, 3 incorrect)
    options = [correct_answer]
    incorrect_options = []
    
    # Add incorrect options from other words
    other_words = [w for i, w in enumerate(words) if i != round_number]
    random.shuffle(other_words)
    
    for w in other_words[:3]:
        if isinstance(w, dict) and 'translation' in w:
            trans = w.get('translation', '')
            if trans and trans != correct_answer and trans not in incorrect_options:
                incorrect_options.append(trans)
        
        if len(incorrect_options) >= 3:
            break
    
    # If we need more options
    while len(incorrect_options) < 3:
        fake_option = f"Option {len(incorrect_options) + 1}"
        if fake_option not in incorrect_options and fake_option != correct_answer:
            incorrect_options.append(fake_option)
    
    # Add incorrect options and shuffle
    options.extend(incorrect_options)
    random.shuffle(options)
    
    # Store current game state
    context.user_data['game_options'] = options
    context.user_data['game_correct'] = correct_answer
    
    # Create keyboard
    keyboard = []
    for i, option in enumerate(options):
        keyboard.append([InlineKeyboardButton(option, callback_data=f"game_answer_{i}")])
    
    # Add flags and language names for clarity
    source_flag = get_flag_emoji(source_lang)
    source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    
    target_flag = get_flag_emoji(target_lang)
    target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    
    # Create message
    message = (
        f"🎮 *Vocabulary Game - Round {round_number + 1}/{len(words)}*\n\n"
        f"{source_flag} {source_name}: *{question_word}*\n\n"
        f"Select the correct {target_flag} {target_name} translation:"
    )
    
    # Add transcription if available
    transcription = get_transcription(question_word, source_lang)
    if transcription:
        message = message.replace(f"*{question_word}*", f"*{question_word}* _({transcription})_")
    
    await context.bot.send_message(
        chat_id=chat_id,
        text=message,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def send_vocab_game_round(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a round of vocabulary matching game."""
    if 'game_words' not in context.user_data or 'current_word_index' not in context.user_data:
        await context.bot.send_message(
            chat_id=chat_id,
            text="😓 Sorry, there was an error with the game data. Please try starting a new game."
        )
        return
    
    game_words = context.user_data['game_words']
    current_index = context.user_data['current_word_index']
    
    if current_index >= len(game_words):
        # Game is complete
        score = context.user_data['game_score']
        total = len(game_words)
        
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"🎮 Game complete! Your score: {score}/{total}\n\n"
                 f"Play again or try another game?",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🎮 Play Again", callback_data="game_vocab_match")],
                [InlineKeyboardButton("🎯 Try Translation Challenge", callback_data="game_challenge")]
            ])
        )
        return
    
    # Get current word
    current_word = game_words[current_index]
    
    # Generate options (including the correct answer)
    options = [current_word['translation']]
    
    # Add incorrect options
    other_words = [w for w in game_words if w != current_word]
    options.extend([w['translation'] for w in random.sample(other_words, min(3, len(other_words)))])
    
    # Ensure we have 4 options by adding random options if needed
    while len(options) < 4:
        fake_option = f"Option {len(options) + 1}"
        options.append(fake_option)
    
    # Shuffle options
    random.shuffle(options)
    
    # Create keyboard with options
    keyboard = []
    for i, option in enumerate(options):
        keyboard.append([InlineKeyboardButton(
            f"{chr(65 + i)}. {option}",
            callback_data=f"game_answer_{i}_{options.index(current_word['translation'])}"
        )])
    
    # Get transcription if available
    transcription = get_transcription(current_word['word'], current_word['language'])
    transcription_text = f" _({transcription})_" if transcription else ""
    
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"🎮 *Vocabulary Matching Game*\n\n"
             f"What is the meaning of: *{current_word['word']}*{transcription_text}\n\n"
             f"Select the correct translation:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_game_answer(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle game answer buttons."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    # Parse callback data for the new style game answers
    if query.data.startswith("game_answer_"):
        try:
            # Get the selected option index
            option_index = int(query.data.replace("game_answer_", ""))
            
            # Get game data
            options = context.user_data.get('game_options', [])
            correct_answer = context.user_data.get('game_correct', '')
            
            if not options or not correct_answer:
                await query.edit_message_text(
                    "😓 Sorry, there was an error with the game data. Please start a new game with /game."
                )
                return
            
            # Check if answer is correct
            selected_answer = options[option_index] if 0 <= option_index < len(options) else None
            is_correct = selected_answer == correct_answer
            
            # Update score if correct
            if is_correct:
                context.user_data['game_score'] = context.user_data.get('game_score', 0) + 1
            
            # Move to next round
            context.user_data['game_round'] = context.user_data.get('game_round', 0) + 1
            
            # Show feedback
            if is_correct:
                await query.edit_message_text(
                    f"✅ Correct!\n\n"
                    f"Selected: {selected_answer}",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("▶️ Next Question", callback_data="game_next")]
                    ])
                )
            else:
                await query.edit_message_text(
                    f"❌ Not quite right.\n\n"
                    f"You selected: {selected_answer}\n"
                    f"Correct answer: {correct_answer}",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("▶️ Next Question", callback_data="game_next")]
                    ])
                )
                
        except (ValueError, IndexError) as e:
            logger.error(f"Error processing game answer: {e}")
            await query.message.reply_text(
                "😓 Sorry, there was an error processing your answer. Please try again."
            )
            
    # Parse callback data for the old style game answers (backward compatibility)
    elif len(query.data.split('_')) >= 4 and query.data.split('_')[0] == 'game' and query.data.split('_')[1] == 'answer':
        try:
            parts = query.data.split('_')
            selected = int(parts[2])
            correct = int(parts[3])
            
            # Check if answer is correct
            is_correct = selected == correct
            
            # Update game score
            if is_correct:
                context.user_data['game_score'] = context.user_data.get('game_score', 0) + 1
            
            # Get current word
            game_words = context.user_data.get('game_words', [])
            current_index = context.user_data.get('current_word_index', 0)
            
            if game_words and current_index < len(game_words):
                current_word = game_words[current_index]
                
                # Move to next word
                context.user_data['current_word_index'] = current_index + 1
                
                # Create feedback message
                if is_correct:
                    message = f"✅ Correct! {current_word['word']} means {current_word['translation']}."
                else:
                    message = f"❌ Incorrect. {current_word['word']} means {current_word['translation']}."
                
                await query.edit_message_text(
                    message,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("▶️ Next Word", callback_data="game_next")]
                    ])
                )
                
                # Update database statistics
                db.update_game_stats(chat_id, correct=is_correct)
            else:
                await query.message.reply_text(
                    "😓 Sorry, there was an error with the game data. Please try starting a new game."
                )
        except (ValueError, IndexError) as e:
            logger.error(f"Error processing game answer: {e}")
            await query.message.reply_text(
                "😓 Sorry, there was an error processing your answer. Please try again."
            )
            
    # Handle navigation buttons
    elif query.data == "game_next":
        # Check which game type is being played
        if 'game_round' in context.user_data:
            # New style game
            await send_game_round(chat_id, context)
        else:
            # Old style game
            await send_vocab_game_round(chat_id, context)
    elif query.data == "another_daily_word":
        # Random word feature is disabled - inform user
        await context.bot.send_message(
            chat_id=chat_id,
            text="The random word feature has been disabled in this version. Please use translation features instead.",
            parse_mode=ParseMode.MARKDOWN
        )

async def theme_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle the /theme command for thematic learning."""
    chat_id = update.effective_chat.id
    
    await update.message.reply_text(
        "🍎 *Thematic Vocabulary Learning*\n\n"
        "Enter a theme or topic you're interested in learning vocabulary for (e.g., travel, food, technology, business):",
        parse_mode=ParseMode.MARKDOWN
    )
    
    return AWAITING_THEME

async def handle_theme_input(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle theme input for thematic learning."""
    chat_id = update.effective_chat.id
    theme = update.message.text.strip()
    
    # Get user level from database
    user_data = db.get_user(chat_id)
    level = user_data.get('learning_level', 'advanced') if user_data else 'advanced'
    
    # Store theme in user data
    context.user_data['current_theme'] = theme
    
    # Get selected languages from database
    if user_data and user_data.get('selected_languages'):
        selected_langs = json.loads(user_data['selected_languages'])
        # Default to English if no languages are selected
        if not selected_langs:
            selected_langs = ['en']
    else:
        selected_langs = ['en']
    
    # Send loading message
    loading_message = await update.message.reply_text(
        f"🔄 Generating thematic vocabulary for '{theme}' at {level} level..."
    )
    
    try:
        # Check if AI is available
        if not is_ai_service_available():
            await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
            await update.message.reply_text(
                "😓 Sorry, thematic vocabulary generation requires AI services which are not available right now. "
                "Please try again later."
            )
            return ConversationHandler.END
        
        # Choose a random language from selected languages
        target_lang = random.choice(selected_langs)
        
        # Generate vocabulary with AI
        vocab_list = generate_thematic_vocabulary(theme, target_lang, level, count=5)
        
        # Delete loading message
        await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
        
        if not vocab_list:
            await update.message.reply_text(
                f"😓 Sorry, I couldn't generate vocabulary for '{theme}'. Please try a different theme."
            )
            return ConversationHandler.END
        
        # Format message with vocabulary
        message = f"🍎 *Thematic Vocabulary: {theme.capitalize()} ({level.capitalize()})*\n\n"
        
        for i, item in enumerate(vocab_list, 1):
            word = item.get('word', '')
            translation = item.get('translation', '')
            example = item.get('example', '')
            
            # Get transcription if available
            transcription = get_transcription(word, target_lang)
            transcription_text = f" _({transcription})_" if transcription else ""
            
            message += f"{i}. *{word}*{transcription_text}\n"
            message += f"   {translation}\n"
            if example:
                message += f"   Example: _{example}_\n"
            message += "\n"
        
        # Add option to save words
        message += "Would you like to save these words to your vocabulary for practice?"
        
        await update.message.reply_text(
            message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📝 Save to Vocabulary", callback_data=f"save_theme_{theme}")],
                [InlineKeyboardButton("🔄 Try Another Theme", callback_data="another_theme")]
            ])
        )
        
        # Store vocabulary list in user data for saving later
        context.user_data['theme_vocab'] = {
            'theme': theme,
            'level': level,
            'language': target_lang,
            'items': vocab_list
        }
        
        return ConversationHandler.END
        
    except Exception as e:
        logger.error(f"Error generating thematic vocabulary: {e}")
        
        # Delete loading message
        await context.bot.delete_message(chat_id=chat_id, message_id=loading_message.message_id)
        
        await update.message.reply_text(
            f"😓 Sorry, there was an error generating vocabulary for '{theme}': {str(e)[:50]}"
        )
        return ConversationHandler.END

async def handle_theme_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle buttons related to thematic learning."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    if query.data == "another_theme":
        await query.message.reply_text(
            "🍎 Enter another theme or topic you're interested in learning vocabulary for:"
        )
        
        # Add a new handler for this conversation
        application = context.application
        application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_theme_input),
            group=1001  # Use a different group to avoid conflicts
        )
    
    elif query.data.startswith("save_theme_"):
        # Save theme vocabulary to database
        if 'theme_vocab' not in context.user_data:
            await query.message.reply_text(
                "😓 Sorry, the vocabulary data is no longer available. Please try generating again."
            )
            return
        
        vocab_data = context.user_data['theme_vocab']
        items = vocab_data.get('items', [])
        
        # Add each word to database
        added_count = 0
        
        for item in items:
            word = item.get('word', '')
            translation = item.get('translation', '')
            example = item.get('example', '')
            
            if word and translation:
                # Add to database
                vocab_id = db.add_vocabulary(
                    chat_id=chat_id,
                    source_language='en',  # Assuming English translation
                    target_language=vocab_data.get('language', 'en'),
                    word=word,
                    translation=translation,
                    context=json.dumps({'example': example})
                )
                
                if vocab_id:
                    added_count += 1
        
        await query.message.reply_text(
            f"✅ Added {added_count} words to your vocabulary!\n\n"
            f"Use /review to practice these words with spaced repetition."
        )

async def game_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /game command for the matching game."""
    chat_id = update.effective_chat.id
    
    # Get user data from database
    user_data = db.get_user(chat_id)
    
    # Create language pair options keyboard
    await update.message.reply_text(
        "🎮 Select language pairs for the game:",
        reply_markup=get_language_pair_keyboard(user_data)
    )

def get_language_pair_keyboard(user_data):
    """Generate a keyboard for language pair selection."""
    keyboard = []
    
    # Get user's selected languages
    if user_data and user_data.get('selected_languages'):
        selected_langs = json.loads(user_data['selected_languages'])
    else:
        selected_langs = ALL_LANGUAGE_CODES[:6]  # Limit to 6 languages for the keyboard
    
    # Always include English
    if 'en' not in selected_langs:
        selected_langs.insert(0, 'en')
    
    # Create rows with common pairs
    for source_lang in selected_langs[:3]:  # Limit source languages to avoid huge keyboard
        row = []
        for target_lang in selected_langs[:4]:  # Limit target languages per row
            if source_lang != target_lang:
                source_flag = get_flag_emoji(source_lang)
                target_flag = get_flag_emoji(target_lang)
                btn_text = f"{source_flag} → {target_flag}"
                callback_data = f"game_pair_{source_lang}_{target_lang}"
                row.append(InlineKeyboardButton(btn_text, callback_data=callback_data))
        
        if row:  # Only add non-empty rows
            keyboard.append(row)
    
    # Add a "Custom Pair" option
    keyboard.append([InlineKeyboardButton("🔍 Custom Pair", callback_data="game_custom_pair")])
    
    return InlineKeyboardMarkup(keyboard)

async def handle_translate_ai_response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle translating an AI response to selected languages."""
    query = update.callback_query
    await query.answer()  # Acknowledge the button press immediately
    chat_id = query.message.chat_id
    
    # First try to get the AI message from context.user_data
    ai_response = context.user_data.get('last_ai_response', None)
    
    # If we don't have it there, use the message text that has the button
    if not ai_response:
        ai_response = query.message.text
    
    # Check if AI message exists and isn't empty
    if not ai_response or ai_response.strip() == "":
        await query.message.reply_text(
            "😓 Sorry, I don't have the AI message text anymore. Please get a new response."
        )
        return
    
    # Send a status message for translation
    status_message = await context.bot.send_message(
        chat_id=chat_id,
        text="🔄 Translating AI response..."
    )
    
    # Store original chat mode so we can restore it later
    original_chat_mode = context.user_data.get('in_chat_mode', True)
    
    # Temporarily switch to translation mode
    context.user_data['in_chat_mode'] = False
    
    # Create a temporary update object with the AI response text
    from telegram import Message
    temp_update = Update(update.callback_query.id, message=Message(
        message_id=query.message.message_id,
        date=query.message.date,
        chat=query.message.chat,
        text=ai_response,
        from_user=query.from_user,
    ))
    
    try:
        # Process the AI message as a text message for translation
        await handle_text_message(temp_update, context)
        
        # Delete the status message since handle_text_message will send its own message
        await context.bot.delete_message(
            chat_id=chat_id,
            message_id=status_message.message_id
        )
    except Exception as e:
        logger.error(f"Error translating AI response: {e}")
        
        # Update loading message with error
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message.message_id,
            text="😓 Sorry, I couldn't translate the AI response. Please try again later."
        )
    finally:
        # Restore original chat mode
        context.user_data['in_chat_mode'] = original_chat_mode

async def handle_audio_ai_response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle generating audio for an AI response - works exactly like /audio command."""
    query = update.callback_query
    await query.answer()  # Acknowledge the button press immediately
    chat_id = query.message.chat_id
    
    # Check if we have a stored AI response
    if 'last_ai_response' not in context.user_data:
        await context.bot.send_message(
            chat_id=chat_id,
            text="Sorry, I don't have the AI response data anymore."
        )
        return
    
    # Get the AI response text
    ai_response = context.user_data['last_ai_response']
    
    # Use the EXACT same logic as /audio command
    # Send a loading message
    loading_message = await context.bot.send_message(
        chat_id=chat_id,
        text="🔍 Generating audio for AI response..."
    )
    
    try:
        # Get user's selected languages
        user_data = db.get_user(chat_id)
        if not user_data or not user_data.get('selected_languages'):
            selected_langs = DEFAULT_LANGUAGES
        else:
            selected_langs = json.loads(user_data['selected_languages'])
        
        # Update loading message
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_message.message_id,
            text=f"🔊 Generating audio in all selected languages..."
        )
        
        # Generate and send audio for each selected language
        success_count = 0
        error_count = 0
        
        for lang_code in selected_langs:
            lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
            flag = get_flag_emoji(lang_code)
            
            try:
                # Generate audio file (returns tuple: path, provider)
                audio_result = generate_audio(ai_response, lang_code)
                audio_file = audio_result[0] if isinstance(audio_result, tuple) else audio_result
                
                if audio_file:
                    # Send the audio
                    with open(audio_file, 'rb') as audio:
                        await context.bot.send_voice(
                            chat_id=chat_id,
                            voice=audio,
                            caption=f"🎧 {flag} {lang_name}: {ai_response[:50]}{'...' if len(ai_response) > 50 else ''}"
                        )
                    
                    # Delete the temporary file
                    import os
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                logger.error(f"Error generating audio for {lang_code}: {e}")
                error_count += 1
        
        # Delete or update status message
        if success_count > 0:
            # Delete status message if at least one audio was sent
            await context.bot.delete_message(
                chat_id=chat_id,
                message_id=loading_message.message_id
            )
            
            if error_count > 0:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"⚠️ Note: Failed to generate audio for {error_count} language(s)."
                )
        else:
            # Update status message with error if all failed
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=loading_message.message_id,
                text=f"😓 Sorry, I couldn't generate audio in any language. Please try again later."
            )
    except Exception as e:
        logger.error(f"Error in AI audio generation: {e}")
        # Update status message with error
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=loading_message.message_id,
            text=f"😓 Sorry, there was an error generating audio: {str(e)[:50]}"
        )
        if os.path.exists(audio_file):
            os.remove(audio_file)
        
        # Delete status message
        await context.bot.delete_message(
            chat_id=chat_id,
            message_id=status_message.message_id
        )
    except Exception as e:
        logger.error(f"Error generating audio for AI response: {e}")
        
        # Update status message with error
        await context.bot.edit_message_text(
            chat_id=chat_id,
            message_id=status_message.message_id,
            text=f"😓 Sorry, there was an error generating audio. Please try again later."
        )

async def handle_game_pair_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle game pair selection."""
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    
    if not query.data.startswith("game_pair_"):
        return
    
    # Extract language codes
    parts = query.data.split("_")
    if len(parts) >= 3:
        source_lang = parts[2]
        target_lang = parts[3]
        
        # Start a matching game with these languages
        await start_vocabulary_game(chat_id, context, source_lang, target_lang)
        
async def start_vocabulary_game(chat_id: int, context: ContextTypes.DEFAULT_TYPE, source_lang: str, target_lang: str, vocabulary=None) -> None:
    """Start a vocabulary matching game with specified languages.
    
    Args:
        chat_id: The chat ID
        context: The context
        source_lang: Source language code
        target_lang: Target language code
        vocabulary: Optional pre-filtered vocabulary (if None, will be fetched from database)
    """
    # Get user data from database
    user_data = db.get_user(chat_id)
    
    # Get vocabulary for the specified languages if not already provided
    if vocabulary is None:
        vocabulary = db.get_vocabulary(chat_id, target_language=target_lang)
    
    # Check if we have enough vocabulary
    if not vocabulary or len(vocabulary) < 4:
        # Try generating with AI if available
        if is_ai_service_available():
            try:
                # Get user's learning level
                user_data = db.get_user(chat_id)
                level = user_data.get('learning_level', 'intermediate') if user_data else 'intermediate'
                
                # Choose a random theme
                themes = ["travel", "food", "family", "nature", "technology", "business"]
                theme = random.choice(themes)
                
                # Generate vocabulary with AI
                words = generate_thematic_vocabulary(theme, target_lang, level, count=6)
                
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=f"🎮 Let's play the vocabulary matching game!\n\n"
                         f"I've created a game with {len(words)} {theme}-related words in {LANGUAGE_NAMES.get(target_lang, target_lang)}.\n\n"
                         f"Match each word with its correct translation."
                )
                
                # Start the game with AI-generated words
                await start_match_game(chat_id, context, words, source_lang, target_lang)
                return
            
            except Exception as e:
                logger.error(f"Error generating AI vocabulary for game: {e}")
        
        # If AI fails or not available, send error message
        await context.bot.send_message(
            chat_id=chat_id,
            text="😓 You don't have enough vocabulary words saved yet to play the game.\n\n"
                 "Try using the 'Learn Word' button after translations, or the /theme command to build your vocabulary first."
        )
        return
    
    # Group vocabulary by language pairs
    language_pairs = {}
    for item in vocabulary:
        source_lang = item.get('source_language')
        target_lang = item.get('target_language')
        
        if not source_lang or not target_lang:
            continue
            
        pair_key = f"{source_lang}_{target_lang}"
        if pair_key not in language_pairs:
            language_pairs[pair_key] = []
            
        language_pairs[pair_key].append(item)
    
    # Check if we have enough words for any language pair
    valid_pairs = {k: v for k, v in language_pairs.items() if len(v) >= 4}
    
    if not valid_pairs:
        await context.bot.send_message(
            chat_id=chat_id,
            text="😓 You don't have enough vocabulary words in any single language pair for a game.\n\n"
                 "A minimum of 4 words in the same language pair is required. Try using the 'Learn Word' button after translations to build your vocabulary."
        )
        return
    
    # Choose a random language pair that has enough words
    pair_key = random.choice(list(valid_pairs.keys()))
    words = valid_pairs[pair_key]
    
    # Extract source and target languages
    source_lang, target_lang = pair_key.split('_')
    
    # Start the matching game
    await context.bot.send_message(
        chat_id=chat_id,
        text=f"🎮 Let's play the vocabulary matching game!\n\n"
             f"I'll show you words in {LANGUAGE_NAMES.get(source_lang, source_lang)} and {LANGUAGE_NAMES.get(target_lang, target_lang)}.\n\n"
             f"Match each word with its correct translation."
    )
    
    await start_match_game(chat_id, context, words, source_lang, target_lang)

# Review functionality removed as requested

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the dispatcher."""
    logger.error(f"Exception while handling an update: {context.error}")
    
    # Send error message to user if possible
    if update and update.effective_chat:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="😓 Sorry, an error occurred while processing your request. Please try again later."
        )

async def flags_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Toggle displaying language flags and names in translations."""
    chat_id = update.effective_chat.id
    
    # Toggle the show_flags setting (default is True/ON)
    current_setting = context.user_data.get('show_flags', True)  # Default to True if not set
    new_setting = not current_setting
    context.user_data['show_flags'] = new_setting
    
    # Save the setting to database
    db.set_user_preference(chat_id, 'show_flags', new_setting)
    
    # Inform the user about the change
    if new_setting:
        await update.message.reply_text(
            "🏳️ *Language flags and names are now ON*\n\n"
            "Translations will include language flags and names like:\n"
            "🇪🇸 *Spanish:* Hola mundo",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await update.message.reply_text(
            "🏴 *Language flags and names are now OFF*\n\n"
            "Translations will be displayed without language identification:\n"
            "Hola mundo",
            parse_mode=ParseMode.MARKDOWN
        )

async def shop_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Open the marketplace shop."""
    # Create inline keyboard with the shop button
    keyboard = [
        [InlineKeyboardButton("🛍️ Open Cyber Market", url="https://t.me/QRQODESBOT")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🛍️ *Welcome to Cyber Market!*\n\n"
        "Find premium personalised QR-coded hoodies & t-shirts, book an online language teacher with a first demo class FREE! "
        "Or just collect some cyberpunk art 🖼️ to support our project 💚\n\n"
        "Click the button below to browse our offerings:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )

async def webapp_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Open the web version of UNI LINGUS."""
    keyboard = [
        [InlineKeyboardButton("🌐 Open Web App", url="https://e715b3eb-020b-4929-b7ef-4e4dcac24fef-00-2kf8za2qvhhgm.kirk.replit.dev")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🌐 *UNI LINGUS Web App*\n\n"
        "Access the full web version with:\n"
        "• Advanced translation features\n"
        "• Word definitions (click any word)\n"
        "• Voice synthesis\n"
        "• Full language support\n\n"
        "Click the button below to open:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=reply_markup
    )


async def handle_asr_model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle ASR model selection callbacks."""
    try:
        from gradio_models import GRADIO_ASR_MODELS
        
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        chat_id = query.message.chat_id
        
        if query.data == "asr_cancel":
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=query.message.message_id,
                text="❌ ASR model selection cancelled."
            )
            return
        
        elif query.data == "gradio_test_all":
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=query.message.message_id,
                text="🔬 **Speech Recognition**\n\n"
                     "Send a voice message now for transcription.\n\n"
                     "🎙️ **Ready for your voice message...**",
                parse_mode='Markdown'
            )
            context.user_data['test_all_gradio_models'] = True
            return
            
        elif query.data == "asr_comparison":
            comparison_text = "📊 **Detailed Model Comparison:**\n\n"
            
            for model_id, model_info in ASR_MODELS.items():
                comparison_text += f"**{model_info['name']}**\n"
                comparison_text += f"💪 Strength: {model_info['strength']}\n"
                comparison_text += f"🇨🇳 Languages: {', '.join(model_info['languages'])}\n"
                comparison_text += f"📝 {model_info['description']}\n\n"
            
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=query.message.message_id,
                text=comparison_text[:4000]
            )
            return
        
        elif query.data.startswith("asr_select_"):
            model_id = query.data.replace("asr_select_", "")
            
            if model_id in ASR_MODELS:
                asr_manager.set_user_preference(user_id, model_id)
                model_info = ASR_MODELS[model_id]
                
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=query.message.message_id,
                    text=f"✅ **Model Selected Successfully!**\n\n"
                         f"**Your new voice recognition model:**\n"
                         f"{model_info['name']}\n\n"
                         f"💪 **Strength:** {model_info['strength']}\n"
                         f"🇨🇳 **Languages:** {', '.join(model_info['languages'])}\n"
                         f"📝 **Description:** {model_info['description']}\n\n"
                         f"🎙️ **Try sending a voice message to test your new model!**"
                )
            else:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=query.message.message_id,
                    text="❌ Invalid model selection. Please try again."
                )
        
    except Exception as e:
        logger.error(f"ASR callback failed: {e}")
        await context.bot.edit_message_text(
            chat_id=query.message.chat_id,
            message_id=query.message.message_id,
            text="❌ Model selection failed. Please try again."
        )


def main() -> None:
    """Start the enhanced bot."""
    # Set up the application with persistence
    persistence = PicklePersistence(filepath="bot_data.pickle")
    
    application = Application.builder().token(TOKEN).persistence(persistence).build()
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    # Help command removed as requested
    
    # Adding only the stats command
    application.add_handler(CommandHandler("stats", stats_command))
    # History command is registered but hidden from the command menu
    application.add_handler(CommandHandler("history", history_command))
    
    # Only keeping the AI and chat command handlers, removing translate command
    application.add_handler(CommandHandler("chat", chat_command))
    application.add_handler(CommandHandler("translator", translator_command))
    application.add_handler(CommandHandler("audio", audio_command))
    application.add_handler(CommandHandler("flags", flags_command))
    application.add_handler(CommandHandler("shop", shop_command))
    application.add_handler(CommandHandler("webapp", webapp_command))

    
    # Advanced AI feature commands removed as requested
    
    # Register the advanced features from our new module
    try:
        from advanced_features import register_advanced_features
        from huggingface_asr_models import asr_manager, get_asr_model_keyboard, transcribe_with_selected_model, ASR_MODELS
        register_advanced_features(application)
        logger.info("Advanced features successfully registered")
    except ImportError:
        logger.warning("Advanced features module not available")
    except Exception as e:
        logger.error(f"Error registering advanced features: {e}")
    
    # Add language selection conversation - only process toggle_ callbacks
    application.add_handler(ConversationHandler(
        entry_points=[CommandHandler("languages", languages_command)],
        states={
            AWAITING_LANGUAGE_SELECTION: [CallbackQueryHandler(handle_language_selection, pattern=r"^toggle_")]
        },
        fallbacks=[
            CommandHandler("cancel", lambda u, c: ConversationHandler.END),
            CommandHandler("languages", languages_command)  # Allow restarting the language selection
        ]
    ))
    
    # Removed level and theme conversation handlers
    
    # Removed settings command as requested
    
    # Add callback query handlers
    application.add_handler(CallbackQueryHandler(handle_audio_button, pattern=r"^audio_"))
    application.add_handler(CallbackQueryHandler(handle_learn_button, pattern=r"^learn_"))
    # Removed game and theme handlers
    application.add_handler(CallbackQueryHandler(handle_chat_callback, pattern=r"^chat_"))
    application.add_handler(CallbackQueryHandler(handle_time_selection, pattern=r"^time_"))
    application.add_handler(CallbackQueryHandler(handle_notifications_selection, pattern=r"^notify_"))

    application.add_handler(CallbackQueryHandler(handle_action_button, pattern=r"^action_"))

    # Removed game answer handler
    
    # Add text message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    # Add voice message handler for audio transcription and translation
    # Define the voice message handler function - simplified to just show a message
    async def handle_voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Use Google Speech Recognition for reliable voice transcription"""
        
        voice = update.message.voice
        chat_id = update.effective_chat.id
        
        # Send processing message
        status_message = await update.message.reply_text(
            "🎤 Processing your voice message..."
        )
        
        try:
            # Create audio directory if it doesn't exist
            import os
            if not os.path.exists("audio"):
                os.makedirs("audio")
                
            # Get file from Telegram
            voice_file = await context.bot.get_file(voice.file_id)
            
            # Create a temporary file with unique name to save the voice message
            import uuid
            unique_id = uuid.uuid4().hex[:8]
            voice_path = f"audio/voice_{unique_id}_{voice.file_id[-8:]}.ogg"
            await voice_file.download_to_drive(voice_path)
            
            # Determine the possible languages from user settings
            user_data = db.get_user(chat_id)
            language_candidates = ['en']  # Default to English as fallback
            
            if user_data and user_data.get('selected_languages'):
                selected_langs = json.loads(user_data['selected_languages'])
                if selected_langs and len(selected_langs) > 0:
                    # Use all selected languages as candidates, with the first one as primary
                    language_candidates = selected_langs
            
            # Use Google Speech Recognition for transcription
            transcription = None
            
            try:
                # Use Google Speech Recognition as primary method (reliable and fast)
                import speech_recognition as sr
                from pydub import AudioSegment
                
                user_id = update.effective_user.id
                language_hint = language_candidates[0] if language_candidates else 'en'
                
                logger.info(f"Using Google Speech Recognition for transcription...")
                
                # Convert OGG to WAV for speech recognition
                audio = AudioSegment.from_ogg(voice_path)
                wav_path = voice_path.replace('.ogg', '.wav')
                audio.export(wav_path, format="wav")
                
                # Use speech recognition
                r = sr.Recognizer()
                with sr.AudioFile(wav_path) as source:
                    audio_data = r.record(source)
                
                # Try Google Speech Recognition
                try:
                    text = r.recognize_google(audio_data, language=language_hint)
                    transcription = text
                    logger.info(f"Google Speech Recognition successful: {transcription}")
                except sr.UnknownValueError:
                    logger.warning("Google Speech Recognition could not understand audio")
                    transcription = None
                except sr.RequestError as e:
                    logger.error(f"Google Speech Recognition error: {e}")
                    transcription = None
                
            except Exception as e:
                logger.error(f"Voice transcription failed: {e}")
                transcription = None
            
            if transcription:
                # Check if user is in chat mode
                user_data = db.get_user(chat_id)
                chat_mode = False
                if user_data and 'chat_mode' in user_data and user_data['chat_mode']:
                    chat_mode = True
                
                if chat_mode:
                    # In chat mode - send to AI
                    from ai_services import translate_with_openai, translate_with_anthropic
                    
                    ai_response = None
                    try:
                        ai_response = translate_with_anthropic(f"Respond naturally to this voice message: {transcription}", "English")
                        if not ai_response:
                            ai_response = translate_with_openai(f"Respond naturally to this voice message: {transcription}", "English")
                    except Exception as e:
                        logger.error(f"AI response failed: {e}")
                        ai_response = f"I heard: '{transcription}'"
                    
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message.message_id,
                        text=f"🎤 Voice: {transcription}\n\n🤖 AI: {ai_response or 'Sorry, I could not generate a response.'}"
                    )
                else:
                    # Translation mode - translate the transcribed text
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message.message_id,
                        text=f"🎤 Transcribed: {transcription}\n\n📝 Translating..."
                    )
                    
                    # Translate the transcribed text directly
                    user_data = db.get_user(chat_id)
                    selected_languages = []
                    if user_data and user_data.get('selected_languages'):
                        selected_languages = json.loads(user_data['selected_languages'])
                    
                    if selected_languages:
                        # Use shared services (same quality as web app)
                        translations_data = translate_to_all_languages(transcription, selected_languages)
                        translations = {'translations': translations_data}
                        
                        # Format and send translation with transliteration support
                        if translations_data:
                            # Detect language for display
                            from langdetect import detect
                            detected_lang = detect(transcription)
                            transcription_display = transcription
                            
                            translation_text = f"🎤 **Voice ({detected_lang.upper()}):** {transcription_display}\n\n"
                            for lang_code in selected_languages:
                                if lang_code in translations_data:
                                    trans_data = translations_data[lang_code]
                                    if isinstance(trans_data, dict) and 'text' in trans_data:
                                        translation = trans_data['text']
                                    else:
                                        translation = str(trans_data)
                                    
                                    if translation and translation != transcription:
                                        flag = get_flag_emoji(lang_code)
                                        # Use translation as-is (transcriptions already included)
                                        translation_display = translation
                                        translation_text += f"{flag} **{lang_code.upper()}:** {translation_display}\n"
                            
                            await context.bot.edit_message_text(
                                chat_id=chat_id,
                                message_id=status_message.message_id,
                                text=translation_text,
                                parse_mode='Markdown'
                            )
                        else:
                            await context.bot.edit_message_text(
                                chat_id=chat_id,
                                message_id=status_message.message_id,
                                text=f"🎤 **Transcribed:** {transcription}\n\n❌ Translation failed. Please try again."
                            )
                    else:
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=status_message.message_id,
                            text=f"🎤 **Transcribed:** {transcription}\n\n💡 Use /languages to select target languages for translation."
                        )
            else:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message.message_id,
                    text="❌ Sorry, I couldn't understand your voice message. Please try again or send a text message."
                )
            
            # Clean up audio files
            try:
                if os.path.exists(voice_path):
                    os.remove(voice_path)
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error processing voice message: {e}")
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_message.message_id,
                text="❌ Sorry, there was an error processing your voice message. Please try again."
            )

    async def handle_voice_message_old(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle voice messages with transcription."""
        chat_id = update.effective_chat.id
        voice = update.message.voice
        voice_path = None
        
        # Send a status message
        status_message = await update.message.reply_text(
            "🎤 Processing your voice message..."
        )
        
        try:
            # Create audio directory if it doesn't exist
            import os
            if not os.path.exists("audio"):
                os.makedirs("audio")
                
            # Get file from Telegram
            voice_file = await context.bot.get_file(voice.file_id)
            
            # Create a temporary file with unique name to save the voice message
            import uuid
            unique_id = uuid.uuid4().hex[:8]
            voice_path = f"audio/voice_{unique_id}_{voice.file_id[-8:]}.ogg"
            await voice_file.download_to_drive(voice_path)
            
            # Determine the possible languages from user settings
            user_data = db.get_user(chat_id)
            language_candidates = ['en']  # Default to English as fallback
            
            if user_data and user_data.get('selected_languages'):
                selected_langs = json.loads(user_data['selected_languages'])
                if selected_langs and len(selected_langs) > 0:
                    # Use all selected languages as candidates, with the first one as primary
                    language_candidates = selected_langs
            
            # Check if user is in HF test mode
            hf_test_mode = context.user_data.get('test_all_hf_models', False)
            
            if hf_test_mode:
                # Test all Hugging Face models and show comparison
                try:
                    from huggingface_api_asr import test_all_hf_models
                    
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message.message_id,
                        text="🔬 **Testing with ALL 12 Hugging Face models...**\n\n"
                             "⏳ Testing your voice with:\n"
                             "• Whisper Large V3 🎯\n"
                             "• Distil-Whisper ⚡\n" 
                             "• SeamlessM4T 🌍\n"
                             "• MMS 1B 🗺️\n"
                             "• And 8 more models!\n\n"
                             "📊 Results coming up..."
                    )
                    
                    # Test with all models
                    results = test_all_hf_models(voice_path)
                    
                    # Format results
                    message = "🔬 **Model Comparison Results**\n\n"
                    successful_models = 0
                    
                    for model_name, result in results.items():
                        if result and result != "Model loading, please retry":
                            message += f"✅ **{model_name}**\n"
                            message += f"   📝 {result[:80]}{'...' if len(result) > 80 else ''}\n\n"
                            successful_models += 1
                        elif result == "Model loading, please retry":
                            message += f"⏳ **{model_name}**\n"
                            message += f"   📝 Model is loading, please try again\n\n"
                        else:
                            message += f"❌ **{model_name}**\n"
                            message += f"   📝 No result\n\n"
                    
                    message += f"📊 **Summary:** {successful_models}/{len(results)} models worked successfully!"
                    
                    # Send results
                    if len(message) > 4000:
                        # Split into multiple messages if too long
                        parts = [message[i:i+3900] for i in range(0, len(message), 3900)]
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=status_message.message_id,
                            text=parts[0]
                        )
                        for part in parts[1:]:
                            await context.bot.send_message(chat_id=chat_id, text=part)
                    else:
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=status_message.message_id,
                            text=message
                        )
                    
                    # Clear test mode and clean up
                    context.user_data['test_all_hf_models'] = False
                    os.unlink(voice_path)
                    return
                    
                except Exception as e:
                    logger.error(f"ASR test mode failed: {e}")
                    # Continue with normal transcription
            
            # Use selected Hugging Face model for transcription
            transcription = None
            
            try:
                # Use 12 Advanced AI Models as PRIMARY transcription system
                from gradio_models import gradio_asr_manager
                from pydub import AudioSegment
                
                user_id = update.effective_user.id
                language_hint = language_candidates[0] if language_candidates else 'en'
                
                logger.info(f"🎯 Using 12 Advanced AI Models as PRIMARY transcription system...")
                
                # Convert OGG to WAV for AI models
                audio = AudioSegment.from_ogg(voice_path)
                wav_path = voice_path.replace('.ogg', '.wav')
                audio.export(wav_path, format="wav")
                
                # Try the best AI models first
                priority_models = ["whisper-large-v3", "distil-whisper", "seamless-m4t", "wav2vec2-large-960h", "hubert-large"]
                
                for model_id in priority_models:
                    try:
                        logger.info(f"🎯 Trying AI model: {model_id}")
                        result = gradio_asr_manager.transcribe_with_gradio_model(wav_path, model_id)
                        if result and result.get("text"):
                            transcription = result["text"]
                            logger.info(f"🎯 AI model {model_id} successful: {transcription[:100]}...")
                            break
                    except Exception as gradio_error:
                        logger.warning(f"AI model {model_id} failed: {gradio_error}")
                        continue
                
                # Fallback to Google Speech Recognition ONLY if all AI models fail
                if not transcription:
                    logger.info("🔄 All AI models failed, trying Google Speech Recognition as fallback...")
                    import speech_recognition as sr
                    
                    r = sr.Recognizer()
                    with sr.AudioFile(wav_path) as source:
                        audio_data = r.record(source)
                        
                        # Try recognition with multiple languages
                        languages = ['en-US', 'es-ES', 'fr-FR', 'de-DE', 'it-IT', 'pt-PT', 'ru-RU', 'zh-CN', 'ja-JP', 'ko-KR']
                        
                        for lang in languages:
                            try:
                                transcription = r.recognize_google(audio_data, language=lang)
                                if transcription and transcription.strip():
                                    logger.info(f"Google fallback successful with {lang}: {transcription[:100]}...")
                                    break
                            except Exception:
                                continue
                
                # Clean up temporary file
                os.unlink(wav_path)
                
                if transcription and transcription.strip():
                    logger.info(f"Voice transcription successful: {transcription[:100]}...")
                else:
                    logger.warning("Voice transcription returned empty result")
                    
            except Exception as e:
                logger.error(f"Voice transcription failed: {e}")
                transcription = None
                
                # If enhanced transcription fails, fall back to basic speech recognition
                if not transcription:
                    import speech_recognition as sr
                    from pydub import AudioSegment
                    import tempfile
                    
                    logger.info("Enhanced transcription failed, falling back to basic speech recognition")
                    
                    # Convert OGG to WAV (required by SpeechRecognition)
                    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                    audio = AudioSegment.from_file(voice_path)
                    audio.export(wav_path, format="wav")
                    
                    # Try 12 Advanced AI Models FIRST (primary transcription system)
                    try:
                        from gradio_models import gradio_asr_manager
                        logger.info("🎯 Using 12 Advanced AI Models as PRIMARY transcription system")
                        
                        # Test with the best models first
                        priority_models = ["whisper-large-v3", "distil-whisper", "seamless-m4t"]
                        for model_id in priority_models:
                            try:
                                logger.info(f"Trying primary AI model: {model_id}")
                                result = gradio_asr_manager.transcribe_with_gradio_model(wav_path, model_id)
                                if result and result.get("text"):
                                    transcription = result["text"]
                                    logger.info(f"🎯 AI model {model_id} successful: {transcription[:100]}...")
                                    break
                            except Exception as gradio_error:
                                logger.warning(f"AI model {model_id} failed: {gradio_error}")
                                continue
                    except ImportError:
                        logger.warning("AI models not available, falling back to Google Speech Recognition")
                    except Exception as e:
                        logger.error(f"Error using AI models: {e}")
                    
                    # Use Google Speech Recognition as fallback ONLY if AI models fail
                    if not transcription:
                        logger.info("🔄 AI models failed, trying Google Speech Recognition as fallback")
                    
                    # Try recognition with each language until one succeeds
                    r = sr.Recognizer()
                    with sr.AudioFile(wav_path) as source:
                        audio_data = r.record(source)
                    
                    # Convert language codes to proper format for Google Speech Recognition
                    speech_recognition_langs = []
                    for lang in language_candidates:
                        if lang == 'ru':
                            speech_recognition_langs.append('ru-RU')
                        elif lang == 'zh-CN':
                            speech_recognition_langs.append('zh-CN')
                        elif lang == 'es':
                            speech_recognition_langs.append('es-ES')
                        elif lang == 'fr':
                            speech_recognition_langs.append('fr-FR')
                        elif lang == 'de':
                            speech_recognition_langs.append('de-DE')
                        elif lang == 'it':
                            speech_recognition_langs.append('it-IT')
                        elif lang == 'pt':
                            speech_recognition_langs.append('pt-PT')
                        elif lang == 'ja':
                            speech_recognition_langs.append('ja-JP')
                        elif lang == 'ko':
                            speech_recognition_langs.append('ko-KR')
                        elif lang == 'en':
                            speech_recognition_langs.append('en-US')
                        else:
                            speech_recognition_langs.append(lang)
                    
                    # Try primary language first (first in the list)
                    primary_lang = speech_recognition_langs[0] if speech_recognition_langs else 'en-US'
                    try:
                        logger.info(f"Trying recognition with primary language: {primary_lang}")
                        transcription = r.recognize_google(audio_data, language=primary_lang)
                        if transcription:
                            logger.info(f"Google Speech Recognition successful with {primary_lang}: {transcription[:100]}...")
                    except Exception as primary_error:
                        logger.warning(f"Primary language recognition failed: {primary_error}")
                        
                        # If primary language fails, try other languages
                        for lang in speech_recognition_langs[1:]:
                            try:
                                logger.info(f"Trying recognition with alternative language: {lang}")
                                transcription = r.recognize_google(audio_data, language=lang)
                                if transcription:
                                    logger.info(f"Google Speech Recognition successful with {lang}: {transcription[:100]}...")
                                    break
                            except Exception as lang_error:
                                logger.warning(f"Recognition with {lang} failed: {lang_error}")
                                continue
                        
                        # If all selected languages fail, try English as a last resort
                        if not transcription and 'en-US' not in speech_recognition_langs:
                            try:
                                logger.info("Trying recognition with English as fallback")
                                transcription = r.recognize_google(audio_data, language='en-US')
                                if transcription:
                                    logger.info(f"Google Speech Recognition successful with English fallback: {transcription[:100]}...")
                            except Exception as en_error:
                                logger.warning(f"English fallback recognition failed: {en_error}")
                
                # Try Gradio models FIRST (primary transcription system)
                if not transcription:
                    try:
                        from gradio_models import gradio_asr_manager
                        logger.info("Using Gradio ASR models as primary transcription system")
                        
                        # Test with the best models first
                        priority_models = ["whisper-large-v3", "distil-whisper", "seamless-m4t"]
                        for model_id in priority_models:
                            try:
                                logger.info(f"Trying primary Gradio model: {model_id}")
                                result = gradio_asr_manager.transcribe_with_gradio_model(wav_path, model_id)
                                if result and result.get("text"):
                                    transcription = result["text"]
                                    logger.info(f"🎯 Gradio model {model_id} successful: {transcription[:100]}...")
                                    break
                            except Exception as gradio_error:
                                logger.warning(f"Gradio model {model_id} failed: {gradio_error}")
                                continue
                    except ImportError:
                        logger.warning("Gradio models not available, falling back to Google Speech Recognition")
                    except Exception as e:
                        logger.error(f"Error using Gradio models: {e}")
                
                # Use Google Speech Recognition as fallback ONLY if Gradio models fail
                if not transcription:
                    logger.info("Gradio models failed, trying Google Speech Recognition as fallback")
                
                # Clean up temporary file
                try:
                    os.unlink(wav_path)
                except:
                    pass
                
                if not transcription:
                    error_messages.append("Google Speech Recognition could not understand the audio in any language")
                    logger.warning("Google Speech Recognition returned empty transcription for all languages")
            except Exception as e:
                logger.warning(f"Google Speech Recognition failed completely: {e}")
                error_messages.append(f"Google Speech Recognition error: {str(e)}")
            
            # If Google Speech Recognition fails, try the II-Agent approach
            if not transcription:
                try:
                    # Import the II-Agent module
                    from ii_agent import process_voice_message
                    
                    logger.info(f"Processing voice message with II-Agent: {voice_path}")
                    
                    # Convert language codes to Google Speech Recognition format
                    # Russian needs to be 'ru-RU', Chinese 'zh-CN', etc.
                    speech_recognition_langs = []
                    for lang in language_candidates:
                        if lang == 'ru':
                            speech_recognition_langs.append('ru-RU')
                        elif lang == 'zh-CN':
                            speech_recognition_langs.append('zh-CN')
                        elif lang == 'es':
                            speech_recognition_langs.append('es-ES')
                        elif lang == 'fr':
                            speech_recognition_langs.append('fr-FR')
                        elif lang == 'it':
                            speech_recognition_langs.append('it-IT')
                        elif lang == 'pt':
                            speech_recognition_langs.append('pt-PT')
                        else:
                            speech_recognition_langs.append(lang)
                    
                    # Try II-Agent with each language until one works
                    for lang_idx, lang in enumerate(language_candidates):
                        try:
                            speech_lang = speech_recognition_langs[lang_idx] if lang_idx < len(speech_recognition_langs) else lang
                            logger.info(f"Trying II-Agent with language: {lang} (Speech format: {speech_lang})")
                            # Process the voice message with the II-Agent approach
                            # This will try multiple transcription methods with fallbacks
                            lang_transcription, agent_errors = await process_voice_message(voice_path, speech_lang)
                            
                            if agent_errors:
                                logger.warning(f"II-Agent errors with {lang}: {agent_errors}")
                                error_messages.extend([f"[{lang}] {err}" for err in agent_errors])
                            
                            if lang_transcription:
                                transcription = lang_transcription
                                logger.info(f"II-Agent transcription successful with {lang}: {transcription[:100]}...")
                                break
                        except Exception as lang_error:
                            logger.warning(f"II-Agent with {lang} failed: {lang_error}")
                            continue
                    
                    # If all languages fail, try one more time with English as fallback
                    if not transcription and 'en' not in language_candidates:
                        try:
                            logger.info("Trying II-Agent with English as fallback")
                            transcription, agent_errors = await process_voice_message(voice_path, 'en')
                            if agent_errors:
                                error_messages.extend([f"[en] {err}" for err in agent_errors])
                            if transcription:
                                logger.info(f"II-Agent transcription successful with English fallback: {transcription[:100]}...")
                        except Exception as en_error:
                            logger.warning(f"English fallback II-Agent failed: {en_error}")
                    
                    if not transcription:
                        logger.error(f"II-Agent transcription failed with all languages")
                except Exception as ii_error:
                    logger.error(f"Error in II-Agent voice processing: {ii_error}")
                    error_messages.append(f"II-Agent error: {str(ii_error)}")
                    
                    # Fall back to standard transcription if II-Agent completely fails
                    try:
                        from transcription import transcribe_audio
                        logger.info("Attempting standard transcription after II-Agent failure...")
                        transcription = transcribe_audio(voice_path, likely_lang)
                        if transcription:
                            logger.info(f"Standard transcription successful: {transcription[:100]}...")
                    except Exception as trans_error:
                        logger.error(f"Standard transcription failed: {trans_error}")
                        error_messages.append(f"Standard transcription error: {str(trans_error)}")
                        transcription = None
            
            # Check if we got a valid transcription
            if transcription and transcription.strip():
                # Update status message with the transcription
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message.message_id,
                    text=f"🎤 Transcription: {transcription}"
                )
                
                # Store the transcription in user data for future reference
                context.user_data['last_transcription'] = transcription
                context.user_data['last_message'] = transcription  # Store as last message for automatic translation
                
                # Check if user wants to test all Gradio models
                if context.user_data.get('test_all_gradio_models', False):
                    # User wants to test all 12 Gradio models
                    context.user_data['test_all_gradio_models'] = False  # Reset flag
                    
                    # Update status message
                    await context.bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=status_message.message_id,
                        text="🔬 **Testing with all 12 AI models...**\n\nThis will take a moment...",
                        parse_mode='Markdown'
                    )
                    
                    try:
                        from gradio_models import gradio_asr_manager
                        
                        # Test with all Gradio models
                        results = gradio_asr_manager.test_all_gradio_models(voice_path)
                        
                        # Generate comparison text
                        comparison_message = gradio_asr_manager.get_model_comparison_text(results)
                        
                        # Send the comparison results
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=status_message.message_id,
                            text=comparison_message,
                            parse_mode='Markdown'
                        )
                        
                        # Clean up
                        if os.path.exists(voice_path):
                            os.remove(voice_path)
                        return
                        
                    except Exception as e:
                        logger.error(f"Error testing Gradio models: {e}")
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=status_message.message_id,
                            text="❌ Error testing with all models. Please try again."
                        )
                        return
                
                # Check if we're in chat mode or translation mode
                elif context.user_data.get('in_chat_mode', False):
                    # In chat mode, process the voice message as chat input for AI
                    # First delete the status message to avoid clutter
                    await context.bot.delete_message(
                        chat_id=chat_id,
                        message_id=status_message.message_id
                    )
                    
                    # Send a processing message
                    ai_processing_message = await update.message.reply_text(
                        "🤖 Processing your voice message with AI..."
                    )
                    
                    try:
                        # Get chat history or create an empty one
                        chat_history = context.user_data.get('chat_history', [])
                        
                        # Add the transcribed message to chat history
                        chat_history.append({
                            'role': 'user',
                            'content': transcription
                        })
                        
                        # Determine which AI model to use
                        ai_model = context.user_data.get('chat_model', 'grok')
                        response = None
                        
                        # Try the selected AI model first
                        if ai_model == 'grok' and os.environ.get("XAI_API_KEY"):
                            try:
                                # Use Grok via XAI API for processing
                                from xai import chat_with_grok
                                logger.info(f"Processing voice message with Grok: {transcription[:50]}...")
                                
                                # Format system message for voice context
                                messages = [
                                    {"role": "system", "content": "The user is sending voice messages. You are Grok, an AI assistant helping with language learning and translation."}
                                ]
                                
                                # Add the chat history for context
                                messages.extend(chat_history[-10:])  # Use last 10 messages for context
                                
                                # Call Grok
                                response = chat_with_grok(messages)
                                
                                if not response:
                                    raise Exception("Empty response from Grok")
                                    
                            except Exception as grok_error:
                                logger.error(f"Error with Grok AI: {grok_error}")
                                # Fall back to Claude if Grok fails
                                if os.environ.get("ANTHROPIC_API_KEY"):
                                    response = None
                                else:
                                    raise grok_error
                        
                        # Fall back to Claude if Grok is not available or failed
                        if (not response) and os.environ.get("ANTHROPIC_API_KEY"):
                            try:
                                from amurex_ai import chat_with_claude
                                logger.info(f"Processing voice message with Claude: {transcription[:50]}...")
                                
                                # Format system message with voice context
                                messages = [
                                    {"role": "system", "content": "The user is sending voice messages. You are Claude, an AI assistant helping with language learning and translation."}
                                ]
                                
                                # Add the chat history for context
                                messages.extend(chat_history[-10:])  # Use last 10 messages for context
                                
                                # Call Claude
                                response = await chat_with_claude(messages)
                                
                                if not response:
                                    raise Exception("Empty response from Claude")
                                    
                            except Exception as claude_error:
                                logger.error(f"Error with Claude AI: {claude_error}")
                                raise claude_error
                        
                        # If all AI services fail, provide a simple response
                        if not response:
                            response = f"I heard you say: \"{transcription}\"\n\nBut I couldn't generate an AI response right now. Please check your API keys or try again later."
                        
                        # Add the response to chat history
                        chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                        
                        # Store the updated chat history
                        context.user_data['chat_history'] = chat_history
                        
                        # Limit history to last 10 exchanges (20 messages)
                        if len(context.user_data['chat_history']) > 20:
                            context.user_data['chat_history'] = context.user_data['chat_history'][-20:]
                        
                        # Add switch to translator mode button
                        keyboard = [
                            [InlineKeyboardButton("🔄 Switch to Translation Mode", callback_data="action_switch_translation")]
                        ]
                        reply_markup = InlineKeyboardMarkup(keyboard)
                        
                        # Edit the processing message with the AI response
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=ai_processing_message.message_id,
                            text=response,
                            reply_markup=reply_markup,
                            parse_mode=ParseMode.MARKDOWN
                        )
                        
                    except Exception as ai_error:
                        logger.error(f"Error processing voice with AI: {ai_error}")
                        await context.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=ai_processing_message.message_id,
                            text=f"Sorry, I couldn't process your voice message with AI. Please try again or switch to translation mode."
                        )
                else:
                    # In translation mode, translate the transcription
                    # Save original chat mode status
                    original_chat_mode = context.user_data.get('in_chat_mode', False)
                    # Force translation mode for voice messages
                    context.user_data['in_chat_mode'] = False
                    
                    # Get user's selected languages
                    if not user_data or not user_data.get('selected_languages'):
                        selected_langs = DEFAULT_LANGUAGES
                    else:
                        selected_langs = json.loads(user_data['selected_languages'])
                    
                    # Detect the language of the transcription
                    source_lang, confidence = detect_language(transcription)
                    
                    # Delete status message first
                    await context.bot.delete_message(chat_id=chat_id, message_id=status_message.message_id)
                    
                    # Create a fake update message with the transcription text to reuse existing logic
                    # This ensures voice messages use EXACTLY the same code path as text messages
                    fake_message = type('obj', (object,), {
                        'text': transcription,
                        'message_id': update.message.message_id,
                        'chat': update.message.chat,
                        'from_user': update.message.from_user,
                        'date': update.message.date,
                        'reply_text': update.message.reply_text
                    })()
                    
                    fake_update = type('obj', (object,), {
                        'message': fake_message,
                        'effective_chat': update.effective_chat,
                        'effective_user': update.effective_user
                    })()
                    
                    # Process the transcription using shared services (same quality as web app)
                    user_data = context.user_data or {}
                    target_languages = user_data.get('target_languages', ['en', 'es', 'fr', 'zh-CN', 'ru'])
                    
                    # Translate using shared services
                    translations_dict = translate_to_all_languages(transcription, target_languages)
                    
                    # Format and send the translation results
                    response_lines = [f"🎤 Voice: {transcription}\n"]
                    
                    for lang_code, trans_data in translations_dict.items():
                        translation = trans_data.get('text', '') if isinstance(trans_data, dict) else str(trans_data)
                        if translation and translation.strip():
                            lang_name = LANGUAGE_NAMES.get(lang_code, lang_code)
                            flag = get_flag_emoji(lang_code)
                            response_lines.append(f"{flag} {lang_name}: {translation}")
                    
                    # Send the complete translation response
                    response_text = "\n".join(response_lines)
                    await update.message.reply_text(response_text)
                    
                    # Store history in database (commented out due to missing method)
                    simple_translations = {}
                    for lang_code, trans_data in translations_dict.items():
                        simple_translations[lang_code] = trans_data.get('text', '')
                    
                    # Store the translation data in user context for extraction
                    context.user_data['last_text'] = transcription
                    
                    # Skip database storage as the method is missing
                    # history_id = db.add_translation_history(
                    #     chat_id=chat_id, 
                    #     source_text=transcription, 
                    #     source_lang=source_lang,
                    #     translations=simple_translations
                    # )
                    history_id = None
                    
                    # Store history ID for save action
                    if history_id:
                        context.user_data['last_history_id'] = history_id
                    
                    # Restore original chat mode
                    context.user_data['in_chat_mode'] = original_chat_mode
            else:
                # Transcription failed or returned empty result
                error_text = "❌ I couldn't transcribe your voice message."
                
                # Add some detailed error information if available
                if error_messages:
                    error_details = "\n\nI tried multiple voice recognition systems, but encountered these issues:"
                    # Limit to first 3 errors to avoid overwhelming the user
                    for i, error in enumerate(error_messages[:3]):
                        error_details += f"\n• {error}"
                    
                    error_text += f"{error_details}\n\nFor best results, please try again with clearer audio or type your message as text."
                else:
                    error_text += " For best results, please type your message as text for translation."
                
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message.message_id,
                    text=error_text
                )
        except Exception as e:
            logger.error(f"Error processing voice message: {e}")
            try:
                # Provide more detailed error information
                detailed_error = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Detailed voice processing error: {detailed_error}")
                
                error_text = "❌ I couldn't process your voice message."
                if error_messages:
                    error_details = "\n\nI tried multiple voice recognition systems, but encountered these issues:"
                    for i, error in enumerate(error_messages[:3]):
                        error_details += f"\n• {error}"
                    error_text += f"{error_details}"
                
                error_text += "\n\nFor best results, please try again with clearer audio or type your message as text."
                
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_message.message_id,
                    text=error_text
                )
            except Exception as msg_error:
                logger.error(f"Failed to update status message: {msg_error}")
                # Try to send a new message if editing failed
                try:
                    error_text = "❌ I couldn't process your voice message."
                    if error_messages:
                        error_text += " I tried multiple recognition systems, but none worked."
                    error_text += " For best results, please try again with clearer audio or type your message as text."
                    
                    await update.message.reply_text(error_text)
                except Exception:
                    pass
        
        # Clean up temporary files
        try:
            if voice_path and os.path.exists(voice_path):
                os.remove(voice_path)
                logger.info(f"Cleaned up temporary voice file: {voice_path}")
        except Exception as e:
            logger.error(f"Error cleaning up voice files: {e}")
    
    # Register the voice message handler
    # Use ElevenLabs voice handler to match web version functionality
    try:
        from elevenlabs_voice_handler import handle_voice_message_elevenlabs
        application.add_handler(MessageHandler(filters.VOICE, handle_voice_message_elevenlabs))
        logger.info("ElevenLabs voice handler loaded successfully")
    except ImportError as e:
        logger.error(f"Could not load ElevenLabs voice handler: {e}")
        application.add_handler(MessageHandler(filters.VOICE, handle_voice_message_old))
    
    # Register callback handlers for Gradio models
    application.add_handler(CallbackQueryHandler(handle_asr_model_callback, pattern="^gradio_"))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Set bot commands (This updates the menu in Telegram)
    # Removed /translate command as the bot now translates by default
    # history command completely removed from menu but still works as a handler
    commands = [
        ("start", "Start the bot"),
        ("languages", "Select languages to translate to"),
        ("audio", "Generate speech from last message"),
        ("chat", "Chat directly with AI"),
        ("translator", "Switch back to translation mode"),
        ("flags", "Toggle language flags and names in translations"),
        ("shop", "Browse our language learning shop")
    ]
    
    # We'll use post_init to set commands after bot is initialized
    async def post_init(application):
        try:
            await application.bot.set_my_commands(commands)
            logger.info("Bot commands menu updated successfully")
        except Exception as e:
            logger.error(f"Error updating bot commands menu: {e}")
    
    # Start the Bot with post_init callback
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()