"""
Advanced Features Integration Module for Enhanced Language Bot.
This module integrates all the newly developed advanced features:
1. Augmented reality translation overlay
2. Personalized language learning soundtrack
3. Emoji reaction for learning milestones
4. Cyberpunk progress animation
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import time
import uuid
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Import database module
import database as db

# Import feature modules
try:
    from augmented_reality import create_augmented_translation_image, extract_text_from_image, detect_text_regions
    AR_AVAILABLE = True
except ImportError:
    logger.warning("Augmented reality module not available")
    AR_AVAILABLE = False

try:
    from learning_soundtrack import generate_learning_soundtrack, get_available_themes
    SOUNDTRACK_AVAILABLE = True
except ImportError:
    logger.warning("Learning soundtrack module not available")
    SOUNDTRACK_AVAILABLE = False

try:
    from learning_emoji import (
        check_milestone_achievements, check_tier_achievements, 
        generate_achievement_message, generate_streak_emoji_animation,
        get_tier_progress, MILESTONE_REACTIONS, ACHIEVEMENT_TIERS
    )
    EMOJI_AVAILABLE = True
except ImportError:
    logger.warning("Learning emoji module not available")
    EMOJI_AVAILABLE = False

try:
    from cyberpunk_animation import (
        generate_cyberpunk_text_animation, generate_cyberpunk_progress_bar,
        generate_language_level_animation, generate_achievement_unlock_animation,
        generate_cyberpunk_image_animation
    )
    ANIMATION_AVAILABLE = True
except ImportError:
    logger.warning("Cyberpunk animation module not available")
    ANIMATION_AVAILABLE = False

async def handle_ar_translation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /artranslate command to create augmented reality translation overlay.
    This command works when replying to a photo message.
    """
    chat_id = update.effective_chat.id
    
    # Check if AR feature is available
    if not AR_AVAILABLE:
        await update.message.reply_text(
            "🔧 The augmented reality translation feature is not available.\n"
            "Please contact the administrator or try again later."
        )
        return
    
    # Check if the message is a reply to a photo
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        # Send instructions on how to use this command
        await update.message.reply_text(
            "🖼️ *Augmented Reality Translation*\n\n"
            "To use this feature, reply to a photo message with `/artranslate`.\n"
            "I'll overlay translations on the image, highlighting text areas.\n\n"
            "You can also add text to translate after the command:\n"
            "`/artranslate This text will be translated and overlaid`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    # Get user's selected languages
    user_data = db.get_user(chat_id)
    if not user_data or not user_data.get('selected_languages'):
        await update.message.reply_text(
            "Please select your target languages first using `/languages`."
        )
        return
    
    # Get selected languages
    selected_langs = json.loads(user_data['selected_languages'])
    if not selected_langs:
        await update.message.reply_text(
            "Please select at least one target language using `/languages`."
        )
        return
    
    # Send a status message
    status_message = await update.message.reply_text("🔍 Processing image for AR translation...")
    
    # Download the photo
    photo = update.message.reply_to_message.photo[-1]  # Get the largest photo size
    photo_file = await context.bot.get_file(photo.file_id)
    
    # Create temporary file path
    photo_path = f"static/ar_overlay/original_{uuid.uuid4()}.jpg"
    
    # Ensure directory exists
    os.makedirs("static/ar_overlay", exist_ok=True)
    
    # Download the photo
    await photo_file.download_to_drive(photo_path)
    
    # Determine source text to use
    source_text = ""
    
    # Check if command has arguments (which would be the text to translate)
    if context.args:
        source_text = ' '.join(context.args)
    else:
        # Try to extract text from the image
        source_text = extract_text_from_image(photo_path)
    
    # If we still don't have text, ask the user to provide it
    if not source_text:
        await status_message.edit_text(
            "❓ I couldn't extract any text from this image.\n"
            "Please add the text you want to translate after the command:\n"
            "`/artranslate Text to translate`"
        )
        return
    
    # Get translations for the text
    translated_texts = {}
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=chat_id, action='typing')
    
    # Translate to each selected language
    from enhanced_translator import translate_text_all_services
    
    for lang_code in selected_langs:
        translation, _ = translate_text_all_services(source_text, lang_code)
        if translation:
            translated_texts[lang_code] = translation
    
    # If no translations were generated, inform the user
    if not translated_texts:
        await status_message.edit_text(
            "❌ Sorry, I couldn't generate translations for this text."
        )
        return
    
    # Detect text regions in the image
    text_regions = detect_text_regions(photo_path)
    
    # Create AR translation overlay
    ar_image_path = create_augmented_translation_image(
        photo_path, source_text, translated_texts, text_regions
    )
    
    # Check if AR image was created successfully
    if not ar_image_path:
        await status_message.edit_text(
            "❌ Failed to create AR translation overlay. Please try again later."
        )
        return
    
    # Send the AR image
    with open(ar_image_path, 'rb') as ar_image:
        await update.message.reply_photo(
            photo=ar_image,
            caption="🌐 AR Translation Overlay"
        )
    
    # Delete the status message
    await status_message.delete()
    
    # Add keyboard with button to translate more text
    keyboard = [
        [InlineKeyboardButton("🔄 Translate more text on this image", callback_data="action_more_ar_translation")]
    ]
    
    await update.message.reply_text(
        "I've created an AR translation overlay!\n"
        "Need to translate more text on this image?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    
    # Store this photo path and languages in user context for potential future use
    context.user_data['last_ar_photo'] = photo_path
    context.user_data['last_ar_languages'] = selected_langs

async def handle_soundtrack(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /soundtrack command to generate a personalized language learning soundtrack.
    """
    chat_id = update.effective_chat.id
    
    # Check if soundtrack feature is available
    if not SOUNDTRACK_AVAILABLE:
        await update.message.reply_text(
            "🔧 The language learning soundtrack feature is not available.\n"
            "Please contact the administrator or try again later."
        )
        return
    
    # Get available themes
    available_themes = get_available_themes()
    
    # Check if the message has arguments (theme and/or duration)
    theme = "focus"  # Default theme
    duration = 180   # Default duration (3 minutes)
    
    if context.args:
        # Parse arguments
        for arg in context.args:
            # Check if argument is a known theme
            if arg.lower() in available_themes:
                theme = arg.lower()
            # Check if argument is a duration in seconds or minutes
            elif arg.isdigit():
                duration = int(arg)
            elif arg.lower().endswith("s") and arg[:-1].isdigit():
                duration = int(arg[:-1])
            elif arg.lower().endswith("m") and arg[:-1].isdigit():
                duration = int(arg[:-1]) * 60
    
    # Cap duration at 5 minutes (300 seconds) to avoid large files
    duration = min(duration, 300)
    
    # Get user's primary language
    user_data = db.get_user(chat_id)
    language_code = "en"  # Default to English
    
    if user_data and user_data.get('selected_languages'):
        selected_langs = json.loads(user_data['selected_languages'])
        if selected_langs and len(selected_langs) > 0:
            # Use the first selected language
            language_code = selected_langs[0]
    
    # Send a status message
    status_message = await update.message.reply_text(
        f"🎵 Generating a {theme} language learning soundtrack "
        f"for {duration} seconds in {language_code}..."
    )
    
    # Generate the soundtrack
    soundtrack_path = generate_learning_soundtrack(
        language_code=language_code,
        theme=theme,
        duration=duration
    )
    
    # Check if soundtrack was generated successfully
    if not soundtrack_path:
        await status_message.edit_text(
            "❌ Failed to generate a language learning soundtrack. Please try again later."
        )
        return
    
    # Send the audio file
    with open(soundtrack_path, 'rb') as audio_file:
        await update.message.reply_audio(
            audio=audio_file,
            title=f"Language Learning Soundtrack - {theme.capitalize()}",
            caption=f"🎵 {theme.capitalize()} learning soundtrack for {language_code}",
            duration=duration
        )
    
    # Delete the status message
    await status_message.delete()
    
    # Send themes keyboard for easy access
    keyboard = []
    row = []
    
    for i, theme_option in enumerate(available_themes):
        row.append(InlineKeyboardButton(
            theme_option.capitalize(), 
            callback_data=f"soundtrack_{theme_option}"
        ))
        
        # 3 buttons per row
        if (i + 1) % 3 == 0 or i == len(available_themes) - 1:
            keyboard.append(row)
            row = []
    
    await update.message.reply_text(
        "Choose another soundtrack theme:",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_progress(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /progress command to show cyberpunk progress animation.
    """
    chat_id = update.effective_chat.id
    
    # Check if animation feature is available
    if not ANIMATION_AVAILABLE or not EMOJI_AVAILABLE:
        await update.message.reply_text(
            "🔧 The cyberpunk progress animation feature is not available.\n"
            "Please contact the administrator or try again later."
        )
        return
    
    # Get user stats from database
    user_stats = db.get_user_stats(chat_id)
    
    if not user_stats:
        user_stats = {
            "translations_count": 0,
            "vocabulary_count": 0,
            "streak_days": 0,
            "languages_count": 0
        }
    
    # Get tier progress information
    current_tier, progress_value, next_tier, main_requirement = get_tier_progress(user_stats)
    
    # Send a status message
    status_message = await update.message.reply_text("🧮 Calculating your language learning progress...")
    
    # Generate cyberpunk progress text animation
    progress_text = (
        f"Progress: {current_tier} → {next_tier}\n"
        f"{main_requirement}"
    )
    
    # Generate cyberpunk image animation
    animation_path = generate_cyberpunk_image_animation(
        text=f"Language Learning Progress\n{current_tier} → {next_tier}",
        progress=progress_value,
        level=None,
        achievement=None,
        style="neon"
    )
    
    # Delete status message
    await status_message.delete()
    
    # Check if image animation was created
    if animation_path and os.path.exists(animation_path):
        # Send the progress animation
        with open(animation_path, 'rb') as animation_file:
            await update.message.reply_animation(
                animation=animation_file,
                caption=f"⚡ *Cyberpunk Progress* ⚡\n\n{progress_text}\n\nProgress: {int(progress_value * 100)}%",
                parse_mode=ParseMode.MARKDOWN
            )
    else:
        # Fallback to text-only progress bar
        progress_bar = generate_cyberpunk_progress_bar(progress_value, "neon")[0]
        
        # Check if there are any milestone achievements to unlock
        achievements = check_milestone_achievements(user_stats)
        
        if achievements:
            # Generate achievement message
            achievement_msg = generate_achievement_message(achievements)
            
            await update.message.reply_text(
                f"⚡ *Cyberpunk Progress* ⚡\n\n"
                f"{progress_text}\n\n"
                f"`{progress_bar}`\n\n"
                f"{achievement_msg}",
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                f"⚡ *Cyberpunk Progress* ⚡\n\n"
                f"{progress_text}\n\n"
                f"`{progress_bar}`",
                parse_mode=ParseMode.MARKDOWN
            )
    
    # Add keyboard with options
    keyboard = [
        [
            InlineKeyboardButton("🏆 Achievements", callback_data="action_view_achievements"),
            InlineKeyboardButton("📊 Stats", callback_data="action_view_stats")
        ],
        [
            InlineKeyboardButton("🎵 Learning Soundtrack", callback_data="action_soundtrack"),
            InlineKeyboardButton("🔄 Update Progress", callback_data="action_update_progress")
        ]
    ]
    
    await update.message.reply_text(
        "What would you like to do next?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def handle_achievements(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle the /achievements command to show emoji reactions for learning milestones.
    """
    chat_id = update.effective_chat.id
    
    # Check if emoji feature is available
    if not EMOJI_AVAILABLE:
        await update.message.reply_text(
            "🔧 The emoji reaction feature is not available.\n"
            "Please contact the administrator or try again later."
        )
        return
    
    # Get user stats from database
    user_stats = db.get_user_stats(chat_id)
    
    if not user_stats:
        await update.message.reply_text(
            "You don't have any learning achievements yet.\n"
            "Start translating and using the bot to earn achievements!"
        )
        return
    
    # Get all achievements (don't filter by acknowledged status)
    all_achievements = []
    for milestone_type, milestones in MILESTONE_REACTIONS.items():
        # Skip special achievements which are handled separately
        if milestone_type == "special_achievements":
            continue
            
        # Get the current stat value
        current_value = user_stats.get(milestone_type, 0)
        
        # Check all milestones and include those that are unlocked
        for milestone in milestones:
            if current_value >= milestone["count"]:
                all_achievements.append({
                    "type": milestone_type,
                    "achievement_key": f"{milestone_type}_{milestone['count']}",
                    "emoji": milestone["emoji"],
                    "message": milestone["message"],
                    "value": current_value,
                    "threshold": milestone["count"],
                    "unlocked": True
                })
    
    # Get all tier achievements
    all_tiers = []
    for tier in ACHIEVEMENT_TIERS:
        # Check if all requirements are met
        requirements_met = True
        
        for stat_key, required_value in tier["requirements"].items():
            current_value = user_stats.get(stat_key, 0)
            if current_value < required_value:
                requirements_met = False
                break
        
        all_tiers.append({
            "name": tier["name"],
            "emoji": tier["emoji"],
            "reward": tier["reward"],
            "unlocked": requirements_met
        })
    
    # Format message with all achievements
    message = "🏆 *Your Learning Achievements* 🏆\n\n"
    
    # Add tiers section
    message += "*Progress Tiers:*\n"
    for tier in all_tiers:
        if tier["unlocked"]:
            message += f"{tier['emoji']} {tier['name']} - Unlocked! ✓\n"
        else:
            message += f"⭐ {tier['name']} - Locked\n"
    
    message += "\n*Milestones:*\n"
    
    # Group achievements by type for cleaner display
    achievement_types = {
        "translations_count": "Translation",
        "vocabulary_count": "Vocabulary",
        "streak_days": "Streak",
        "languages_count": "Languages",
        "audio_usage": "Audio",
        "voice_messages": "Voice",
        "ai_interactions": "AI",
        "usage_days": "Usage"
    }
    
    for type_key, type_name in achievement_types.items():
        type_achievements = [a for a in all_achievements if a["type"] == type_key]
        if type_achievements:
            message += f"\n*{type_name} Achievements:*\n"
            for achievement in type_achievements:
                message += f"{achievement['emoji']} {achievement['message']}\n"
    
    # Send the achievements message
    await update.message.reply_text(
        message,
        parse_mode=ParseMode.MARKDOWN
    )
    
    # Generate streak animation if streak is substantial
    streak_days = user_stats.get("streak_days", 0)
    if streak_days >= 7:
        # Generate animation frames
        animation_frames = generate_streak_emoji_animation(streak_days)
        
        # Send initial frame
        animation_message = await update.message.reply_text(
            f"🔥 *Streak Animation* 🔥\n\n{animation_frames[0]}",
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Animate by editing the message
        for frame in animation_frames[1:]:
            await animation_message.edit_text(
                f"🔥 *Streak Animation* 🔥\n\n{frame}",
                parse_mode=ParseMode.MARKDOWN
            )
            time.sleep(0.5)  # Half-second delay between frames

# Handler for Callback Queries related to these features
async def handle_advanced_features_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle callback queries related to advanced features."""
    query = update.callback_query
    chat_id = update.effective_chat.id
    
    # Acknowledge the query to stop the loading animation
    await query.answer()
    
    # Extract the callback data
    callback_data = query.data
    
    # Handle different callback actions
    if callback_data.startswith("soundtrack_"):
        # Handle soundtrack theme selection
        theme = callback_data.split("_")[1]
        
        # Get user's primary language
        user_data = db.get_user(chat_id)
        language_code = "en"  # Default to English
        
        if user_data and user_data.get('selected_languages'):
            selected_langs = json.loads(user_data['selected_languages'])
            if selected_langs and len(selected_langs) > 0:
                # Use the first selected language
                language_code = selected_langs[0]
        
        # Send a status message by editing the current message
        await query.edit_message_text(
            f"🎵 Generating a {theme} language learning soundtrack "
            f"for 180 seconds in {language_code}..."
        )
        
        # Generate the soundtrack
        soundtrack_path = generate_learning_soundtrack(
            language_code=language_code,
            theme=theme,
            duration=180  # 3 minutes default
        )
        
        # Check if soundtrack was generated successfully
        if not soundtrack_path:
            await query.edit_message_text(
                "❌ Failed to generate a language learning soundtrack. Please try again later."
            )
            return
        
        # Send the audio file as a new message
        with open(soundtrack_path, 'rb') as audio_file:
            await context.bot.send_audio(
                chat_id=chat_id,
                audio=audio_file,
                title=f"Language Learning Soundtrack - {theme.capitalize()}",
                caption=f"🎵 {theme.capitalize()} learning soundtrack for {language_code}",
                duration=180
            )
        
        # Delete the callback message or edit it to show success
        await query.edit_message_text(
            f"✅ Generated a {theme} language learning soundtrack!"
        )
    
    elif callback_data == "action_more_ar_translation":
        # Handle request for more AR translation on the same image
        if 'last_ar_photo' in context.user_data and 'last_ar_languages' in context.user_data:
            # Show a text input prompt
            await query.edit_message_text(
                "Please send the text you want to translate and overlay on the image."
            )
            
            # Set a flag in user data to indicate waiting for text
            context.user_data['waiting_for_ar_text'] = True
        else:
            await query.edit_message_text(
                "No previous image found. Please use the /artranslate command with a photo."
            )
    
    elif callback_data == "action_view_achievements":
        # Handle request to view achievements
        await query.edit_message_text("Loading your achievements...")
        
        # Call the achievements handler
        await handle_achievements(update, context)
    
    elif callback_data == "action_view_stats":
        # Handle request to view stats
        await query.edit_message_text("Loading your statistics...")
        
        # Redirect to stats command
        await context.bot.send_message(chat_id=chat_id, text="/stats")
    
    elif callback_data == "action_soundtrack":
        # Handle request to generate soundtrack
        await query.edit_message_text("Opening soundtrack generator...")
        
        # Redirect to soundtrack command
        await context.bot.send_message(chat_id=chat_id, text="/soundtrack")
    
    elif callback_data == "action_update_progress":
        # Handle request to update progress
        await query.edit_message_text("Updating your progress...")
        
        # Call the progress handler
        await handle_progress(update, context)

# Register these command handlers in the main application
def register_advanced_features(application):
    """Register advanced feature commands and handlers."""
    from telegram.ext import CommandHandler, CallbackQueryHandler
    
    # Advanced feature commands removed as requested
    
    # Register callback query handler (keeping this for backward compatibility)
    application.add_handler(CallbackQueryHandler(
        handle_advanced_features_callback, 
        pattern="^(soundtrack_|action_more_ar_translation|action_view_achievements|action_view_stats|action_soundtrack|action_update_progress)"
    ))
    
    logger.info("Advanced features registered successfully")
    
    return True