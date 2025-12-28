"""
Emoji Reaction for Learning Milestones.
Provides gamified feedback and celebrations for language learning achievements.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import random
import json
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Define milestone types and their associated emoji reactions
MILESTONE_REACTIONS = {
    # Translation milestones
    "translations_count": [
        {"count": 1, "emoji": "🎯", "message": "First translation! You're on your way!"},
        {"count": 10, "emoji": "🔥", "message": "10 translations! You're building momentum!"},
        {"count": 50, "emoji": "⚡", "message": "50 translations! You're on fire!"},
        {"count": 100, "emoji": "💯", "message": "100 translations! Triple digits achievement!"},
        {"count": 500, "emoji": "🌟", "message": "500 translations! You're a translation powerhouse!"},
        {"count": 1000, "emoji": "🏆", "message": "1000 translations! You've reached translation mastery!"},
        {"count": 5000, "emoji": "👑", "message": "5000 translations! You're a language royalty!"},
    ],
    
    # Vocabulary milestones
    "vocabulary_count": [
        {"count": 10, "emoji": "📝", "message": "10 words learned! Building your vocabulary!"},
        {"count": 50, "emoji": "📚", "message": "50 words learned! Your vocabulary is growing!"},
        {"count": 100, "emoji": "📖", "message": "100 words learned! You're expanding your language horizons!"},
        {"count": 500, "emoji": "🧠", "message": "500 words learned! Your language brain is growing!"},
        {"count": 1000, "emoji": "🎓", "message": "1000 words learned! You've reached conversational level!"},
        {"count": 2000, "emoji": "🔍", "message": "2000 words learned! You've reached advanced vocabulary!"},
        {"count": 5000, "emoji": "🌍", "message": "5000 words learned! You're approaching native fluency!"},
    ],
    
    # Streak milestones
    "streak_days": [
        {"count": 1, "emoji": "🔥", "message": "1 day streak! The journey begins!"},
        {"count": 3, "emoji": "🔥🔥", "message": "3 day streak! Consistency is key!"},
        {"count": 7, "emoji": "🔥🔥🔥", "message": "7 day streak! A whole week of learning!"},
        {"count": 14, "emoji": "🌟", "message": "14 day streak! Two weeks strong!"},
        {"count": 30, "emoji": "🌟🌟", "message": "30 day streak! A whole month devoted to language learning!"},
        {"count": 60, "emoji": "🌟🌟🌟", "message": "60 day streak! Two months of dedication!"},
        {"count": 100, "emoji": "💎", "message": "100 day streak! You're a language learning diamond!"},
        {"count": 180, "emoji": "💎💎", "message": "180 day streak! Half a year of consistent learning!"},
        {"count": 365, "emoji": "👑", "message": "365 day streak! A full year of language learning! Phenomenal!"},
    ],
    
    # Languages count milestones
    "languages_count": [
        {"count": 2, "emoji": "🗣️", "message": "You're learning 2 languages! Becoming bilingual!"},
        {"count": 3, "emoji": "🌐", "message": "3 languages! You're on your way to becoming a polyglot!"},
        {"count": 5, "emoji": "🧩", "message": "5 languages! Your linguistic puzzle is expanding!"},
        {"count": 10, "emoji": "🔮", "message": "10 languages! You're a language wizard!"},
    ],
    
    # Audio usage milestones
    "audio_usage": [
        {"count": 10, "emoji": "🎵", "message": "You've listened to 10 audio pronunciations! Training your ear!"},
        {"count": 50, "emoji": "🎧", "message": "50 audio clips! Your listening skills are developing!"},
        {"count": 100, "emoji": "🎶", "message": "100 audio clips! Your ears are tuning into the language!"},
        {"count": 500, "emoji": "🔊", "message": "500 audio clips! You're developing a strong ear for the language!"},
    ],
    
    # Voice message milestones
    "voice_messages": [
        {"count": 1, "emoji": "🎤", "message": "First voice message! Speaking practice begins!"},
        {"count": 10, "emoji": "🗣️", "message": "10 voice messages! You're practicing speaking!"},
        {"count": 50, "emoji": "📢", "message": "50 voice messages! Your speaking confidence is growing!"},
        {"count": 100, "emoji": "🎭", "message": "100 voice messages! You're becoming a confident speaker!"},
    ],
    
    # AI interactions milestones
    "ai_interactions": [
        {"count": 1, "emoji": "🤖", "message": "First AI interaction! Using technology to enhance learning!"},
        {"count": 10, "emoji": "💬", "message": "10 AI interactions! Leveraging AI for language mastery!"},
        {"count": 50, "emoji": "🧠", "message": "50 AI interactions! Deep diving into AI-assisted learning!"},
        {"count": 100, "emoji": "🔮", "message": "100 AI interactions! You're an AI language learning expert!"},
    ],
    
    # Time-based milestones (usage days)
    "usage_days": [
        {"count": 7, "emoji": "📅", "message": "You've used the translator for a week! Great start!"},
        {"count": 30, "emoji": "📆", "message": "A month of language learning! Consistency is key!"},
        {"count": 90, "emoji": "🗓️", "message": "Three months of language journey! You're committed!"},
        {"count": 180, "emoji": "⏳", "message": "Six months of language exploration! Time well spent!"},
        {"count": 365, "emoji": "⌛", "message": "A full year of language learning! What an achievement!"},
    ],
    
    # Special achievements
    "special_achievements": [
        {"id": "night_owl", "emoji": "🦉", "message": "Night Owl! You're learning even late at night!"},
        {"id": "early_bird", "emoji": "🐦", "message": "Early Bird! Starting your day with language learning!"},
        {"id": "weekend_warrior", "emoji": "⚔️", "message": "Weekend Warrior! Learning on weekends too!"},
        {"id": "global_traveler", "emoji": "✈️", "message": "Global Traveler! You've used multiple languages in a day!"},
        {"id": "vocabulary_builder", "emoji": "📚", "message": "Vocabulary Builder! You've saved many words!"},
        {"id": "conversation_master", "emoji": "💬", "message": "Conversation Master! You're having extended AI chats!"},
        {"id": "pronunciation_pro", "emoji": "🎤", "message": "Pronunciation Pro! You use audio features frequently!"},
        {"id": "daily_dedication", "emoji": "⏰", "message": "Daily Dedication! You use the bot every day!"},
    ]
}

# Achievement tiers with special rewards
ACHIEVEMENT_TIERS = [
    {
        "name": "Beginner",
        "requirements": {
            "translations_count": 20,
            "streak_days": 3
        },
        "emoji": "🌱",
        "reward": "Access to basic vocabulary lists"
    },
    {
        "name": "Novice",
        "requirements": {
            "translations_count": 50,
            "streak_days": 7,
            "vocabulary_count": 30
        },
        "emoji": "🔍",
        "reward": "Access to language learning soundtracks"
    },
    {
        "name": "Apprentice",
        "requirements": {
            "translations_count": 100,
            "streak_days": 14,
            "vocabulary_count": 50,
            "audio_usage": 30
        },
        "emoji": "📘",
        "reward": "Access to AR translation features"
    },
    {
        "name": "Adept",
        "requirements": {
            "translations_count": 200,
            "streak_days": 30,
            "vocabulary_count": 100,
            "ai_interactions": 30
        },
        "emoji": "🔆",
        "reward": "Custom language exercises"
    },
    {
        "name": "Expert",
        "requirements": {
            "translations_count": 500,
            "streak_days": 60,
            "vocabulary_count": 300,
            "voice_messages": 50
        },
        "emoji": "⭐",
        "reward": "Advanced pronunciation feedback"
    },
    {
        "name": "Master",
        "requirements": {
            "translations_count": 1000,
            "streak_days": 100,
            "vocabulary_count": 500,
            "languages_count": 3
        },
        "emoji": "🏆",
        "reward": "Custom language learning path"
    },
    {
        "name": "Grandmaster",
        "requirements": {
            "translations_count": 5000,
            "streak_days": 365,
            "vocabulary_count": 2000,
            "languages_count": 5
        },
        "emoji": "👑",
        "reward": "Ultimate polyglot status and exclusive features"
    }
]

def check_milestone_achievements(user_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Check for milestone achievements based on user stats.
    
    Args:
        user_stats: Dictionary with user statistics
        
    Returns:
        List of milestone achievements (with emoji and message)
    """
    achievements = []
    
    # Check numeric milestones
    for stat_key, milestones in MILESTONE_REACTIONS.items():
        # Skip special achievements which are handled separately
        if stat_key == "special_achievements":
            continue
            
        # Get the current stat value
        current_value = user_stats.get(stat_key, 0)
        
        # Check if we have "unlocked" milestones
        for milestone in milestones:
            if current_value >= milestone["count"]:
                # Check if this milestone has already been acknowledged
                achievement_key = f"{stat_key}_{milestone['count']}"
                if achievement_key not in user_stats.get("acknowledged_achievements", []):
                    achievements.append({
                        "type": stat_key,
                        "achievement_key": achievement_key,
                        "emoji": milestone["emoji"],
                        "message": milestone["message"],
                        "value": current_value,
                        "threshold": milestone["count"]
                    })
    
    # Check special achievements
    for achievement in MILESTONE_REACTIONS.get("special_achievements", []):
        achievement_id = achievement["id"]
        
        # Check if user has specific achievement criteria
        # This would be determined by specific logic for each achievement type
        if _check_special_achievement(user_stats, achievement_id):
            # Check if this achievement has already been acknowledged
            if achievement_id not in user_stats.get("acknowledged_achievements", []):
                achievements.append({
                    "type": "special",
                    "achievement_key": achievement_id,
                    "emoji": achievement["emoji"],
                    "message": achievement["message"],
                    "value": 1,
                    "threshold": 1
                })
    
    return achievements

def check_tier_achievements(user_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Check for tier-based achievements that unlock rewards.
    
    Args:
        user_stats: Dictionary with user statistics
        
    Returns:
        List of tier achievements (with emoji, name, and reward)
    """
    achievements = []
    
    # Check each tier in order
    for tier in ACHIEVEMENT_TIERS:
        # Check if all requirements are met
        requirements_met = True
        
        for stat_key, required_value in tier["requirements"].items():
            current_value = user_stats.get(stat_key, 0)
            if current_value < required_value:
                requirements_met = False
                break
        
        # If all requirements are met and this tier hasn't been acknowledged yet
        if requirements_met:
            tier_key = f"tier_{tier['name'].lower()}"
            if tier_key not in user_stats.get("acknowledged_tiers", []):
                achievements.append({
                    "type": "tier",
                    "achievement_key": tier_key,
                    "emoji": tier["emoji"],
                    "name": tier["name"],
                    "message": f"You've reached the {tier['name']} tier!",
                    "reward": tier["reward"]
                })
    
    return achievements

def _check_special_achievement(user_stats: Dict[str, Any], achievement_id: str) -> bool:
    """
    Check if a user has earned a specific special achievement.
    
    Args:
        user_stats: Dictionary with user statistics
        achievement_id: ID of the special achievement to check
        
    Returns:
        True if the achievement criteria are met, False otherwise
    """
    # Get the current time
    now = datetime.now()
    
    # Check each special achievement type
    if achievement_id == "night_owl":
        # Check if user has used the bot late at night (11 PM - 4 AM)
        late_night_usage = user_stats.get("usage_hours", {})
        for hour in ["23", "0", "1", "2", "3", "4"]:
            if late_night_usage.get(hour, 0) > 5:
                return True
    
    elif achievement_id == "early_bird":
        # Check if user has used the bot early morning (5 AM - 8 AM)
        early_morning_usage = user_stats.get("usage_hours", {})
        for hour in ["5", "6", "7", "8"]:
            if early_morning_usage.get(hour, 0) > 5:
                return True
    
    elif achievement_id == "weekend_warrior":
        # Check if user has used the bot on weekends
        weekend_usage = user_stats.get("usage_days_of_week", {})
        if weekend_usage.get("5", 0) > 2 and weekend_usage.get("6", 0) > 2:  # Saturday and Sunday
            return True
    
    elif achievement_id == "global_traveler":
        # Check if user has used multiple languages in a single day
        languages_today = user_stats.get("languages_used_today", [])
        if len(languages_today) >= 3:
            return True
    
    elif achievement_id == "vocabulary_builder":
        # Check if user has saved many vocabulary items
        saved_vocabulary = user_stats.get("saved_vocabulary", 0)
        if saved_vocabulary >= 50:
            return True
    
    elif achievement_id == "conversation_master":
        # Check if user has had long conversations with AI
        long_chats = user_stats.get("ai_long_conversations", 0)
        if long_chats >= 5:
            return True
    
    elif achievement_id == "pronunciation_pro":
        # Check if user frequently uses audio features
        audio_usage = user_stats.get("audio_usage", 0)
        voice_messages = user_stats.get("voice_messages", 0)
        if audio_usage + voice_messages >= 100:
            return True
    
    elif achievement_id == "daily_dedication":
        # Check if user has consistent daily usage
        consecutive_days = user_stats.get("consecutive_usage_days", 0)
        if consecutive_days >= 14:
            return True
    
    return False

def generate_achievement_message(achievements: List[Dict[str, Any]]) -> str:
    """
    Generate a formatted message for achievements.
    
    Args:
        achievements: List of achievements to format
        
    Returns:
        Formatted message string
    """
    if not achievements:
        return ""
    
    # Sort achievements by type (tier first, then others)
    sorted_achievements = sorted(achievements, key=lambda a: 0 if a["type"] == "tier" else 1)
    
    # Create message
    message = "🎉 *Achievement Unlocked!* 🎉\n\n"
    
    for achievement in sorted_achievements:
        emoji = achievement["emoji"]
        
        if achievement["type"] == "tier":
            # Format tier achievement
            message += f"{emoji} *{achievement['name']} Tier!*\n"
            message += f"➡️ Reward: {achievement['reward']}\n\n"
        else:
            # Format regular milestone achievement
            message += f"{emoji} *{achievement['message']}*\n\n"
    
    # Add encouragement
    if len(achievements) == 1:
        message += "Keep up the great work! 💪"
    else:
        message += f"You've unlocked {len(achievements)} new achievements! Amazing progress! 🚀"
    
    return message

def generate_streak_emoji_animation(streak_days: int) -> List[str]:
    """
    Generate a series of emoji frames for a streak animation.
    
    Args:
        streak_days: Number of days in the streak
        
    Returns:
        List of emoji strings representing animation frames
    """
    animation_frames = []
    
    if streak_days < 7:
        # Fire animation for short streaks
        animation_frames = [
            "🔥",
            "🔥 🔥",
            "🔥 🔥 🔥",
            "✨ 🔥 ✨",
            "✨ 🔥 🔥 ✨",
            "✨ 🔥 🔥 🔥 ✨"
        ]
    elif streak_days < 30:
        # Star animation for medium streaks
        animation_frames = [
            "⭐",
            "✨ ⭐ ✨",
            "🌟 ⭐ 🌟",
            "✨ 🌟 ⭐ 🌟 ✨",
            "🔥 🌟 ⭐ 🌟 🔥"
        ]
    elif streak_days < 100:
        # Gem animation for longer streaks
        animation_frames = [
            "💎",
            "✨ 💎 ✨",
            "🌟 💎 🌟",
            "✨ 🌟 💎 🌟 ✨",
            "🔥 🌟 💎 🌟 🔥"
        ]
    else:
        # Crown animation for very long streaks
        animation_frames = [
            "👑",
            "✨ 👑 ✨",
            "🌟 👑 🌟",
            "💎 🌟 👑 🌟 💎",
            "🔥 💎 🌟 👑 🌟 💎 🔥"
        ]
    
    return animation_frames

def generate_cyberpunk_progress_animation(progress: float) -> List[str]:
    """
    Generate a cyberpunk-themed progress animation.
    
    Args:
        progress: Progress value between 0 and 1
        
    Returns:
        List of text strings representing animation frames
    """
    # Progress bar length
    bar_length = 10
    filled_length = int(bar_length * progress)
    
    # Create base animation frames
    animation_frames = []
    
    # Cyberpunk-themed progress bars
    for i in range(5):  # Number of different animation frames
        if i == 0:
            frame = f"[{'█' * filled_length}{'▒' * (bar_length - filled_length)}] {int(progress * 100)}%"
        elif i == 1:
            frame = f"[{'▓' * filled_length}{'░' * (bar_length - filled_length)}] {int(progress * 100)}%"
        elif i == 2:
            frame = f"<{'=' * filled_length}{' ' * (bar_length - filled_length)}> {int(progress * 100)}%"
        elif i == 3:
            frame = f"({filled_length * '|'}{'·' * (bar_length - filled_length)}) {int(progress * 100)}%"
        else:
            frame = f"《{'■' * filled_length}{'□' * (bar_length - filled_length)}》{int(progress * 100)}%"
        
        # Add cyberpunk flair
        if progress < 0.3:
            frame = f"🔌 INITIALIZING {frame}"
        elif progress < 0.6:
            frame = f"⚡ PROCESSING {frame}"
        elif progress < 0.9:
            frame = f"🔋 ACCELERATING {frame}"
        else:
            frame = f"💻 FINALIZING {frame}"
        
        animation_frames.append(frame)
    
    return animation_frames

def get_tier_progress(user_stats: Dict[str, Any]) -> Tuple[str, float, str, str]:
    """
    Calculate the user's progress towards the next tier.
    
    Args:
        user_stats: Dictionary with user statistics
        
    Returns:
        Tuple of (current tier, progress percentage, next tier, main requirement)
    """
    # Determine current tier
    current_tier_index = -1
    for i, tier in enumerate(ACHIEVEMENT_TIERS):
        # Check if all requirements are met
        requirements_met = True
        
        for stat_key, required_value in tier["requirements"].items():
            current_value = user_stats.get(stat_key, 0)
            if current_value < required_value:
                requirements_met = False
                break
        
        if requirements_met:
            current_tier_index = i
    
    # If no tier requirements met, set as pre-beginner
    if current_tier_index == -1:
        current_tier = "Newcomer"
        next_tier = ACHIEVEMENT_TIERS[0]["name"]
        
        # Calculate progress towards first tier
        tier = ACHIEVEMENT_TIERS[0]
        translations = user_stats.get("translations_count", 0)
        required_translations = tier["requirements"]["translations_count"]
        progress = min(1.0, translations / required_translations)
        
        main_requirement = f"Translations: {translations}/{required_translations}"
    
    # If at max tier
    elif current_tier_index == len(ACHIEVEMENT_TIERS) - 1:
        current_tier = ACHIEVEMENT_TIERS[current_tier_index]["name"]
        next_tier = "Legendary"  # Beyond the defined tiers
        progress = 1.0
        main_requirement = "All requirements met!"
    
    # Normal tier progression
    else:
        current_tier = ACHIEVEMENT_TIERS[current_tier_index]["name"]
        next_tier = ACHIEVEMENT_TIERS[current_tier_index + 1]["name"]
        
        # Calculate progress towards next tier based on translations (simplified)
        current_translations = user_stats.get("translations_count", 0)
        next_tier_translations = ACHIEVEMENT_TIERS[current_tier_index + 1]["requirements"]["translations_count"]
        current_tier_translations = ACHIEVEMENT_TIERS[current_tier_index]["requirements"]["translations_count"]
        
        # Calculate progress from current tier to next tier
        if next_tier_translations > current_tier_translations:
            progress = min(1.0, (current_translations - current_tier_translations) / 
                         (next_tier_translations - current_tier_translations))
        else:
            progress = 1.0
            
        main_requirement = f"Translations: {current_translations}/{next_tier_translations}"
    
    return (current_tier, progress, next_tier, main_requirement)