"""
Database module for Enhanced Language Learning Bot.
Handles database connections and operations.
"""

import os
import json
import logging
import datetime
from typing import Dict, List, Optional, Any, Tuple
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Database configuration
DB_URL = os.environ.get('DATABASE_URL')
USE_POSTGRES = DB_URL is not None

class Database:
    """Database wrapper class for the bot."""
    
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.initialize_db()
    
    def initialize_db(self):
        """Initialize database connection and create tables if needed."""
        try:
            if USE_POSTGRES:
                logger.info("Connecting to PostgreSQL database")
                self.conn = psycopg2.connect(DB_URL)
                self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            else:
                logger.info("Connecting to SQLite database")
                os.makedirs('data', exist_ok=True)
                self.conn = sqlite3.connect('data/language_bot.db')
                self.conn.row_factory = sqlite3.Row
                self.cursor = self.conn.cursor()
            
            self._create_tables()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            
    def ensure_connection(self):
        """Ensure database connection is active or reopen it if closed."""
        try:
            # Try a simple query to check if the connection is still active
            if self.conn and self.cursor:
                self.cursor.execute("SELECT 1")
                return True
        except (psycopg2.OperationalError, psycopg2.InterfaceError, 
                sqlite3.OperationalError, sqlite3.ProgrammingError) as e:
            logger.warning(f"Database connection lost: {e}. Attempting to reconnect...")
            try:
                # Close any existing connection
                self.close()
                # Reopen the connection
                if USE_POSTGRES:
                    self.conn = psycopg2.connect(DB_URL)
                    self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
                else:
                    self.conn = sqlite3.connect('data/language_bot.db')
                    self.conn.row_factory = sqlite3.Row
                    self.cursor = self.conn.cursor()
                logger.info("Successfully reconnected to the database")
                return True
            except Exception as e2:
                logger.error(f"Failed to reconnect to database: {e2}")
                return False
        except Exception as e:
            logger.error(f"Unknown database error: {e}")
            return False
            
        return True
            
    def update_vocabulary_example(self, vocab_id: int, translation: str, context: str) -> bool:
        """Update vocabulary entry with translation and context."""
        try:
            if USE_POSTGRES:
                self.cursor.execute(
                    "UPDATE vocabulary SET translation = %s, context = %s WHERE id = %s",
                    (translation, context, vocab_id)
                )
            else:
                self.cursor.execute(
                    "UPDATE vocabulary SET translation = ?, context = ? WHERE id = ?",
                    (translation, context, vocab_id)
                )
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating vocabulary example: {e}")
            self.conn.rollback()
            return False
            # Fallback to SQLite if PostgreSQL connection fails
            if USE_POSTGRES:
                logger.info("Falling back to SQLite database")
                os.makedirs('data', exist_ok=True)
                self.conn = sqlite3.connect('data/language_bot.db')
                self.conn.row_factory = sqlite3.Row
                self.cursor = self.conn.cursor()
                self._create_tables()
    
    def _create_tables(self):
        """Create required tables if they don't exist."""
        try:
            # Users table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    chat_id BIGINT PRIMARY KEY,
                    username TEXT,
                    selected_languages TEXT,
                    learning_level TEXT DEFAULT 'advanced',
                    notification_time TEXT,
                    notify_daily BOOLEAN DEFAULT FALSE,
                    notify_review BOOLEAN DEFAULT FALSE,
                    notify_facts BOOLEAN DEFAULT FALSE,
                    in_chat_mode BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP,
                    last_active TIMESTAMP
                )
            ''')
            
            # Vocabulary table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS vocabulary (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT,
                    source_language TEXT,
                    target_language TEXT,
                    word TEXT,
                    translation TEXT,
                    context TEXT,
                    stage INTEGER DEFAULT 0,
                    correct_count INTEGER DEFAULT 0,
                    incorrect_count INTEGER DEFAULT 0,
                    next_review TIMESTAMP,
                    last_reviewed TIMESTAMP,
                    created_at TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES users(chat_id)
                )
            ''')
            
            # Statistics table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT,
                    vocabulary_count INTEGER DEFAULT 0,
                    translations_requested INTEGER DEFAULT 0,
                    games_played INTEGER DEFAULT 0,
                    correct_answers INTEGER DEFAULT 0,
                    incorrect_answers INTEGER DEFAULT 0,
                    streak_days INTEGER DEFAULT 0,
                    last_activity DATE,
                    FOREIGN KEY (chat_id) REFERENCES users(chat_id)
                )
            ''')
            
            # Achievements table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS achievements (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT,
                    achievement_type TEXT,
                    achievement_data TEXT,
                    earned_at TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES users(chat_id)
                )
            ''')
            
            # Translation history table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS translation_history (
                    id SERIAL PRIMARY KEY,
                    chat_id BIGINT,
                    source_text TEXT,
                    source_language TEXT,
                    translations JSONB,
                    saved BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP,
                    FOREIGN KEY (chat_id) REFERENCES users(chat_id)
                )
            ''')
            
            self.conn.commit()
            logger.info("Tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            self.conn.rollback()
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def get_user(self, chat_id):
        """Get user data by chat ID."""
        try:
            if not self.ensure_connection():
                return None
                
            self.cursor.execute("SELECT * FROM users WHERE chat_id = %s", (chat_id,))
            user = self.cursor.fetchone()
            return dict(user) if user else None
        except Exception as e:
            logger.error(f"Error getting user: {e}")
            return None
    
    def create_or_update_user(self, chat_id, username=None, selected_languages=None, learning_level=None):
        """Create a new user or update existing user."""
        try:
            if not self.ensure_connection():
                return False
                
            # Check if user exists
            self.cursor.execute("SELECT * FROM users WHERE chat_id = %s", (chat_id,))
            user = self.cursor.fetchone()
            
            now = datetime.datetime.now()
            
            if user:
                # Update existing user
                query = "UPDATE users SET last_active = %s"
                params = [now]
                
                if username:
                    query += ", username = %s"
                    params.append(username)
                
                if selected_languages:
                    query += ", selected_languages = %s"
                    params.append(json.dumps(selected_languages))
                
                if learning_level:
                    query += ", learning_level = %s"
                    params.append(learning_level)
                
                query += " WHERE chat_id = %s"
                params.append(chat_id)
                
                self.cursor.execute(query, params)
            else:
                # Create new user
                self.cursor.execute('''
                    INSERT INTO users (
                        chat_id, username, selected_languages, learning_level, created_at, last_active, in_chat_mode
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ''', (
                    chat_id,
                    username,
                    json.dumps(selected_languages) if selected_languages else None,
                    learning_level or 'advanced',
                    now,
                    now,
                    False  # Default to translator mode, not chat mode
                ))
                
                # Initialize statistics for new user
                self.cursor.execute('''
                    INSERT INTO statistics (chat_id, last_activity)
                    VALUES (%s, %s)
                ''', (chat_id, now.date()))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error creating/updating user: {e}")
            self.conn.rollback()
            return False
    
    def update_user_preferences(self, chat_id, preferences):
        """Update user preferences."""
        try:
            if not self.ensure_connection():
                return False
                
            fields = []
            params = []
            
            for key, value in preferences.items():
                if key in ('selected_languages',):
                    value = json.dumps(value)
                
                fields.append(f"{key} = %s")
                params.append(value)
            
            if not fields:
                return False
            
            params.append(chat_id)
            query = f"UPDATE users SET {', '.join(fields)} WHERE chat_id = %s"
            
            self.cursor.execute(query, params)
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            self.conn.rollback()
            return False
            
    def update_chat_mode(self, chat_id, in_chat_mode):
        """Update user's chat mode (AI chat vs. translation mode).
        
        Args:
            chat_id (int): User's chat ID
            in_chat_mode (bool): True if in AI chat mode, False if in translation mode
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.ensure_connection():
                return False
            
            # Store the chat mode in user preferences with a simple UPDATE
            # We'll use the method above which already has error handling
            return self.update_user_preferences(chat_id, {'in_chat_mode': in_chat_mode})
        except Exception as e:
            logger.error(f"Error updating chat mode: {e}")
            return False
            
    def set_chat_mode(self, chat_id, mode):
        """Set user's chat mode by string name or boolean.
        
        Args:
            chat_id (int): User's chat ID
            mode (str or bool): 'chat' for AI chat mode, 'translator' for translation mode,
                               or True for AI chat mode, False for translation mode
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Handle both string and boolean inputs
            if isinstance(mode, bool):
                in_chat_mode = mode
            elif isinstance(mode, str):
                in_chat_mode = (mode.lower() == 'chat')
            else:
                in_chat_mode = bool(mode)  # Fallback conversion
            
            # Use existing update_chat_mode method
            return self.update_chat_mode(chat_id, in_chat_mode)
        except Exception as e:
            logger.error(f"Error setting chat mode: {e}")
            return False
    
    def add_vocabulary(self, chat_id, source_language, target_language, word, translation, context=None):
        """Add vocabulary item for a user."""
        try:
            if not self.ensure_connection():
                return None
                
            now = datetime.datetime.now()
            
            # Check if vocabulary already exists
            self.cursor.execute('''
                SELECT id FROM vocabulary 
                WHERE chat_id = %s AND word = %s AND target_language = %s
            ''', (chat_id, word, target_language))
            
            existing = self.cursor.fetchone()
            
            if existing:
                # Update existing vocabulary
                self.cursor.execute('''
                    UPDATE vocabulary SET 
                    translation = %s,
                    context = %s,
                    last_reviewed = %s
                    WHERE id = %s
                ''', (translation, context, now, existing['id']))
                vocab_id = existing['id']
            else:
                # Add new vocabulary
                next_review = now + datetime.timedelta(days=1)
                
                self.cursor.execute('''
                    INSERT INTO vocabulary (
                        chat_id, source_language, target_language, word, translation, 
                        context, next_review, last_reviewed, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                ''', (
                    chat_id, source_language, target_language, word, translation,
                    context, next_review, now, now
                ))
                
                vocab_id = self.cursor.fetchone()['id']
                
                # Update vocabulary count in statistics
                self.cursor.execute('''
                    UPDATE statistics SET 
                    vocabulary_count = vocabulary_count + 1,
                    last_activity = %s
                    WHERE chat_id = %s
                ''', (now.date(), chat_id))
            
            self.conn.commit()
            return vocab_id
        except Exception as e:
            logger.error(f"Error adding vocabulary: {e}")
            self.conn.rollback()
            return None
    
    def get_vocabulary(self, chat_id, target_language=None):
        """Get all vocabulary for a user."""
        try:
            if not self.ensure_connection():
                return []
                
            query = "SELECT * FROM vocabulary WHERE chat_id = %s"
            params = [chat_id]
            
            if target_language:
                query += " AND target_language = %s"
                params.append(target_language)
            
            self.cursor.execute(query, params)
            return [dict(row) for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting vocabulary: {e}")
            return []
    
    def get_due_vocabulary(self, chat_id, limit=10):
        """Get vocabulary items due for review."""
        try:
            if not self.ensure_connection():
                return []
                
            now = datetime.datetime.now()
            
            self.cursor.execute('''
                SELECT * FROM vocabulary
                WHERE chat_id = %s AND next_review <= %s
                ORDER BY next_review
                LIMIT %s
            ''', (chat_id, now, limit))
            
            return [dict(row) for row in self.cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting due vocabulary: {e}")
            return []
    
    def update_vocabulary_review(self, vocab_id, correct):
        """Update vocabulary after review."""
        try:
            if not self.ensure_connection():
                return False
                
            # Get current vocabulary data
            self.cursor.execute("SELECT * FROM vocabulary WHERE id = %s", (vocab_id,))
            vocab = self.cursor.fetchone()
            
            if not vocab:
                return False
            
            # Update review data
            now = datetime.datetime.now()
            stage = vocab['stage']
            
            if correct:
                # Move to next stage if correct
                stage = min(5, stage + 1)
                # Increment correct count
                correct_count = vocab['correct_count'] + 1
                
                # Calculate next review based on spaced repetition
                days_until_next_review = {
                    0: 1,   # Stage 0: review next day
                    1: 3,   # Stage 1: review in 3 days
                    2: 7,   # Stage 2: review in 1 week
                    3: 14,  # Stage 3: review in 2 weeks
                    4: 30,  # Stage 4: review in 1 month
                    5: 90   # Stage 5: review in 3 months
                }
                
                next_review = now + datetime.timedelta(days=days_until_next_review[stage])
                
                self.cursor.execute('''
                    UPDATE vocabulary SET
                    stage = %s,
                    correct_count = %s,
                    next_review = %s,
                    last_reviewed = %s
                    WHERE id = %s
                ''', (stage, correct_count, next_review, now, vocab_id))
                
                # Update statistics
                self.cursor.execute('''
                    UPDATE statistics SET
                    correct_answers = correct_answers + 1,
                    last_activity = %s
                    WHERE chat_id = %s
                ''', (now.date(), vocab['chat_id']))
            else:
                # Reset to stage 0 if incorrect
                stage = max(0, stage - 1)
                # Increment incorrect count
                incorrect_count = vocab['incorrect_count'] + 1
                
                # Review again tomorrow
                next_review = now + datetime.timedelta(days=1)
                
                self.cursor.execute('''
                    UPDATE vocabulary SET
                    stage = %s,
                    incorrect_count = %s,
                    next_review = %s,
                    last_reviewed = %s
                    WHERE id = %s
                ''', (stage, incorrect_count, next_review, now, vocab_id))
                
                # Update statistics
                self.cursor.execute('''
                    UPDATE statistics SET
                    incorrect_answers = incorrect_answers + 1,
                    last_activity = %s
                    WHERE chat_id = %s
                ''', (now.date(), vocab['chat_id']))
            
            self.conn.commit()
            
            # Check for achievements
            self._check_vocabulary_achievements(vocab['chat_id'])
            
            return True
        except Exception as e:
            logger.error(f"Error updating vocabulary review: {e}")
            self.conn.rollback()
            return False
    
    def get_user_statistics(self, chat_id):
        """Get user statistics."""
        try:
            if not self.ensure_connection():
                return None
                
            self.cursor.execute("SELECT * FROM statistics WHERE chat_id = %s", (chat_id,))
            stats = self.cursor.fetchone()
            
            if not stats:
                return None
            
            stats_dict = dict(stats)
            
            # Calculate additional statistics
            total_answers = stats_dict.get('correct_answers', 0) + stats_dict.get('incorrect_answers', 0)
            if total_answers > 0:
                stats_dict['accuracy'] = round(stats_dict.get('correct_answers', 0) / total_answers * 100, 1)
            else:
                stats_dict['accuracy'] = 0
            
            # Get vocabulary mastery data
            self.cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN stage >= 4 THEN 1 ELSE 0 END) as mastered
                FROM vocabulary
                WHERE chat_id = %s
            ''', (chat_id,))
            
            mastery_data = self.cursor.fetchone()
            if mastery_data:
                stats_dict['total_vocabulary'] = mastery_data['total']
                stats_dict['mastered_vocabulary'] = mastery_data['mastered']
                
                if mastery_data['total'] > 0:
                    stats_dict['mastery_percentage'] = round(mastery_data['mastered'] / mastery_data['total'] * 100, 1)
                else:
                    stats_dict['mastery_percentage'] = 0
            
            # Get streak data
            stats_dict['streak'] = self._calculate_streak(chat_id)
            
            # Get achievements
            self.cursor.execute("SELECT * FROM achievements WHERE chat_id = %s", (chat_id,))
            stats_dict['achievements'] = [dict(row) for row in self.cursor.fetchall()]
            
            return stats_dict
        except Exception as e:
            logger.error(f"Error getting user statistics: {e}")
            return None
    
    def update_translation_count(self, chat_id):
        """Update translation request count."""
        try:
            if not self.ensure_connection():
                return False
                
            now = datetime.datetime.now()
            
            self.cursor.execute('''
                UPDATE statistics SET
                translations_requested = translations_requested + 1,
                last_activity = %s
                WHERE chat_id = %s
            ''', (now.date(), chat_id))
            
            self.conn.commit()
            
            # Check for streak
            self._update_streak(chat_id)
            
            # Check for achievements
            self._check_translation_achievements(chat_id)
            
            return True
        except Exception as e:
            logger.error(f"Error updating translation count: {e}")
            self.conn.rollback()
            return False
    
    def save_translation_history(self, chat_id, source_text, source_language, translations):
        """Save translation to history.
        
        Args:
            chat_id: User's chat ID
            source_text: Original text that was translated
            source_language: Source language code
            translations: Dictionary of translations (language code to translation text)
            
        Returns:
            int: ID of the saved translation or None if error
        """
        try:
            if not self.ensure_connection():
                return None
                
            now = datetime.datetime.now()
            
            self.cursor.execute('''
                INSERT INTO translation_history (
                    chat_id, source_text, source_language, translations, created_at
                ) VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            ''', (
                chat_id, 
                source_text, 
                source_language, 
                json.dumps(translations), 
                now
            ))
            
            history_id = self.cursor.fetchone()['id']
            self.conn.commit()
            return history_id
        except Exception as e:
            logger.error(f"Error saving translation history: {e}")
            self.conn.rollback()
            return None
    
    def mark_translation_saved(self, history_id, saved=True):
        """Mark a translation as saved/favorited.
        
        Args:
            history_id: ID of the translation history entry
            saved: Boolean indicating whether it's saved/favorited
            
        Returns:
            bool: Success status
        """
        try:
            if not self.ensure_connection():
                return False
                
            self.cursor.execute('''
                UPDATE translation_history SET
                saved = %s
                WHERE id = %s
            ''', (saved, history_id))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error marking translation as saved: {e}")
            self.conn.rollback()
            return False
    
    def get_translation_history(self, chat_id, limit=10, saved_only=False):
        """Get translation history for a user.
        
        Args:
            chat_id: User's chat ID
            limit: Maximum number of entries to return
            saved_only: If True, only return saved/favorited translations
            
        Returns:
            list: List of translation history entries
        """
        try:
            if not self.ensure_connection():
                return []
                
            query = "SELECT * FROM translation_history WHERE chat_id = %s"
            params = [chat_id]
            
            if saved_only:
                query += " AND saved = TRUE"
                
            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)
            
            self.cursor.execute(query, params)
            
            # Convert JSONB column back to dictionary
            history = []
            for row in self.cursor.fetchall():
                entry = dict(row)
                if isinstance(entry['translations'], str):
                    entry['translations'] = json.loads(entry['translations'])
                history.append(entry)
                
            return history
        except Exception as e:
            logger.error(f"Error getting translation history: {e}")
            return []
    
    def delete_translation_history(self, history_id=None, chat_id=None):
        """Delete translation history entries.
        
        Args:
            history_id: Specific history ID to delete (optional)
            chat_id: Delete all history for this user (optional)
            
        Note: At least one of history_id or chat_id must be provided
        
        Returns:
            bool: Success status
        """
        try:
            if not self.ensure_connection():
                return False
                
            if not history_id and not chat_id:
                return False
                
            if history_id:
                self.cursor.execute('''
                    DELETE FROM translation_history
                    WHERE id = %s
                ''', (history_id,))
            else:
                self.cursor.execute('''
                    DELETE FROM translation_history
                    WHERE chat_id = %s
                ''', (chat_id,))
                
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error deleting translation history: {e}")
            self.conn.rollback()
            return False
    
    def update_game_stats(self, chat_id, correct=True):
        """Update game statistics."""
        try:
            if not self.ensure_connection():
                return False
                
            now = datetime.datetime.now()
            
            query = '''
                UPDATE statistics SET
                games_played = games_played + 1,
            '''
            
            if correct:
                query += 'correct_answers = correct_answers + 1,'
            else:
                query += 'incorrect_answers = incorrect_answers + 1,'
            
            query += '''
                last_activity = %s
                WHERE chat_id = %s
            '''
            
            self.cursor.execute(query, (now.date(), chat_id))
            self.conn.commit()
            
            # Update streak
            self._update_streak(chat_id)
            
            # Check for achievements
            self._check_game_achievements(chat_id)
            
            return True
        except Exception as e:
            logger.error(f"Error updating game stats: {e}")
            self.conn.rollback()
            return False
    
    def _update_streak(self, chat_id):
        """Update user streak if they've been active today."""
        try:
            if not self.ensure_connection():
                return False
                
            now = datetime.datetime.now()
            today = now.date()
            
            # Get last activity date
            self.cursor.execute('''
                SELECT last_activity FROM statistics
                WHERE chat_id = %s
            ''', (chat_id,))
            
            stat = self.cursor.fetchone()
            if not stat:
                return False
            
            last_activity = stat['last_activity']
            
            if last_activity is None:
                # First activity, start streak
                self.cursor.execute('''
                    UPDATE statistics SET
                    streak_days = 1,
                    last_activity = %s
                    WHERE chat_id = %s
                ''', (today, chat_id))
            elif last_activity < today:
                # Check if this is consecutive day
                yesterday = today - datetime.timedelta(days=1)
                
                if last_activity == yesterday:
                    # Consecutive day, increment streak
                    self.cursor.execute('''
                        UPDATE statistics SET
                        streak_days = streak_days + 1,
                        last_activity = %s
                        WHERE chat_id = %s
                    ''', (today, chat_id))
                else:
                    # Not consecutive, reset streak
                    self.cursor.execute('''
                        UPDATE statistics SET
                        streak_days = 1,
                        last_activity = %s
                        WHERE chat_id = %s
                    ''', (today, chat_id))
            
            # Already updated today, do nothing to streak
            
            self.conn.commit()
            
            # Check for streak achievements
            self._check_streak_achievements(chat_id)
            
            return True
        except Exception as e:
            logger.error(f"Error updating streak: {e}")
            self.conn.rollback()
            return False
    
    def _calculate_streak(self, chat_id):
        """Calculate current streak."""
        try:
            if not self.ensure_connection():
                return 0
                
            self.cursor.execute('''
                SELECT streak_days, last_activity FROM statistics
                WHERE chat_id = %s
            ''', (chat_id,))
            
            stat = self.cursor.fetchone()
            if not stat:
                return 0
            
            streak = stat['streak_days'] or 0
            last_activity = stat['last_activity']
            
            if not last_activity:
                return 0
            
            # Check if streak is still valid
            now = datetime.datetime.now()
            today = now.date()
            yesterday = today - datetime.timedelta(days=1)
            
            if last_activity < yesterday:
                # Streak broken, reset to 0
                self.cursor.execute('''
                    UPDATE statistics SET streak_days = 0
                    WHERE chat_id = %s
                ''', (chat_id,))
                self.conn.commit()
                return 0
            
            return streak
        except Exception as e:
            logger.error(f"Error calculating streak: {e}")
            return 0
    
    def _check_vocabulary_achievements(self, chat_id):
        """Check and award vocabulary-related achievements."""
        try:
            if not self.ensure_connection():
                return False
                
            # Get vocabulary count
            self.cursor.execute('''
                SELECT COUNT(*) as count FROM vocabulary
                WHERE chat_id = %s
            ''', (chat_id,))
            
            result = self.cursor.fetchone()
            count = result['count'] if result else 0
            
            # Check for achievements
            achievements = []
            
            if count >= 10:
                achievements.append(('vocabulary_10', 'Added 10 words to vocabulary'))
            
            if count >= 50:
                achievements.append(('vocabulary_50', 'Added 50 words to vocabulary'))
            
            if count >= 100:
                achievements.append(('vocabulary_100', 'Added 100 words to vocabulary'))
            
            if count >= 500:
                achievements.append(('vocabulary_500', 'Added 500 words to vocabulary'))
            
            # Check mastery
            self.cursor.execute('''
                SELECT COUNT(*) as count FROM vocabulary
                WHERE chat_id = %s AND stage >= 4
            ''', (chat_id,))
            
            result = self.cursor.fetchone()
            mastered = result['count'] if result else 0
            
            if mastered >= 10:
                achievements.append(('mastery_10', 'Mastered 10 words'))
            
            if mastered >= 50:
                achievements.append(('mastery_50', 'Mastered 50 words'))
            
            if mastered >= 100:
                achievements.append(('mastery_100', 'Mastered 100 words'))
            
            # Award achievements
            for achievement_type, achievement_data in achievements:
                self._award_achievement(chat_id, achievement_type, achievement_data)
        except Exception as e:
            logger.error(f"Error checking vocabulary achievements: {e}")
    
    def _check_translation_achievements(self, chat_id):
        """Check and award translation-related achievements."""
        try:
            if not self.ensure_connection():
                return False
                
            # Get translation count
            self.cursor.execute('''
                SELECT translations_requested FROM statistics
                WHERE chat_id = %s
            ''', (chat_id,))
            
            result = self.cursor.fetchone()
            count = result['translations_requested'] if result else 0
            
            # Check for achievements
            achievements = []
            
            if count >= 10:
                achievements.append(('translations_10', 'Requested 10 translations'))
            
            if count >= 50:
                achievements.append(('translations_50', 'Requested 50 translations'))
            
            if count >= 100:
                achievements.append(('translations_100', 'Requested 100 translations'))
            
            if count >= 500:
                achievements.append(('translations_500', 'Requested 500 translations'))
            
            # Award achievements
            for achievement_type, achievement_data in achievements:
                self._award_achievement(chat_id, achievement_type, achievement_data)
        except Exception as e:
            logger.error(f"Error checking translation achievements: {e}")
    
    def _check_game_achievements(self, chat_id):
        """Check and award game-related achievements."""
        try:
            if not self.ensure_connection():
                return False
                
            # Get game stats
            self.cursor.execute('''
                SELECT games_played, correct_answers, incorrect_answers FROM statistics
                WHERE chat_id = %s
            ''', (chat_id,))
            
            result = self.cursor.fetchone()
            if not result:
                return
            
            games = result['games_played']
            correct = result['correct_answers']
            
            # Check for achievements
            achievements = []
            
            if games >= 10:
                achievements.append(('games_10', 'Played 10 games'))
            
            if games >= 50:
                achievements.append(('games_50', 'Played 50 games'))
            
            if games >= 100:
                achievements.append(('games_100', 'Played 100 games'))
            
            if correct >= 10:
                achievements.append(('correct_10', 'Answered 10 questions correctly'))
            
            if correct >= 50:
                achievements.append(('correct_50', 'Answered 50 questions correctly'))
            
            if correct >= 100:
                achievements.append(('correct_100', 'Answered 100 questions correctly'))
            
            # Check accuracy
            total = correct + (result['incorrect_answers'] or 0)
            if total >= 20 and correct / total >= 0.9:
                achievements.append(('accuracy_90', 'Achieved 90% accuracy in games'))
            
            # Award achievements
            for achievement_type, achievement_data in achievements:
                self._award_achievement(chat_id, achievement_type, achievement_data)
        except Exception as e:
            logger.error(f"Error checking game achievements: {e}")
    
    def _check_streak_achievements(self, chat_id):
        """Check and award streak-related achievements."""
        try:
            if not self.ensure_connection():
                return False
                
            # Get streak
            self.cursor.execute('''
                SELECT streak_days FROM statistics
                WHERE chat_id = %s
            ''', (chat_id,))
            
            result = self.cursor.fetchone()
            streak = result['streak_days'] if result else 0
            
            # Check for achievements
            achievements = []
            
            if streak >= 3:
                achievements.append(('streak_3', '3-day streak'))
            
            if streak >= 7:
                achievements.append(('streak_7', '7-day streak'))
            
            if streak >= 14:
                achievements.append(('streak_14', '14-day streak'))
            
            if streak >= 30:
                achievements.append(('streak_30', '30-day streak'))
            
            # Award achievements
            for achievement_type, achievement_data in achievements:
                self._award_achievement(chat_id, achievement_type, achievement_data)
        except Exception as e:
            logger.error(f"Error checking streak achievements: {e}")
    
    def _award_achievement(self, chat_id, achievement_type, achievement_data):
        """Award an achievement if not already awarded."""
        try:
            if not self.ensure_connection():
                return False
                
            # Check if already awarded
            self.cursor.execute('''
                SELECT id FROM achievements
                WHERE chat_id = %s AND achievement_type = %s
            ''', (chat_id, achievement_type))
            
            if self.cursor.fetchone():
                # Already awarded
                return False
            
            # Award new achievement
            now = datetime.datetime.now()
            
            self.cursor.execute('''
                INSERT INTO achievements (chat_id, achievement_type, achievement_data, earned_at)
                VALUES (%s, %s, %s, %s)
            ''', (chat_id, achievement_type, achievement_data, now))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error awarding achievement: {e}")
            self.conn.rollback()
            return False
    
    def set_notification_preferences(self, chat_id, notify_daily=None, notify_review=None, 
                                    notify_facts=None, notification_time=None):
        """Set notification preferences for a user."""
        try:
            if not self.ensure_connection():
                return False
                
            query = "UPDATE users SET"
            params = []
            
            if notify_daily is not None:
                query += " notify_daily = %s,"
                params.append(notify_daily)
            
            if notify_review is not None:
                query += " notify_review = %s,"
                params.append(notify_review)
            
            if notify_facts is not None:
                query += " notify_facts = %s,"
                params.append(notify_facts)
            
            if notification_time is not None:
                query += " notification_time = %s,"
                params.append(notification_time)
            
            # Remove trailing comma
            query = query.rstrip(',')
            
            query += " WHERE chat_id = %s"
            params.append(chat_id)
            
            self.cursor.execute(query, params)
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error setting notification preferences: {e}")
            self.conn.rollback()
            return False
    
    def get_notification_preferences(self, chat_id):
        """Get notification preferences for a user."""
        try:
            if not self.ensure_connection():
                return None
                
            self.cursor.execute('''
                SELECT notify_daily, notify_review, notify_facts, notification_time
                FROM users WHERE chat_id = %s
            ''', (chat_id,))
            
            result = self.cursor.fetchone()
            return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error getting notification preferences: {e}")
            return None
            
    def set_user_preference(self, chat_id, preference_name, preference_value):
        """Set a generic user preference.
        
        Args:
            chat_id (int): The user's chat ID
            preference_name (str): The name of the preference
            preference_value: The value to set (can be string, boolean, int, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.ensure_connection():
                return False
            
            # Check if user exists first
            self.cursor.execute(
                "SELECT 1 FROM users WHERE chat_id = %s",
                (chat_id,)
            )
            
            if not self.cursor.fetchone():
                # Create user record if it doesn't exist
                self.cursor.execute(
                    "INSERT INTO users (chat_id) VALUES (%s)",
                    (chat_id,)
                )
            
            # Convert preference value to JSON string if it's not a simple type
            if not isinstance(preference_value, (str, int, float, bool, type(None))):
                preference_value = json.dumps(preference_value)
                
            # Check if preference column exists
            self.cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'users' AND column_name = %s
            """, (preference_name,))
            
            if not self.cursor.fetchone():
                # Preference column doesn't exist, so we'll store in preferences JSON field
                # First, get the current preferences
                self.cursor.execute(
                    "SELECT preferences FROM users WHERE chat_id = %s",
                    (chat_id,)
                )
                
                result = self.cursor.fetchone()
                if result and result['preferences']:
                    preferences = json.loads(result['preferences'])
                else:
                    preferences = {}
                
                # Update the preference
                preferences[preference_name] = preference_value
                
                # Save back to database
                self.cursor.execute(
                    "UPDATE users SET preferences = %s WHERE chat_id = %s",
                    (json.dumps(preferences), chat_id)
                )
            else:
                # Preference column exists, update directly
                query = f"UPDATE users SET {preference_name} = %s WHERE chat_id = %s"
                self.cursor.execute(query, (preference_value, chat_id))
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error setting user preference {preference_name}: {e}")
            self.conn.rollback()
            return False

# Initialize global database instance
db = Database()