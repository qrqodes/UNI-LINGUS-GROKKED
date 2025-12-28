"""
Learning module for the translation bot.
Implements spaced repetition and vocabulary tracking algorithms.
"""
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from vocabulary_data import get_vocab_by_level, get_word_options, get_translation_challenge

# Spaced repetition intervals in days
SR_INTERVALS = [1, 3, 7, 14, 30, 90, 180]

class VocabularyItem:
    """A vocabulary item with spaced repetition tracking."""
    
    def __init__(self, word_data: Dict[str, Any], level: str = 'beginner'):
        self.word = word_data['word']
        self.definition = word_data.get('definition', '')
        self.example = word_data.get('example', '')
        self.synonym = word_data.get('synonym', '')
        self.translations = word_data.get('translations', {})
        self.level = level
        
        # Spaced repetition data
        self.learning_stage = 0  # Index into SR_INTERVALS
        self.next_review = datetime.now()
        self.review_count = 0
        self.correct_count = 0
        self.last_reviewed = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            'word': self.word,
            'definition': self.definition,
            'example': self.example,
            'synonym': self.synonym,
            'translations': self.translations,
            'level': self.level,
            'learning_stage': self.learning_stage,
            'next_review': self.next_review.timestamp() if self.next_review else None,
            'review_count': self.review_count,
            'correct_count': self.correct_count,
            'last_reviewed': self.last_reviewed.timestamp() if self.last_reviewed else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VocabularyItem':
        """Create a VocabularyItem from a dictionary."""
        item = cls({
            'word': data['word'],
            'definition': data.get('definition', ''),
            'example': data.get('example', ''),
            'synonym': data.get('synonym', ''),
            'translations': data.get('translations', {})
        }, data.get('level', 'beginner'))
        
        item.learning_stage = data.get('learning_stage', 0)
        
        next_review = data.get('next_review')
        if next_review:
            item.next_review = datetime.fromtimestamp(next_review)
        
        item.review_count = data.get('review_count', 0)
        item.correct_count = data.get('correct_count', 0)
        
        last_reviewed = data.get('last_reviewed')
        if last_reviewed:
            item.last_reviewed = datetime.fromtimestamp(last_reviewed)
        
        return item
    
    def process_review(self, correct: bool):
        """
        Update item status after a review.
        
        Args:
            correct: Whether the user got this item correct
        """
        self.review_count += 1
        self.last_reviewed = datetime.now()
        
        if correct:
            self.correct_count += 1
            # Move to next stage in spaced repetition
            if self.learning_stage < len(SR_INTERVALS) - 1:
                self.learning_stage += 1
        else:
            # Go back a stage if wrong (but not below 0)
            if self.learning_stage > 0:
                self.learning_stage -= 1
        
        # Set next review time based on current stage
        days = SR_INTERVALS[self.learning_stage]
        self.next_review = datetime.now() + timedelta(days=days)
    
    def is_due_for_review(self) -> bool:
        """Check if this item is due for review."""
        return datetime.now() >= self.next_review
    
    def mastery_percentage(self) -> float:
        """Calculate mastery percentage based on correct answers and current stage."""
        if self.review_count == 0:
            return 0.0
        
        # Base percentage on correct answers
        accuracy = (self.correct_count / self.review_count) * 100
        
        # Factor in the learning stage (higher stage = more mastery)
        stage_factor = (self.learning_stage / (len(SR_INTERVALS) - 1)) * 100
        
        # Weight: 70% accuracy, 30% stage progression
        return (accuracy * 0.7) + (stage_factor * 0.3)


class LearningManager:
    """Manages vocabulary learning and spaced repetition."""
    
    def __init__(self):
        self.user_vocabulary = {}  # Dict to store user vocabulary items by chat_id
    
    def get_user_vocabulary(self, chat_id: int) -> Dict[str, VocabularyItem]:
        """Get a user's vocabulary items."""
        if chat_id not in self.user_vocabulary:
            self.user_vocabulary[chat_id] = {}
        return self.user_vocabulary[chat_id]
    
    def add_word_to_vocabulary(self, chat_id: int, word_data: Dict[str, Any], level: str = 'beginner'):
        """Add a word to a user's vocabulary."""
        vocabulary = self.get_user_vocabulary(chat_id)
        word = word_data['word']
        
        if word not in vocabulary:
            vocabulary[word] = VocabularyItem(word_data, level)
        
        return vocabulary[word]
    
    def get_due_for_review(self, chat_id: int, max_words: int = 5) -> List[VocabularyItem]:
        """Get words that are due for review."""
        vocabulary = self.get_user_vocabulary(chat_id)
        due_words = [item for item in vocabulary.values() if item.is_due_for_review()]
        
        # Sort by oldest review date first
        due_words.sort(key=lambda x: x.next_review)
        
        return due_words[:max_words]
    
    def get_mastery_statistics(self, chat_id: int) -> Dict[str, Any]:
        """Get statistics on a user's vocabulary mastery."""
        vocabulary = self.get_user_vocabulary(chat_id)
        
        if not vocabulary:
            return {
                'total_words': 0,
                'mastered_words': 0,
                'average_mastery': 0,
                'review_count': 0,
                'words_by_level': {'beginner': 0, 'intermediate': 0, 'advanced': 0}
            }
        
        mastery_threshold = 80.0  # Consider a word mastered if mastery is over 80%
        mastered_count = 0
        total_mastery = 0.0
        total_reviews = 0
        levels = {'beginner': 0, 'intermediate': 0, 'advanced': 0}
        
        for word_item in vocabulary.values():
            mastery = word_item.mastery_percentage()
            total_mastery += mastery
            if mastery >= mastery_threshold:
                mastered_count += 1
            
            total_reviews += word_item.review_count
            levels[word_item.level] += 1
        
        average_mastery = total_mastery / len(vocabulary) if vocabulary else 0
        
        return {
            'total_words': len(vocabulary),
            'mastered_words': mastered_count,
            'average_mastery': average_mastery,
            'review_count': total_reviews,
            'words_by_level': levels
        }
    
    def create_review_session(self, chat_id: int, max_words: int = 5) -> List[Dict[str, Any]]:
        """Create a review session with due words and options."""
        due_words = self.get_due_for_review(chat_id, max_words)
        
        if not due_words:
            # If no words are due, suggest some new words from their level
            return []
        
        # Create review items for each word
        review_items = []
        for word_item in due_words:
            # For each word, create a review challenge
            target_language = random.choice(list(word_item.translations.keys()))
            
            # Create a quiz item with options
            options = self._get_quiz_options(word_item, target_language)
            
            review_items.append({
                'word': word_item.word,
                'definition': word_item.definition,
                'target_language': target_language,
                'correct_answer': word_item.translations[target_language],
                'options': options,
                'word_id': word_item.word  # Used to identify this word when processing the answer
            })
        
        return review_items
    
    def _get_quiz_options(self, word_item: VocabularyItem, target_language: str) -> List[str]:
        """Get quiz options for a vocabulary item."""
        # The correct answer
        correct = word_item.translations[target_language]
        options = [correct]
        
        # Add 3 random distractors from the same language
        all_words = get_vocab_by_level(word_item.level)
        distractors = []
        
        for word_data in all_words:
            if word_data['word'] != word_item.word and target_language in word_data.get('translations', {}):
                distractors.append(word_data['translations'][target_language])
        
        # If we don't have enough distractors, just use some random strings
        if len(distractors) >= 3:
            options.extend(random.sample(distractors, 3))
        else:
            # Add what we have
            options.extend(distractors)
            # Add random translations from other words to fill in
            while len(options) < 4:
                random_word = random.choice(all_words)
                if random_word['word'] != word_item.word:
                    for lang, trans in random_word.get('translations', {}).items():
                        if lang != target_language and trans not in options:
                            options.append(trans)
                            break
        
        # Shuffle the options
        random.shuffle(options)
        return options
    
    def process_review_answer(self, chat_id: int, word_id: str, correct: bool) -> Optional[VocabularyItem]:
        """Process a user's answer to a review question."""
        vocabulary = self.get_user_vocabulary(chat_id)
        
        if word_id not in vocabulary:
            return None
        
        word_item = vocabulary[word_id]
        word_item.process_review(correct)
        
        return word_item
    
    def generate_thematic_session(self, level: str = 'beginner', theme: str = None) -> List[Dict[str, Any]]:
        """Generate a thematic vocabulary learning session."""
        # Get words from the specified level
        words = get_vocab_by_level(level)
        
        # If a theme is specified, filter words that might relate to it
        # This is a simple implementation - a more advanced version could use word embeddings
        if theme:
            theme_lower = theme.lower()
            theme_words = [w for w in words if (
                theme_lower in w['word'].lower() or
                theme_lower in w['definition'].lower() or
                theme_lower in w['example'].lower()
            )]
            
            # If we found enough theme-related words, use them
            if len(theme_words) >= 5:
                words = theme_words
        
        # Select random words for the session
        selected_words = random.sample(words, min(5, len(words)))
        
        # Create session items
        session_items = []
        for word_data in selected_words:
            target_language = random.choice(list(word_data['translations'].keys()))
            
            session_items.append({
                'word': word_data['word'],
                'definition': word_data['definition'],
                'example': word_data['example'],
                'synonym': word_data['synonym'],
                'target_language': target_language,
                'translation': word_data['translations'][target_language]
            })
        
        return session_items
    
    def generate_sentence_practice(self, level: str = 'beginner') -> Dict[str, Any]:
        """Generate a sentence translation practice challenge."""
        # Get a translation challenge for the specified level
        challenge = get_translation_challenge(level)
        
        if not challenge:
            return None
        
        # Select a random source and target language
        all_langs = list(challenge.keys())
        source_lang = random.choice(all_langs)
        target_langs = [lang for lang in all_langs if lang != source_lang]
        target_lang = random.choice(target_langs)
        
        return {
            'source_language': source_lang,
            'target_language': target_lang,
            'source_text': challenge[source_lang],
            'correct_translation': challenge[target_lang]
        }

# Global instance of the learning manager
learning_manager = LearningManager()