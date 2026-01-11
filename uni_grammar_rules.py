"""
Enhanced UNI Grammar Rules and Vocabulary Database
Complete implementation of UNI language rules for AI translation systems
"""

import logging
from typing import Dict, List, Any
import json

logger = logging.getLogger(__name__)

class UNIGrammarDatabase:
    """Complete UNI grammar system with rules, vocabulary, and examples"""
    
    def __init__(self):
        self.initialize_grammar_system()
    
    def initialize_grammar_system(self):
        """Initialize complete UNI grammar rules and vocabulary"""
        
        # Core alphabet and pronunciation
        self.alphabet = {
            'full': ['A', 'B', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'X', 'Z'],
            'whisper': ['A', 'E', 'F', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'X']
        }
        
        # Pronunciation rules
        self.pronunciation = {
            'vowels': {'A': 'ah (father)', 'E': 'eh (met)', 'I': 'ee (machine)', 'O': 'oh (for)', 'U': 'oo (rule)'},
            'voiced_consonants': {'B': 'b (Barbara)', 'D': 'd (down)', 'G': 'g (green)', 'J': 'zh (beige)', 'V': 'v (very)', 'Z': 'z (zero)'},
            'voiceless_consonants': {'P': 'p (Peter)', 'T': 't (top)', 'K': 'k (kind)', 'X': 'sh (sheep)', 'F': 'f (fine)', 'S': 's (smart)'},
            'continuous': {'L': 'l (later)', 'M': 'm (meter)', 'N': 'n (normal)', 'R': 'r (robot)'}
        }
        
        # Whisper conversion rules
        self.whisper_conversion = {'B': 'P', 'D': 'T', 'G': 'K', 'V': 'F', 'J': 'X', 'Z': 'S'}
        
        # Personal pronouns
        self.pronouns = {
            'personal': {'I': 'MA', 'you': 'NA', 'he/she/it': 'TA', 'we': 'MAS', 'you_plural': 'NAS', 'they': 'TAS'},
            'possessive': {'my': 'MAU', 'your': 'NAU', 'his/her/its': 'TAU', 'our': 'MAUS', 'your_plural': 'NAUS', 'their': 'TAUS'},
            'demonstrative': {'this/that': 'TA'}
        }
        
        # Numbers system
        self.numbers = {
            'cardinal': {
                0: 'NULA', 1: 'UNA', 2: 'DUA', 3: 'TRA', 4: 'KUATRA', 5: 'SINKA',
                6: 'SESA', 7: 'SETA', 8: 'OKTA', 9: 'NOVA', 10: 'DEKA',
                20: 'DUA-DEKA', 30: 'TRA-DEKA', 100: 'SENTA', 1000: 'MILA'
            },
            'ordinal_suffix': 'U',  # UNAU (first), DUAU (second)
            'fraction_suffix': 'RA'  # DUARA (half), TRARA (third)
        }
        
        # Verb conjugation system
        self.verb_tenses = {
            'past': 'ARÒ',
            'present': 'AR',
            'future': 'ARÈ',
            'conditional': 'ARÈBE',
            'continuous': 'ANDU',
            'imperative': 'ARI'
        }
        
        # Word formation rules
        self.word_formation = {
            'noun_singular': 'A',
            'noun_plural': 'AS',
            'verb_infinitive': 'AR',
            'adjective_neutral': 'U',
            'adjective_active': 'ADU',
            'adjective_passive': 'UDU',
            'adverb': 'URU',
            'participle_active': 'ANDU',
            'participle_passive': 'UNDU'
        }
        
        # Prepositions
        self.prepositions = {
            'or': 'O', 'and': 'I', 'of': 'DU', 'so': 'SOU', 'as': 'KOMU',
            'for': 'PARU', 'but': 'PERU', 'by': 'PORU', 'although': 'EMBORU',
            'if': 'SE', 'unless': 'NOSE', 'at': 'ANU', 'not_at': 'NOANU',
            'in': 'INU', 'out': 'NOINU', 'to': 'A', 'from': 'DA',
            'with': 'KONU', 'without': 'NOKONU', 'left': 'LONU', 'right': 'NOLONU',
            'down': 'BAXU', 'up': 'SIMU', 'under': 'SUBU', 'over': 'SUPERU',
            'since': 'DUZDU', 'until': 'ATU', 'about': 'SOBRU', 'around': 'REDORU',
            'before': 'ANTU', 'after': 'DEPU'
        }
        
        # Question words
        self.question_words = {
            'what': 'KEU', 'who': 'KI', 'which': 'KEUALU', 'when': 'KUANDU',
            'where': 'KENDU', 'why': 'KAU', 'how': 'KOMU'
        }
        
        # Essential vocabulary
        self.vocabulary = {
            'greetings': {'hello': 'SALUTA', 'goodbye': 'ADIA', 'thank_you': 'GRASA', 'yes': 'SI', 'no': 'NO'},
            'family': {'mother': 'MAMA', 'father': 'PAPA', 'child': 'INFANTA', 'brother': 'FRATA', 'sister': 'SORORA', 'family': 'FAMILA'},
            'food': {'food': 'KOMARA', 'water': 'AKUA', 'bread': 'PANA', 'fruit': 'FRUTA', 'meat': 'KARNA'},
            'basic_verbs': {'eat': 'KOMAR', 'drink': 'BEBAR', 'see': 'VERAR', 'speak': 'FALAR', 'go': 'ANDAR'},
            'common_nouns': {'house': 'KASA', 'book': 'LIBRA', 'friend': 'AMIGA', 'world': 'MUNDU', 'love': 'AMORA'},
            'adjectives': {'big': 'GRANDU', 'small': 'PIKU', 'happy': 'FELISU', 'good': 'BONU', 'bad': 'NO BONU', 'beautiful': 'BELU'},
            'time': {'today': 'OSJU', 'tomorrow': 'MANIANA', 'yesterday': 'JERU', 'now': 'AGORU', 'never': 'NUNKA', 'always': 'SEMPRU'}
        }
        
        # Core verbs with full conjugations
        self.core_verbs = {
            'be': {'infinitive': 'SAR', 'past': 'SARÒ', 'present': 'SAR', 'future': 'SARÈ'},
            'have': {'infinitive': 'TAR', 'past': 'TARÒ', 'present': 'TAR', 'future': 'TARÈ'},
            'do': {'infinitive': 'FAR', 'past': 'FARÒ', 'present': 'FAR', 'future': 'FARÈ'},
            'say': {'infinitive': 'DIZAR', 'past': 'DIZARÒ', 'present': 'DIZAR', 'future': 'DIZARÈ'},
            'go': {'infinitive': 'ANDAR', 'past': 'ANDARÒ', 'present': 'ANDAR', 'future': 'ANDARÈ'},
            'love': {'infinitive': 'AMAR', 'past': 'AMARÒ', 'present': 'AMAR', 'future': 'AMARÈ'}
        }
        
        # Grammar rules
        self.grammar_rules = {
            'word_order': 'SVO (Subject-Verb-Object) - flexible for emphasis',
            'articles': 'Optional: LA (the), UNA (a/an)',
            'verb_be': 'Optional SAR - only use if necessary',
            'negation': 'NO before verb: MA NO VERAR NA (I don\'t see you)',
            'questions': 'Optional KA at beginning/end, or intonation',
            'reflexive': 'SA for self: MA AMAR SA (I love myself)',
            'reciprocal': 'UNA OTRA: TAS AMAR UNA OTRA (They love each other)',
            'comparison': 'PLU (more), PLUS (most): BONU, PLU BONU, PLUS BONU'
        }
        
        # Example sentences
        self.examples = {
            'basic': [
                {'english': 'Hello world', 'uni': 'SALUTA MUNDU'},
                {'english': 'I love you', 'uni': 'MA AMAR NA'},
                {'english': 'How are you?', 'uni': 'KOMU NA?'},
                {'english': 'Thank you very much', 'uni': 'GRASA MUKU'},
                {'english': 'The cat eats fish', 'uni': 'KATA KOMAR PESKA'}
            ],
            'intermediate': [
                {'english': 'We are happy', 'uni': 'MAS SAR FELISU'},
                {'english': 'She speaks well', 'uni': 'TA FALAR BONURU'},
                {'english': 'They love each other', 'uni': 'TAS AMAR UNA OTRA'},
                {'english': 'My house is beautiful', 'uni': 'MAU KASA SAR BELU'}
            ],
            'advanced': [
                {'english': 'Tomorrow I will go to school', 'uni': 'MANIANA MA ANDARÈ A SKOLA'},
                {'english': 'If you want, we can eat together', 'uni': 'SE NA KERAR, MAS PODAR KOMAR JUNTU'},
                {'english': 'The book that I read yesterday was very interesting', 'uni': 'LIBRA KE MA LARARÒ JERU SARÒ MUKU INTERESANTU'}
            ]
        }
    
    def get_complete_grammar_rules(self) -> Dict[str, Any]:
        """Get complete UNI grammar system for AI training"""
        return {
            'alphabet': self.alphabet,
            'pronunciation': self.pronunciation,
            'whisper_conversion': self.whisper_conversion,
            'pronouns': self.pronouns,
            'numbers': self.numbers,
            'verb_tenses': self.verb_tenses,
            'word_formation': self.word_formation,
            'prepositions': self.prepositions,
            'question_words': self.question_words,
            'vocabulary': self.vocabulary,
            'core_verbs': self.core_verbs,
            'grammar_rules': self.grammar_rules,
            'examples': self.examples
        }
    
    def translate_word_to_uni(self, word: str, word_type: str = 'noun') -> str:
        """Translate individual words to UNI using grammar rules"""
        word_lower = word.lower()
        
        # Check vocabulary first
        for category in self.vocabulary.values():
            if isinstance(category, dict):
                for eng, uni in category.items():
                    if eng == word_lower:
                        return uni
        
        # Check core verbs
        for verb_data in self.core_verbs.values():
            if word_lower in verb_data.get('english_forms', []):
                return verb_data['infinitive']
        
        # Apply UNI word formation rules
        if word_type == 'noun':
            return self.create_uni_noun(word)
        elif word_type == 'verb':
            return self.create_uni_verb(word)
        elif word_type == 'adjective':
            return self.create_uni_adjective(word)
        
        return word.upper()  # Fallback to uppercase
    
    def create_uni_noun(self, word: str) -> str:
        """Create UNI noun following formation rules"""
        # Simplified noun creation - convert to UNI phonetics + A ending
        uni_root = self.convert_to_uni_phonetics(word)
        return uni_root + 'A' if not uni_root.endswith('A') else uni_root
    
    def create_uni_verb(self, word: str) -> str:
        """Create UNI verb following formation rules"""
        uni_root = self.convert_to_uni_phonetics(word)
        return uni_root + 'AR' if not uni_root.endswith('AR') else uni_root
    
    def create_uni_adjective(self, word: str) -> str:
        """Create UNI adjective following formation rules"""
        uni_root = self.convert_to_uni_phonetics(word)
        return uni_root + 'U' if not uni_root.endswith('U') else uni_root
    
    def convert_to_uni_phonetics(self, word: str) -> str:
        """Convert English word to UNI phonetic system"""
        # Simplified phonetic conversion
        word = word.upper()
        
        # Apply basic phonetic rules
        conversions = {
            'PH': 'F', 'TH': 'T', 'CH': 'TX', 'SH': 'X',
            'QU': 'KU', 'Y': 'I', 'W': 'V', 'C': 'K'
        }
        
        for eng, uni in conversions.items():
            word = word.replace(eng, uni)
        
        # Remove non-UNI letters
        allowed_letters = set(self.alphabet['full'])
        word = ''.join(c for c in word if c in allowed_letters)
        
        return word

# Global instance
uni_grammar_db = UNIGrammarDatabase()

def get_uni_grammar_system() -> Dict[str, Any]:
    """Get complete UNI grammar system"""
    return uni_grammar_db.get_complete_grammar_rules()

def translate_to_uni_enhanced(text: str, source_lang: str = "en") -> str:
    """Enhanced UNI translation using complete grammar rules"""
    try:
        words = text.split()
        uni_words = []
        
        for word in words:
            # Clean word
            clean_word = word.strip('.,!?;:"').lower()
            
            # Try vocabulary lookup first
            uni_word = uni_grammar_db.translate_word_to_uni(clean_word)
            uni_words.append(uni_word)
        
        return ' '.join(uni_words)
    
    except Exception as e:
        logger.error(f"UNI translation error: {e}")
        return text.upper()

if __name__ == "__main__":
    # Test the system
    test_phrases = [
        "Hello world",
        "I love you",
        "The cat is beautiful",
        "We are happy"
    ]
    
    print("Enhanced UNI Translation Test:")
    for phrase in test_phrases:
        uni_translation = translate_to_uni_enhanced(phrase)
        print(f"English: {phrase}")
        print(f"UNI: {uni_translation}")
        print("-" * 40)