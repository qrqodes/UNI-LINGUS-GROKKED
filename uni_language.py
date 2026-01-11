"""
UNI Language Implementation
A simplified international constructed language with phonetic spelling and regular grammar.
Updated with comprehensive grammar rules from user specifications.
"""

class UNILanguage:
    """Implementation of the UNI artificial language."""
    
    def __init__(self):
        # Updated alphabet based on new specifications
        self.alphabet = "ABDEFGHIJKLMNOPRSTUVXZ"  # 21 letters
        self.whisper_alphabet = "AEFIKLMNOPRSTUX"   # 15 letters for whisper mode
        self.vowels = "AOUEI"
        self.consonants = "BDFGHIJKLMNPRSTUVXZ"
        
        # Voiced/Voiceless consonant pairs
        self.voiced_voiceless = {
            'B': 'P', 'D': 'T', 'G': 'K', 
            'J': 'X', 'V': 'F', 'Z': 'S'
        }
        
        # Core vocabulary based on updated UNI specification
        self.vocabulary = {
            # Personal pronouns
            'ma': 'I',
            'na': 'you',
            'ta': 'he/she/it/this/that',
            'mas': 'we',
            'nas': 'you (plural)',
            'tas': 'they',
            
            # Possessive pronouns
            'mau': 'my',
            'nau': 'your',
            'tau': 'his/her/its',
            'maus': 'our',
            'naus': 'your (plural)',
            'taus': 'their',
            
            # Articles (optional)
            'la': 'the',
            'una': 'a/an',
            
            # Core verbs (infinitive -AR form)
            'amar': 'to love',
            'verar': 'to see',
            'komar': 'to eat',
            'kosinar': 'to cook',
            'felisar': 'to rejoice',
            'belar': 'to beautify',
            'sar': 'to be',
            'liberar': 'to liberate',
            'fasilitar': 'to facilitate',
            'pronunsiar': 'to pronounce',
            'dar': 'to give',
            'tomar': 'to take',
            'ir': 'to go',
            'venir': 'to come',
            'saber': 'to know',
            'pensar': 'to think',
            'desear': 'to want',
            'nesesitar': 'to need',
            'aprendar': 'to learn',
            'parlar': 'to speak',
            'komprener': 'to understand',
            'ayudar': 'to help',
            'kantar': 'to sing',
            
            # Nouns (end in -A singular, -AS plural)
            'umana': 'human',
            'umanas': 'humans',
            'planeta': 'planet',
            'planetas': 'planets',
            'universa': 'universe',
            'kata': 'cat',
            'katas': 'cats',
            'peska': 'fish',
            'peskas': 'fish (plural)',
            'libra': 'book',
            'libras': 'books',
            'kaza': 'house', 
            'kazas': 'houses',
            'sopa': 'soup',
            'sopas': 'soups',
            'kosina': 'kitchen/cuisine',
            'kosinas': 'kitchens/cuisines',
            'felisa': 'happiness',
            'bela': 'beauty',
            'doma': 'house',
            'mama': 'mother',
            'komara': 'food',
            'felisara': 'something that makes you happy',
            'belara': 'beauty product',
            'beladora': 'beautician',
            'vera': 'sight/vision',
            'amora': 'love',
            
            # Adjectives (end in -U, invariable)
            'bonu': 'good',
            'nobonu': 'bad', 
            'felisu': 'happy',
            'belu': 'beautiful',
            'rapidu': 'quick',
            'importantu': 'important',
            'grandu': 'big',
            'pekenu': 'small',
            'novu': 'new',
            'antiku': 'old',
            'veru': 'visible',
            'amaru': 'lovely',
            
            # Adverbs (end in -URU)
            'bonuru': 'well',
            'nobonuru': 'badly',
            'felisuru': 'happily',
            'belaru': 'beautifully',
            'rapiduru': 'quickly',
            
            # Prepositions (comprehensive set)
            'o': 'or',
            'i': 'and',
            'sou': 'so',
            'komu': 'as/how',
            'poru': 'by',
            'paru': 'for',
            'peru': 'but',
            'du': 'of',
            'emboru': 'although',
            'se': 'if',
            'nose': 'unless',
            'amenosse': 'unless',
            'inu': 'at/in/on',
            'noinu': 'not at/out/off',
            'foru': 'out/off',
            'a': 'to/for',
            'noa': 'from/away',
            'da': 'from',
            'konu': 'with',
            'nokonu': 'without',
            'senu': 'without',
            'lonu': 'left',
            'nolonu': 'right',
            'baxu': 'down',
            'nobaxu': 'up',
            'simu': 'up',
            'subu': 'under/sub',
            'dubaxu': 'under',
            'nosubu': 'over',
            'dunobaxu': 'over',
            'superu': 'over/super',
            'duzdu': 'since',
            'dunouzdu': 'until',
            'atu': 'until',
            'sobru': 'about/around',
            'redoru': 'around',
            'anusentidaorariu': 'clockwise',
            'noinusentidaorariu': 'counterclockwise',
            'antu': 'before',
            'noantu': 'after',
            'depu': 'after',
            'kon': 'with/by',
            
            # Question words
            'ka': 'question marker (optional)',
            'keu': 'what',
            'ki': 'who',
            'kualu': 'which',
            'kuandu': 'when',
            'kendu': 'where',
            'kau': 'why',
            'komu': 'how',
            
            # Numbers (0-20, then key numbers)
            'nula': '0',
            'una': '1',
            'dua': '2',
            'tra': '3',
            'kuatra': '4',
            'sinka': '5',
            'sesa': '6',
            'seta': '7',
            'okta': '8',
            'nova': '9',
            'deka': '10',
            'dekauna': '11',
            'dekadua': '12',
            'dekatra': '13',
            'dekakuatra': '14',
            'dekasinka': '15',
            'dekasesa': '16',
            'dekaseta': '17',
            'dekaokta': '18',
            'dekanova': '19',
            'duadeka': '20',
            'tradeka': '30',
            'kuatradeka': '40',
            'sinkadeka': '50',
            'sesadeka': '60',
            'setadeka': '70',
            'oktadeka': '80',
            'novadeka': '90',
            'senta': '100',
            'mila': '1000',
            'miliona': '1000000',
            
            # Ordinal numbers (add -U)
            'unau': 'first',
            'duau': 'second',
            'trau': 'third',
            'kuatrau': 'fourth',
            'sinkau': 'fifth',
            'dekau': 'tenth',
            'duadekau': 'twentieth',
            'sentau': 'hundredth',
            
            # Fractions (add -RA)
            'duara': 'half',
            'trara': 'third',
            'kuatrara': 'quarter',
            
            # Basic vocabulary expansion
            'saluta': 'hello',
            'despidar': 'goodbye',
            'si': 'yes',
            'no': 'no',
            'porufavoru': 'please',
            'grasias': 'thank you',
            'perketu': 'excuse me',
            'deskulpa': 'sorry',
            'agua': 'water',
            'foku': 'fire',
            'tera': 'earth',
            'ventu': 'wind',
            'solu': 'sun',
            'luna': 'moon',
            'stella': 'star',
            'stellas': 'stars',
            'dia': 'day',
            'noktu': 'night',
            'tempo': 'time',
            'lugar': 'place',
            'hombre': 'man',
            'dona': 'woman',
            'nixu': 'child',
            'parent': 'parent',
            'familia': 'family',
            'amiku': 'friend',
            'amor': 'love',
            'vita': 'life',
            'mortu': 'death',
            'saluda': 'health',
            'enfermedad': 'illness',
            'medicina': 'medicine',
            'trabajo': 'work',
            'eskuela': 'school',
            'universidad': 'university',
            'dineru': 'money',
            'kolor': 'color',
            'blanka': 'white',
            'negra': 'black',
            'roja': 'red',
            'azula': 'blue',
            'verda': 'green',
            'amarela': 'yellow',
            'oranxa': 'orange',
            'violeta': 'purple',
            'roza': 'pink',
            'griz': 'gray',
            'marron': 'brown',
            
            # Prepositions (updated comprehensive list)
            'i': 'and',
            'o': 'or', 
            'sou': 'so',
            'komu': 'as/how',
            'poru': 'by',
            'paru': 'for',
            'peru': 'but',
            'du': 'of',
            'emboru': 'although',
            'se': 'if',
            'nose': 'unless',
            'a menos se': 'unless',
            'inu': 'in/at/on',
            'noinu': 'out/off/away',
            'foru': 'out/off',
            'a': 'to',
            'noa': 'to (away)',
            'da': 'from',
            'konu': 'with',
            'nokonu': 'without',
            'senu': 'without',
            'lonu': 'left',
            'nolonu': 'right',
            'baxu': 'down',
            'nobaxu': 'up',
            'simu': 'up',
            'subu': 'under',
            'dubaxu': 'under',
            'nosubu': 'over',
            'dunobaxu': 'over', 
            'superu': 'over',
            'duzdu': 'since',
            'dunouzdu': 'until',
            'atu': 'until',
            'sobru': 'about',
            'redoru': 'around',
            'inu sentida orariu': 'clockwise',
            'noinu sentida orariu': 'counterclockwise',
            'antu': 'before',
            'noantu': 'after',
            'depu': 'after',
            
            # Question words and markers
            'ka': 'question marker',
            'keu': 'what',
            'ki': 'who',
            'kualu': 'which',
            'kuandu': 'when',
            'kendu': 'where',
            'kau': 'why',
            'komu': 'how',
            
            # Numbers (comprehensive system)
            'nula': '0',
            'una': '1',
            'dua': '2', 
            'tra': '3',
            'kuatra': '4',
            'sinka': '5',
            'sesa': '6',
            'seta': '7',
            'okta': '8',
            'nova': '9',
            'deka': '10',
            'dekauna': '11',
            'dekadua': '12',
            'dekatra': '13',
            'dekakuatra': '14',
            'dekasinka': '15',
            'dekasesa': '16',
            'dekaseta': '17',
            'dekaokta': '18',
            'dekanova': '19',
            'duadeka': '20',
            'tradeka': '30',
            'kuatradeka': '40',
            'sinkadeka': '50',
            'sesadeka': '60',
            'setadeka': '70',
            'oktadeka': '80',
            'novadeka': '90',
            'senta': '100',
            'mila': '1000',
            'miliona': 'million',
            
            # Ordinal numbers (add -U)
            'unau': 'first',
            'duau': 'second',
            'trau': 'third',
            'dekaunau': 'eleventh',
            'duadekau': 'twentieth',
            
            # Fractions (add -RA)
            'duara': 'half',
            'trara': 'third',
            'kuatrara': 'quarter',
            
            # Time and location words
            'no': 'not/no',
            'si': 'yes',
            'ali': 'here',
            'aki': 'there',
            'dia': 'day',
            'noxe': 'night',
            'oy': 'today',
            'manyana': 'tomorrow',
            'ayer': 'yesterday',
            'tempu': 'time',
            
            # Common words and phrases
            'luma': 'hello',
            'adeu': 'goodbye',
            'plez': 'please',
            'grasias': 'thank you',
            'agua': 'water',
            'komida': 'food',
            'vida': 'life',
            'pasu': 'peace',
            'amiku': 'friend',
            'familia': 'family',
            'trabayu': 'work',
            
            # Sample phrases
            'ma amar na': 'I love you',
            'ta amar mas': 'he/she loves us',
            'kata komar peska': 'the cat eats the fish',
            'maus kaza naus kaza': 'our house is your house',
            'ma verar na': 'I see you',
            'felisu umana': 'happy human',
            'belu libra': 'beautiful book',
            'na kosinar sopa': 'you cook soup',
            'ta sar importantu': 'it is important',
            'ka na amar ma': 'do you love me?',
            'grasias paru ayuda': 'thank you for help',
        }
        
        # Reverse dictionary for English to UNI
        self.english_to_uni = {v: k for k, v in self.vocabulary.items()}
        
        # Add some additional mappings for common English words
        self.english_to_uni.update({
            'hello': 'luma',  # greeting
            'goodbye': 'adeu',  # farewell
            'yes': 'si',
            'please': 'plez',
            'thank you': 'grasias',
            'water': 'agua',
            'food': 'komida',
            'time': 'tempu',
            'life': 'vida',
            'peace': 'pasu',
            'friend': 'amiku',
            'family': 'familia',
            'work': 'trabayu',
            'learn': 'aprendar',
            'speak': 'parlar',
            'understand': 'komprener',
            'help': 'ayudar',
            'go': 'ir',
            'come': 'venir',
            'give': 'dar',
            'take': 'tomar',
            'know': 'saber',
            'think': 'pensar',
            'want': 'desear',
            'need': 'nesesitar',
            'big': 'grandu',
            'small': 'pekenu',
            'new': 'novu',
            'old': 'antiku',
            'day': 'dia',
            'night': 'noxe',
            'today': 'oy',
            'tomorrow': 'manyana',
            'yesterday': 'ayer'
        })
        
        # Update vocabulary with new additions
        for eng, uni in self.english_to_uni.items():
            if uni not in self.vocabulary:
                self.vocabulary[uni] = eng
    
    def convert_to_whisper_mode(self, text):
        """Convert UNI text to whisper mode (voiceless consonants)."""
        # Replace voiced consonants with voiceless ones
        whisper_map = {
            'B': 'P', 'D': 'T', 'G': 'K', 'J': 'X', 'V': 'F', 'Z': 'S'
        }
        
        result = text.upper()
        for voiced, voiceless in whisper_map.items():
            result = result.replace(voiced, voiceless)
        
        return result
    
    def apply_verb_tense(self, verb_root, tense):
        """Apply tense to UNI verbs based on updated grammar."""
        # Remove -AR ending if present
        if verb_root.endswith('AR'):
            root = verb_root[:-2]
        else:
            root = verb_root
        
        tense_endings = {
            'present': 'AR',        # AMAR (love/to love)
            'infinitive': 'AR',     # AMAR (to love)
            'past': 'ARÒ',          # AMARÒ (loved) - with accent
            'future': 'ARÈ',        # AMARÈ (will love) - with accent  
            'conditional': 'ARÈBE', # AMARÈBE (would love)
            'continuous': 'ANDU',   # AMANDU (loving/being in love)
            'imperative': 'ARI',    # AMARI (love!)
            'active_adj': 'ADU',    # AMADU (loving as adjective)
            'passive_adj': 'UDU',   # AMUDU (loved as adjective)
            'active_cont': 'ANDU',  # AMANDU (loving continuously)
            'passive_cont': 'UNDU', # AMUNDU (being loved)
            'gerund': 'ANDU'        # AMANDU (loving)
        }
        
        return root + tense_endings.get(tense, 'AR')
    
    def make_plural(self, noun):
        """Convert UNI noun to plural form."""
        if noun.endswith('A'):
            return noun + 'S'
        return noun + 'AS'
    
    def translate_to_english(self, uni_text):
        """Translate UNI text to English."""
        words = uni_text.upper().split()
        translated_words = []
        
        i = 0
        while i < len(words):
            # Try to match multi-word phrases first
            matched = False
            for phrase_len in range(min(4, len(words) - i), 0, -1):
                phrase = ' '.join(words[i:i+phrase_len]).lower()
                if phrase in self.vocabulary:
                    translated_words.append(self.vocabulary[phrase])
                    i += phrase_len
                    matched = True
                    break
            
            if not matched:
                word = words[i].lower()
                if word in self.vocabulary:
                    translated_words.append(self.vocabulary[word])
                else:
                    # Try to handle verb tenses
                    base_word = self.handle_verb_tenses(word)
                    if base_word and base_word in self.vocabulary:
                        translated_words.append(self.vocabulary[base_word])
                    else:
                        translated_words.append(f"[{word}]")  # Unknown word
                i += 1
        
        return ' '.join(translated_words)
    
    def handle_verb_tenses(self, word):
        """Handle UNI verb tense recognition with updated grammar."""
        # Check for different tense endings (updated patterns)
        tense_patterns = [
            ('ANDU', 'AR'),    # continuous/gerund -> infinitive
            ('UNDU', 'AR'),    # passive continuous -> infinitive
            ('ARÒ', 'AR'),     # past -> infinitive
            ('ARÈ', 'AR'),     # future -> infinitive
            ('ARÈBE', 'AR'),   # conditional -> infinitive
            ('ARI', 'AR'),     # imperative -> infinitive
            ('ADU', 'AR'),     # active adjective -> infinitive
            ('UDU', 'AR'),     # passive adjective -> infinitive
            ('ÒRDU', 'AR'),    # complex past participle -> infinitive
        ]
        
        for ending, replacement in tense_patterns:
            if word.endswith(ending):
                base = word[:-len(ending)] + replacement
                return base.lower()
        
        return None
    
    def translate_from_english(self, english_text):
        """Translate English text to UNI."""
        # Simple word-by-word translation
        words = english_text.lower().split()
        translated_words = []
        
        i = 0
        while i < len(words):
            # Try to match multi-word phrases first
            matched = False
            for phrase_len in range(min(4, len(words) - i), 0, -1):
                phrase = ' '.join(words[i:i+phrase_len])
                if phrase in self.english_to_uni:
                    translated_words.append(self.english_to_uni[phrase].upper())
                    i += phrase_len
                    matched = True
                    break
            
            if not matched:
                word = words[i].strip('.,!?;:')
                if word in self.english_to_uni:
                    translated_words.append(self.english_to_uni[word].upper())
                else:
                    # Try to create UNI-like word
                    uni_word = self.create_uni_word(word)
                    translated_words.append(uni_word)
                i += 1
        
        return ' '.join(translated_words)
    
    def create_uni_word(self, english_word):
        """Create a UNI-like word from English using phonetic rules."""
        # Basic phonetic mapping to UNI alphabet
        word = english_word.upper()
        
        # Replace letters not in UNI alphabet
        replacements = {
            'C': 'K',  # hard C
            'Q': 'K',
            'W': 'V',
            'Y': 'I',
            'PH': 'F',
            'TH': 'T',
            'SH': 'X',
            'CH': 'TX',
        }
        
        for old, new in replacements.items():
            word = word.replace(old, new)
        
        # Remove letters not in UNI alphabet
        uni_word = ''.join(c for c in word if c in self.alphabet)
        
        # Add appropriate ending based on word type
        if english_word.endswith('ing'):
            return uni_word + 'ANDU'
        elif english_word.endswith('ed'):
            return uni_word + 'ADU'
        elif len(uni_word) > 0:
            # Default to noun form
            if not uni_word.endswith('A'):
                return uni_word + 'A'
        
        return uni_word or f"[{english_word}]"
    
    def get_sample_phrases(self):
        """Get sample UNI phrases with translations."""
        return [
            ("LUMA, MA AMAR NA", "Hello, I love you"),
            ("MAUS KAZA NAUS KAZA", "Our house is your house"),
            ("KATA KOMAR PESKA", "The cat eats fish"),
            ("MA VERAR NA", "I see you"),
            ("FELISU UMANA", "Happy human"),
            ("BELU LIBRA", "Beautiful book"),
            ("NA KOSINAR SOPA", "You cook soup"),
            ("TA SAR IMPORTANTU", "It is important"),
            ("KA NA AMAR MA", "Do you love me?"),
            ("GRASIAS PARU AYUDA", "Thank you for help")
        ]
    
    def get_grammar_info(self):
        """Get UNI grammar information."""
        return {
            'alphabet': self.alphabet,
            'vowels': self.vowels,
            'word_order': 'SVO (Subject-Verb-Object)',
            'articles': 'LA (the), UNA (a/an) - optional',
            'verb_tenses': {
                'present': '-AR',
                'past': '-AVO', 
                'future': '-ARE',
                'continuous': '-ANDU'
            },
            'pronouns': {
                'personal': 'MA/NA/TA (I/you/he-she-it)',
                'possessive': 'MAU/NAU/TAU (my/your/his-her-its)'
            },
            'plurals': 'Add -S to nouns ending in -A, -AS to others',
            'adjectives': 'End in -U, invariable',
            'questions': 'KA + statement, or question words (KEU, KI, etc.)'
        }

# Global instance for easy access
uni_language = UNILanguage()