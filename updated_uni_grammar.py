#!/usr/bin/env python3
"""
Updated UNI Grammar System - January 2026
Comprehensive implementation of the latest UNI language rules and vocabulary
Based on the official UNI LINGUS grammar specification
"""

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

UNI_GRAMMAR_TEXT = """
UNI LINGUS

UNI is a simple artificial language. It's much easier to learn and speak than any language you know. If you speak any Romanic language you will understand 70% of this dialogue:

A: OLA, KOMU VIDA? KE NOVU?
(Hello, how is life? What's new?)

B: OI, VIDA BONU, MA KOMENSARO ESTUDAR NOVU LINGUA ARTIFISIALU.
(Hey, life is good, I began to study a new artificial language)

A: KE SIGNIFIKAR? LINGUA KONSTRUKTUDU, KOMU ESPERANTA?
(What does that mean? A constructed language, like Esperanto?)

B: SIMILU, SI.
(Similar, yes.)

A: NO DIFISILU APRENDAR? NORMALU NESESITAR MUITU TEMPA INTENDAR GRAMATIKA, PRONUNSIA, MEMORIZAR FRAZAS NO KONOSUDU?
(Isn't it difficult to learn? You normally need a lot of time to understand grammar, pronunciation, to memorise unfamiliar phrases?)

B: REALU NO, VERDA KE NO FALARE RAPIDU MESMU DIA, PERU PODAR ESPLIKAR KOZAS BAZIKU.
(Not really, the truth is, you won't speak fluently the same day, but you can explain basic things.)

A: SOLU IN(U) UNA DIA? KOMU POSIBLU?
(Just in one day? How is this possible)

B: PARUKE MUITU PARESADU IDIOMAS ROMANU I KOMU NA JA FALAR ESPANIOLA I PORTUGEZA, SARE MUITU FASILU. NA INTENDARE META PALAVRAS INTUITIVU. PROBABILU 90% PERSENTA VOCABULARIA SAREBE FAMILIARU.
(Because it's very similar to Romance languages and since you already speak Spanish and Portuguese, It will be very easy. You'll understand half of the words intuitively. Probably 90% per cent of the vocabulary would be familiar.)

A: LOKU, PERU KE OTRU KOZAS SIMPLIFIKAR PROSESA? ESPANIOLA MAU IDIOMA NATIVU PERU PRONUNSIA PORTUGEZA SARO DIFISILU INU KOMENSA.
(Crazy, but what other things simplify the process? Spanish is my native language but Portuguese pronunciation was difficult in the beginning.)

B: REGLAS TOTALU MINIMU. NO GENERAS PARU SUSTANTIVAS, SOU NO PREOKUPAR SA KON MASKULINU O FEMININU. SUSTANTIVAS SIMPLU TERMINAR KON -A PARU SINGULARU I -AS PARU PLURALU. TIPU ''Book" SAREBE LIBRA I ''Books'' SAREBE LIBRAS.
(The rules are totally minimal. No genders for nouns, so don't worry about masculine or feminine. The nouns simply end with -A for singular and -AS for plural. Like "a book" would be LIBRA and "books" would be LIBRAS.)

A: I VERBAS?
(And verbs?)

B: KUASU MESMU FASILU. SOLU TRA TEMPAS PRINSIPALU: ADISIONAR -AR UZANDU PREZENTU, -RO INU PASADU I -RE KON FUTURU. NO VERBAS NOREGULARU.
(Almost as easy. Just three principal tenses: add -AR using present, -RO in the past and -RE with the future tense. No irregular verbs.)

A: WOW! NO KONJUGARA?
(Wow! No conjugations?)

B: EZATU! KOMU NA FALAREBE ''She loves to sing" ?
(Exactly! How would you say ''She loves to sing"?)

A: TA AMAR KANTAR.
(She loves to sing.)

B: KOREKTU! NA JA PRAKTIKU NATIVU!
(Correct! You are already practically a native)

A: DEFINITIVU FASILITAR KOMUNIKA. KERAR PRAKTIKAR PROKSIMA SEMANA! DIVERTANDU!
(It definitely facilitates communication. I want to practice next week! It's entertaining!)

B: VERAR NA MANIANA!
(I'll see you tomorrow!)
"""

class UNIGrammarSystem:
    """Updated UNI Grammar with comprehensive rules and vocabulary - January 2026"""
    
    def __init__(self):
        self.alphabet = self._init_alphabet()
        self.pronouns = self._init_pronouns()
        self.verbs = self._init_verbs()
        self.vocabulary = self._init_vocabulary()
        self.grammar_rules = self._init_grammar_rules()
        self.numbers = self._init_numbers()
        self.prepositions = self._init_prepositions()
        self.common_verbs_50 = self._init_50_common_verbs()
        
        logger.info("Updated UNI Grammar System initialized with complete vocabulary")
    
    def _init_alphabet(self) -> Dict[str, any]:
        """Initialize UNI alphabet system - 21 letters full, 15 whisper"""
        return {
            "full_alphabet": "A B D E F G I J K L M N O P R S T U V X Z",
            "whisper_alphabet": "A E F I K L M N O P R S T U X",
            "excluded_letters": ["C", "H", "Q", "W", "Y"],
            "vowels": ["A", "E", "I", "O", "U"],
            "vowel_sounds": {
                "A": "as in father, rather",
                "E": "as in met, pet",
                "I": "as in machine, marine",
                "O": "as in for, sport",
                "U": "as in rule, bull"
            },
            "consonants": {
                "voiced": ["B", "D", "G", "J", "V", "Z"],
                "voiceless": ["P", "T", "K", "X", "F", "S"],
                "continuous": ["L", "M", "N", "R"]
            },
            "whisper_replacements": {
                "B": "P", "D": "T", "G": "K", 
                "V": "F", "J": "X", "Z": "S"
            },
            "whisper_examples": {
                "KOZA BAZIKU": "KOSA BASIKU",
                "PORTUGESA": "PORTUKESA",
                "VIDA NOVU": "FITA NOFU"
            }
        }
    
    def _init_pronouns(self) -> Dict[str, Dict]:
        """Initialize UNI pronoun system"""
        return {
            "personal": {
                "I": "MA", "you": "NA", "he/she/it": "TA",
                "we": "MAS", "you_plural": "NAS", "they": "TAS"
            },
            "possessive": {
                "my": "MAU", "your": "NAU", "his/her/its": "TAU",
                "our": "MAUS", "your_plural": "NAUS", "their": "TAUS"
            },
            "reflexive": "SA",
            "reciprocal": "UNA OTRA",
            "note": "TA is borrowed from spoken Chinese where it's pronounced the same for men and women"
        }
    
    def _init_verbs(self) -> Dict[str, Dict]:
        """Initialize comprehensive UNI verb system with updated tenses"""
        return {
            "tenses": {
                "present": {"suffix": "-AR", "example": "MA LIBERAR (I liberate)"},
                "past": {"suffix": "-RO", "example": "MA LIBERARO (I liberated)"},
                "future": {"suffix": "-RE", "example": "MA LIBERARE (I will liberate)"},
                "conditional": {"suffix": "-REBE", "example": "MA LIBERAREBE (I would liberate)"},
                "continuous": {"suffix": "-ANDU", "example": "MA LIBERARANDU (I am liberating)"},
                "imperative": {"suffix": "-RI", "example": "LIBERARI! (Liberate!)"}
            },
            "word_forms": {
                "noun": {"suffix": "-A", "example": "AMORA (love)"},
                "verb": {"suffix": "-AR", "example": "AMORAR/AMAR (to love)"},
                "active_gerund": {"suffix": "-ANDU", "example": "AMORANDU/AMANDU (loving, lover)"},
                "passive_gerund": {"suffix": "-UNDU", "example": "AMORUNDU/AMUNDU (being loved)"},
                "adjective": {"suffix": "-U", "example": "AMORU (lovable, lovely)"},
                "active_adverb": {"suffix": "-ARU", "example": "AMORARU (lovingly)"},
                "passive_adverb": {"suffix": "-ADU/-UDU", "example": "AMADU/AMUDU (loved, beloved)"},
                "diminutive": {"suffix": "-ITA/-SITA", "example": "AMORITA/AMORSITA"}
            },
            "essential_verbs": {
                "be": "SAR", "have": "TENAR", "do": "FAR", "say": "DIZAR",
                "go": "ANDAR", "get": "OBTAR", "make": "FAZAR", "know": "SABAR",
                "think": "PENSAR", "take": "TOMAR", "see": "VERAR", "come": "VENAR",
                "want": "KERAR", "use": "UZAR", "find": "ENKONTRAR", "give": "DAR",
                "tell": "KONTAR", "work": "LABORAR", "call": "XAMAR", "try": "PROVAR",
                "ask": "PREGUNTAR", "need": "NESESITAR", "feel": "SENTIRAR", "become": "TORNAR",
                "leave": "DEXAR", "put": "KOLOKAR", "keep": "MANTENAR", "let": "DEXAR",
                "begin": "KOMENSAR", "write": "SKRIBAR", "read": "LAR", "eat": "KOMAR",
                "drink": "BEBAR", "listen": "ESKUTAR", "watch": "VERAR", "play": "JOGAR",
                "buy": "KOMPRAR", "sell": "VENDAR", "drive": "KONDUKTAR", "love": "AMAR",
                "help": "AJUDAR", "learn": "APRENDAR", "teach": "ENSENAR", "build": "KONSTRUKTAR",
                "send": "ENVIAR", "receive": "RESIBIR", "run": "KORAR", "swim": "NADAR",
                "sleep": "DORMAR", "wake": "DESPERTAR", "open": "ABRAR", "close": "SERAR"
            }
        }
    
    def _init_vocabulary(self) -> Dict[str, Dict]:
        """Initialize comprehensive UNI vocabulary"""
        return {
            "greetings": {
                "hello": "OLA", "hi": "OI", "goodbye": "ADIA", 
                "thank_you": "GRASA", "yes": "SI", "no": "NO", 
                "please": "POR FAVOR", "good": "BONU", "bad": "MALU"
            },
            "family": {
                "mother": "MAMA", "father": "PAPA", "child": "NINA/NINO",
                "brother": "FRATA", "sister": "SISTA", "family": "FAMILA",
                "friend": "AMIGA"
            },
            "food": {
                "food": "KOMIDA", "water": "AGUA", "bread": "PANA",
                "fruit": "FRUTA", "cake": "BOLA", "coffee": "KAFA",
                "apple": "MANSANA"
            },
            "places": {
                "house": "KAZA", "school": "SKOLA", "office": "OFISA",
                "party": "FESTA", "world": "MUNDA", "universe": "UNIVERSA"
            },
            "time": {
                "day": "DIA", "tomorrow": "MANIANA", "week": "SEMANA",
                "year": "ANU", "time": "TEMPA", "now": "AGORA"
            },
            "body": {
                "human": "UMANA", "hand": "MANU", "eye": "OXA"
            },
            "objects": {
                "book": "LIBRA", "letter": "LETRA", "gift": "REGALA",
                "message": "MESAJA", "key": "XAVA", "pen": "BOLIGRAFA",
                "car": "AUTA", "dress": "VESTIDA", "star": "ESTRELA",
                "flower": "FLORA", "movie": "FILMA", "music": "MUZIKA"
            },
            "adjectives": {
                "big": "GRANDU", "small": "PIKU", "happy": "FELIZU",
                "new": "NOVU", "quick": "RAPIDU", "good": "BONU",
                "bad": "MALU", "similar": "SIMILU", "difficult": "DIFISILU",
                "easy": "FASILU", "true": "VERDA", "sure": "SEGURU"
            }
        }
    
    def _init_grammar_rules(self) -> Dict[str, any]:
        """Initialize UNI grammar rules"""
        return {
            "nouns": {
                "singular": "-A (all nouns end in -A)",
                "plural": "-AS",
                "no_gender": True,
                "examples": ["LIBRA/LIBRAS (book/books)", "UMANA (human)", "PLANETAS (planets)"]
            },
            "adjectives": {
                "suffix": "-U",
                "position": "before noun",
                "examples": ["RAPIDU KATA (quick cat)", "NOVU LIBRA (new book)"]
            },
            "word_order": "SVO (Subject-Verb-Object) like English/Chinese",
            "comparison": {
                "comparative": "PLU + adjective",
                "superlative": "PLUS + adjective",
                "example": "BONU, PLU BONU, PLUS BONU (good, better, best)"
            },
            "negation": "NO before the word (NO MALU = not bad)",
            "reflexive": "-SA for self (TA AMAR SA = She loves herself)",
            "reciprocal": "UNA OTRA for each other (TAS AMAR UNA OTRA = They love each other)"
        }
    
    def _init_numbers(self) -> Dict[str, any]:
        """Initialize UNI number system"""
        return {
            "cardinal": {
                0: "NULA", 1: "UNA", 2: "DUA", 3: "TRA", 4: "KUATRA",
                5: "SINKA", 6: "SESA", 7: "SETA", 8: "OKTA", 9: "NOVA",
                10: "DEKA", 100: "SENTA", 1000: "MILA", 1000000: "MILIONA"
            },
            "compound_examples": {
                11: "DEKA-UNA", 20: "DUA-DEKA", 21: "DUA-DEKA-UNA",
                123: "SENTA-DUA-TRA", 2025: "DUA-MILA-SENTA-DUA-DEKA"
            },
            "ordinal": {
                "suffix": "-U",
                "examples": ["UNU (first)", "DUU (second)", "TRU (third)"]
            },
            "fractions": {
                "suffix": "-DA",
                "examples": ["DUADA (half)", "TRADA (third)", "KUATRADA (quarter)"]
            }
        }
    
    def _init_prepositions(self) -> Dict[str, str]:
        """Initialize UNI prepositions"""
        return {
            "or": "O", "and": "I", "of": "DU", "so": "SOU",
            "as": "KOMU", "for": "PARU", "but": "PERU", "by/through": "PORU",
            "although": "EMBORU", "if": "SE", "unless": "A MENUS SE",
            "at": "ATU", "in/on": "INU", "out/off": "NOINU/FORU",
            "to": "A", "from": "DA", "with": "KONU", "without": "NOKONU/SENU",
            "left": "LETU", "right": "RETU", "down": "BAXU", "up": "SIMU",
            "under": "SUBU", "over": "SUPERU", "since": "DEZDU", "until": "ATU",
            "about": "SOBRU", "around": "REDORU", "before": "ANTU", "after": "DEPU"
        }
    
    def _init_50_common_verbs(self) -> List[Dict]:
        """Initialize 50 most common verbs with multilingual comparison"""
        return [
            {"en": "to be", "uni": "SAR", "example": "MA ESTUDIANTA (I am a student)"},
            {"en": "to have", "uni": "TENAR", "example": "MA TENAR LIBRA (I have a book)"},
            {"en": "to do", "uni": "FAR", "example": "MA FAR LABORA (I do my work)"},
            {"en": "to say", "uni": "DIZAR", "example": "MA DIZAR PALAVRA (I say a word)"},
            {"en": "to go", "uni": "ANDAR", "example": "MA ANDAR SKOLA (I go to school)"},
            {"en": "to get", "uni": "OBTAR", "example": "MA OBTAR REGALA (I get a gift)"},
            {"en": "to make", "uni": "FAZAR", "example": "MA FAZAR BOLA (I make a cake)"},
            {"en": "to know", "uni": "SABAR", "example": "MA SABAR FAKTA (I know a fact)"},
            {"en": "to think", "uni": "PENSAR", "example": "MA PENSAR SOBRU VIDA (I think about life)"},
            {"en": "to take", "uni": "TOMAR", "example": "MA TOMAR LIBRA (I take a book)"},
            {"en": "to see", "uni": "VERAR", "example": "MA VERAR ESTRELA (I see a star)"},
            {"en": "to come", "uni": "VENAR", "example": "MA VENAR FESTA (I come to the party)"},
            {"en": "to want", "uni": "KERAR", "example": "MA KERAR KAFA (I want a coffee)"},
            {"en": "to use", "uni": "UZAR", "example": "MA UZAR BOLIGRAFA (I use a pen)"},
            {"en": "to find", "uni": "ENKONTRAR", "example": "MA ENKONTRAR XAVA (I find a key)"},
            {"en": "to give", "uni": "DAR", "example": "MA DAR FLORA (I give a flower)"},
            {"en": "to tell", "uni": "KONTAR", "example": "MA KONTAR ISTORIA (I tell a story)"},
            {"en": "to work", "uni": "LABORAR", "example": "MA LABORAR INU OFISA (I work in an office)"},
            {"en": "to call", "uni": "XAMAR", "example": "MA XAMAR AMIGA (I call my friend)"},
            {"en": "to try", "uni": "PROVAR", "example": "MA PROVAR KOMIDA NOVU (I try new food)"},
            {"en": "to ask", "uni": "PREGUNTAR", "example": "MA PREGUNTAR PREGUNTA (I ask a question)"},
            {"en": "to need", "uni": "NESESITAR", "example": "MA NESESITAR LIBRA (I need a book)"},
            {"en": "to feel", "uni": "SENTIRAR", "example": "MA SENTIRAR FELIZU (I feel happy)"},
            {"en": "to become", "uni": "TORNAR", "example": "MA TORNAR PROFESORA (I become a teacher)"},
            {"en": "to leave", "uni": "DEXAR", "example": "MA DEXAR KAZA (I leave the house)"},
            {"en": "to put", "uni": "KOLOKAR", "example": "MA KOLOKAR LIBRA INU MEZA (I put a book on the table)"},
            {"en": "to keep", "uni": "MANTENAR", "example": "MA MANTENAR PROMESA (I keep my promise)"},
            {"en": "to let", "uni": "DEXAR", "example": "MA DEXAR TA ANDAR (I let her go)"},
            {"en": "to type", "uni": "DIJITAR", "example": "TA KERAR MA DIJITAR (She wants me to type)"},
            {"en": "to begin", "uni": "KOMENSAR", "example": "MA KOMENSAR PROJEKTA NOVU (I begin a new project)"},
            {"en": "to write", "uni": "SKRIBAR", "example": "MA SKRIBAR LETRA (I write a letter)"},
            {"en": "to read", "uni": "LAR", "example": "TA LAR LIBRA (She reads a book)"},
            {"en": "to eat", "uni": "KOMAR", "example": "MAS KOMAR MANSANA (We eat an apple)"},
            {"en": "to drink", "uni": "BEBAR", "example": "TA BEBAR AGUA (He drinks water)"},
            {"en": "to listen", "uni": "ESKUTAR", "example": "TAS ESKUTAR MUZIKA (They listen to music)"},
            {"en": "to watch", "uni": "VERAR", "example": "MA VERAR FILMA (I watch a movie)"},
            {"en": "to play", "uni": "JOGAR", "example": "NINAS JOGAR FUTBOLA (The children play soccer)"},
            {"en": "to buy", "uni": "KOMPRAR", "example": "TA KOMPRAR VESTIDA (She buys a dress)"},
            {"en": "to sell", "uni": "VENDAR", "example": "TA VENDAR AUTA (He sells a car)"},
            {"en": "to drive", "uni": "KONDUKTAR", "example": "MA KONDUKTAR AUTA (I drive a car)"},
            {"en": "to love", "uni": "AMAR", "example": "TAS AMAR UNA OTRA (They love each other)"},
            {"en": "to help", "uni": "AJUDAR", "example": "TA AJUDAR AMIGA (She helps her friend)"},
            {"en": "to learn", "uni": "APRENDAR", "example": "MAS APRENDAR KOZAS NOVU (We learn new things)"},
            {"en": "to teach", "uni": "ENSENAR", "example": "TA ENSENAR INGLEZA (He teaches English)"},
            {"en": "to build", "uni": "KONSTRUKTAR", "example": "TAS KONSTRUKTAR KAZA (They build a house)"},
            {"en": "to send", "uni": "ENVIAR", "example": "MA ENVIAR MESAJA (I send a message)"},
            {"en": "to receive", "uni": "RESIBIR", "example": "TA RESIBIR REGALA (She receives a gift)"},
            {"en": "to run", "uni": "KORAR", "example": "NA KORAR RAPIDU (You run fast)"},
            {"en": "to swim", "uni": "NADAR", "example": "MAS NADAR INU MARA (We swim in the sea)"},
            {"en": "to sleep", "uni": "DORMAR", "example": "TA DORMAR BONU (He sleeps well)"}
        ]
    
    def get_grammar_text(self) -> str:
        """Return the full UNI grammar introduction text"""
        return UNI_GRAMMAR_TEXT
    
    def translate_to_uni(self, english_word: str) -> Optional[str]:
        """Translate an English word to UNI"""
        word_lower = english_word.lower().strip()
        
        for verb in self.common_verbs_50:
            if verb["en"].replace("to ", "") == word_lower:
                return verb["uni"]
        
        for category in self.vocabulary.values():
            if word_lower in category:
                return category[word_lower].upper()
        
        for verb_en, verb_uni in self.verbs["essential_verbs"].items():
            if verb_en == word_lower:
                return verb_uni
        
        return None
    
    def get_verb_conjugation(self, verb_base: str, tense: str = "present") -> str:
        """Conjugate a UNI verb in the specified tense"""
        tense_map = {
            "present": "AR",
            "past": "RO",
            "future": "RE",
            "conditional": "REBE",
            "continuous": "ANDU",
            "imperative": "RI"
        }
        
        suffix = tense_map.get(tense.lower(), "AR")
        
        if verb_base.endswith("AR"):
            base = verb_base[:-2]
        else:
            base = verb_base
        
        return f"{base}{suffix}"
    
    def whisper_mode(self, text: str) -> str:
        """Convert UNI text to whisper mode (voiceless consonants only)"""
        replacements = self.alphabet["whisper_replacements"]
        result = text.upper()
        for voiced, voiceless in replacements.items():
            result = result.replace(voiced, voiceless)
        return result


uni_grammar = UNIGrammarSystem()

def get_uni_grammar() -> UNIGrammarSystem:
    """Get the global UNI grammar instance"""
    return uni_grammar
