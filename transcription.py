"""
Transcription module for Enhanced Language Learning Bot.
Provides pinyin for Chinese and Latin transcription for Russian and other Cyrillic languages.
Also includes audio transcription capabilities.
"""

import logging
import os
from pypinyin import pinyin, Style
import re
import tempfile

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Cyrillic to Latin mapping (Russian, Ukrainian, Bulgarian, etc.)
CYRILLIC_TO_LATIN = {
    # Russian
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    # Ukrainian additions
    'є': 'ye', 'і': 'i', 'ї': 'yi', 'ґ': 'g',
    # Bulgarian additions
    'ђ': 'dj', 'љ': 'lj', 'њ': 'nj', 'ћ': 'ć', 'џ': 'dž',
    # Capital letters
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
    'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
    'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
    'Є': 'Ye', 'І': 'I', 'Ї': 'Yi', 'Ґ': 'G',
    'Ђ': 'Dj', 'Љ': 'Lj', 'Њ': 'Nj', 'Ћ': 'Ć', 'Џ': 'Dž'
}

# Arabic to Latin mapping
ARABIC_TO_LATIN = {
    'ا': 'a', 'ب': 'b', 'ت': 't', 'ث': 'th', 'ج': 'j', 'ح': 'h', 'خ': 'kh',
    'د': 'd', 'ذ': 'dh', 'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'sh', 'ص': 's',
    'ض': 'd', 'ط': 't', 'ظ': 'z', 'ع': "'", 'غ': 'gh', 'ف': 'f', 'ق': 'q',
    'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y',
    'ء': "'", 'ة': 'a', 'ى': 'a', 'أ': 'a', 'إ': 'i', 'آ': 'a', 'ؤ': "'",
    'ئ': "'", 'ّ': '', 'َ': 'a', 'ِ': 'i', 'ُ': 'u', 'ً': 'an', 'ٍ': 'in',
    'ٌ': 'un', 'ْ': ''
}

# Hindi/Devanagari to Latin mapping
DEVANAGARI_TO_LATIN = {
    'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī', 'उ': 'u', 'ऊ': 'ū', 'ए': 'e',
    'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऋ': 'ṛ', 'ॠ': 'ṝ', 'ऌ': 'ḷ', 'ॡ': 'ḹ',
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'ṅa', 'च': 'cha',
    'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'ña', 'ट': 'ṭa', 'ठ': 'ṭha',
    'ड': 'ḍa', 'ढ': 'ḍha', 'ण': 'ṇa', 'त': 'ta', 'थ': 'tha', 'द': 'da',
    'ध': 'dha', 'न': 'na', 'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha',
    'म': 'ma', 'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'śa',
    'ष': 'ṣa', 'स': 'sa', 'ह': 'ha', 'क्ष': 'kṣa', 'त्र': 'tra', 'ज्ञ': 'jña',
    'ा': 'ā', 'ि': 'i', 'ी': 'ī', 'ु': 'u', 'ू': 'ū', 'े': 'e', 'ै': 'ai',
    'ो': 'o', 'ौ': 'au', '्': '', 'ं': 'ṃ', 'ः': 'ḥ',
    'ँ': 'ṃ', '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# Korean (Hangul) to Latin mapping
def transliterate_korean(text):
    """Simple mapping for Korean to Latin. For a more accurate transliteration, 
    a specialized library would be needed."""
    # This is a simplified version. A complete version would need a specialized library
    return text  # Placeholder for now - add a library for proper Korean transliteration

# Japanese (Kana) to Latin mapping
def transliterate_japanese(text):
    """Simple mapping for Japanese to Latin. For a more accurate transliteration, 
    a specialized library would be needed."""
    # This is a simplified version. A complete version would need a specialized library
    return text  # Placeholder for now - add a library for proper Japanese transliteration

def get_pinyin(text):
    """
    Convert Chinese text to pinyin.
    
    Args:
        text (str): Chinese text
        
    Returns:
        str: Pinyin representation
    """
    try:
        # Get pinyin with tone marks
        result = pinyin(text, style=Style.TONE)
        # Flatten the list and join with spaces
        return ' '.join([item[0] for item in result])
    except Exception as e:
        logger.error(f"Error generating pinyin: {e}")
        return None

def transliterate_cyrillic(text):
    """
    Convert Cyrillic text to Latin alphabet.
    
    Args:
        text (str): Cyrillic text
        
    Returns:
        str: Latin representation
    """
    result = ""
    for char in text:
        result += CYRILLIC_TO_LATIN.get(char, char)
    return result

def transliterate_arabic(text):
    """
    Convert Arabic text to Latin alphabet.
    
    Args:
        text (str): Arabic text
        
    Returns:
        str: Latin representation
    """
    result = ""
    for char in text:
        result += ARABIC_TO_LATIN.get(char, char)
    return result

def transliterate_hindi(text):
    """
    Convert Hindi/Devanagari text to Latin alphabet.
    
    Args:
        text (str): Hindi text
        
    Returns:
        str: Latin representation
    """
    # This is a simplified approach - a proper transliteration would be more complex
    result = ""
    skip_next = False
    
    for i in range(len(text)):
        if skip_next:
            skip_next = False
            continue
            
        char = text[i]
        next_char = text[i+1] if i+1 < len(text) else None
        
        # Check for conjuncts (consonant + virama + consonant)
        if next_char and next_char == '्':
            if i+2 < len(text):
                # This is a simplified handling of conjuncts
                result += DEVANAGARI_TO_LATIN.get(char, char).replace('a', '')
                skip_next = True
            else:
                result += DEVANAGARI_TO_LATIN.get(char, char)
        else:
            result += DEVANAGARI_TO_LATIN.get(char, char)
            
    return result

# Thai to Latin mapping
THAI_TO_LATIN = {
    'ก': 'k', 'ข': 'kh', 'ฃ': 'kh', 'ค': 'kh', 'ฅ': 'kh', 'ฆ': 'kh', 'ง': 'ng',
    'จ': 'ch', 'ฉ': 'ch', 'ช': 'ch', 'ซ': 's', 'ฌ': 'ch', 'ญ': 'y', 'ฎ': 'd',
    'ฏ': 't', 'ฐ': 'th', 'ฑ': 'th', 'ฒ': 'th', 'ณ': 'n', 'ด': 'd', 'ต': 't',
    'ถ': 'th', 'ท': 'th', 'ธ': 'th', 'น': 'n', 'บ': 'b', 'ป': 'p', 'ผ': 'ph',
    'ฝ': 'f', 'พ': 'ph', 'ฟ': 'f', 'ภ': 'ph', 'ม': 'm', 'ย': 'y', 'ร': 'r',
    'ล': 'l', 'ว': 'w', 'ศ': 's', 'ษ': 's', 'ส': 's', 'ห': 'h', 'ฬ': 'l',
    'อ': '', 'ฮ': 'h',
    '่': '', '้': '', '๊': '', '๋': '', '็': '', '์': '', 'ำ': 'am', 'ิ': 'i',
    'ี': 'i', 'ึ': 'ue', 'ื': 'ue', 'ุ': 'u', 'ู': 'u', 'เ': 'e', 'แ': 'ae',
    'โ': 'o', 'ใ': 'ai', 'ไ': 'ai', '๋': '', '็': '', '์': '', 'ั': 'a',
    '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4', '๕': '5', '๖': '6',
    '๗': '7', '๘': '8', '๙': '9'
}

# Hebrew to Latin mapping
HEBREW_TO_LATIN = {
    'א': 'a', 'ב': 'b', 'ג': 'g', 'ד': 'd', 'ה': 'h', 'ו': 'v', 'ז': 'z',
    'ח': 'ch', 'ט': 't', 'י': 'y', 'כ': 'k', 'ל': 'l', 'מ': 'm', 'נ': 'n',
    'ס': 's', 'ע': '', 'פ': 'p', 'צ': 'ts', 'ק': 'k', 'ר': 'r', 'ש': 'sh',
    'ת': 't', 'ך': 'kh', 'ם': 'm', 'ן': 'n', 'ף': 'f', 'ץ': 'ts',
    '־': '-', '׳': "'", '״': '"'
}

# Greek to Latin mapping
GREEK_TO_LATIN = {
    'α': 'a', 'β': 'v', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'ζ': 'z', 'η': 'i',
    'θ': 'th', 'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm', 'ν': 'n', 'ξ': 'x',
    'ο': 'o', 'π': 'p', 'ρ': 'r', 'σ': 's', 'ς': 's', 'τ': 't', 'υ': 'y',
    'φ': 'f', 'χ': 'ch', 'ψ': 'ps', 'ω': 'o',
    'Α': 'A', 'Β': 'V', 'Γ': 'G', 'Δ': 'D', 'Ε': 'E', 'Ζ': 'Z', 'Η': 'I',
    'Θ': 'Th', 'Ι': 'I', 'Κ': 'K', 'Λ': 'L', 'Μ': 'M', 'Ν': 'N', 'Ξ': 'X',
    'Ο': 'O', 'Π': 'P', 'Ρ': 'R', 'Σ': 'S', 'Τ': 'T', 'Υ': 'Y', 'Φ': 'F',
    'Χ': 'Ch', 'Ψ': 'Ps', 'Ω': 'O',
    'ά': 'á', 'έ': 'é', 'ή': 'í', 'ί': 'í', 'ό': 'ó', 'ύ': 'ý', 'ώ': 'ó',
    'Ά': 'Á', 'Έ': 'É', 'Ή': 'Í', 'Ί': 'Í', 'Ό': 'Ó', 'Ύ': 'Ý', 'Ώ': 'Ó'
}

def transliterate_thai(text):
    """
    Convert Thai text to Latin alphabet.
    
    Args:
        text (str): Thai text
        
    Returns:
        str: Latin representation
    """
    result = ""
    for char in text:
        result += THAI_TO_LATIN.get(char, char)
    return result

def transliterate_hebrew(text):
    """
    Convert Hebrew text to Latin alphabet.
    
    Args:
        text (str): Hebrew text
        
    Returns:
        str: Latin representation
    """
    result = ""
    for char in text:
        result += HEBREW_TO_LATIN.get(char, char)
    return result

def transliterate_greek(text):
    """
    Convert Greek text to Latin alphabet.
    
    Args:
        text (str): Greek text
        
    Returns:
        str: Latin representation
    """
    result = ""
    for char in text:
        result += GREEK_TO_LATIN.get(char, char)
    return result

def get_transcription(text, lang_code):
    """
    Get appropriate transcription based on language code.
    
    Args:
        text (str): Text to transcribe
        lang_code (str): Language code
        
    Returns:
        str: Transcription if available, None otherwise
    """
    if lang_code == 'zh-CN' or lang_code == 'zh':
        return get_pinyin(text)
    elif lang_code == 'ru' or lang_code == 'uk':  # Russian and Ukrainian
        return transliterate_cyrillic(text)
    elif lang_code == 'ar' or lang_code == 'fa':  # Arabic and Persian
        return transliterate_arabic(text)
    elif lang_code == 'hi':
        return transliterate_hindi(text)
    elif lang_code == 'ko':
        return transliterate_korean(text)
    elif lang_code == 'ja':
        return transliterate_japanese(text)
    elif lang_code == 'th':
        return transliterate_thai(text)
    elif lang_code == 'he':
        return transliterate_hebrew(text)
    elif lang_code == 'el':
        return transliterate_greek(text)
    # Other languages like Vietnamese, Indonesian, etc. already use Latin script
    # or would require more specialized libraries
    return None

def add_transcription_to_translations(translations):
    """
    Add transcription to translations dictionary.
    
    Args:
        translations (dict): Dictionary of translations
        
    Returns:
        dict: Dictionary with added transcriptions
    """
    result = {}
    for lang_code, text in translations.items():
        result[lang_code] = {
            'text': text,
            'transcription': get_transcription(text, lang_code)
        }
    return result

def transcribe_audio(audio_file_path, lang_code=None):
    """
    Transcribe audio file to text using multiple AI services.
    First tries Claude via amurex (if available), then falls back to OpenAI Whisper API.
    
    Args:
        audio_file_path (str): Path to the audio file
        lang_code (str, optional): Language code to assist with transcription
        
    Returns:
        str: Transcribed text or a friendly error message if transcription fails
    """
    import os
    import os.path
    import subprocess
    import base64
    
    logger.info(f"Attempting to transcribe audio file: {audio_file_path}")
    
    # Check if the audio file exists
    if not os.path.exists(audio_file_path):
        logger.error(f"Audio file not found: {audio_file_path}")
        return "I couldn't find your voice message file. Please try sending it again."
    
    # Convert audio to mp3 which works better with transcription services
    mp3_path = audio_file_path + ".mp3"
    try:
        # Use ffmpeg to convert the file to mp3
        subprocess.run(
            ["ffmpeg", "-i", audio_file_path, "-acodec", "libmp3lame", "-y", mp3_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info(f"Converted audio to MP3 format: {mp3_path}")
        # Use the mp3 file if conversion succeeded
        transcription_file = mp3_path
    except Exception as e:
        logger.warning(f"Failed to convert audio to MP3: {e}, using original file")
        transcription_file = audio_file_path
    
    # First try to use Claude via amurex if available
    try:
        import amurex_ai
        
        if amurex_ai.is_available():
            logger.info("Attempting to transcribe audio with Claude via amurex")
            
            # Read the audio file and encode to base64 for Claude
            with open(transcription_file, "rb") as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            try:
                # Import directly from amurex for the transcription
                from amurex import get_client, ClientType, ResponseType, ModelFamily
                
                # Get the Claude client
                client = get_client(ClientType.Anthropic)
                
                # Create a prompt with language hint if provided
                language_hint = f"The audio is in {lang_code} language." if lang_code else ""
                prompt = (
                    f"Please transcribe the content of this audio file accurately. {language_hint} "
                    f"Provide ONLY the transcription with no additional commentary, explanations, or notes."
                )
                
                # Send the request to Claude with the audio
                response = client.generate(
                    model=ModelFamily.Claude35Sonnet.value,
                    prompt=prompt,
                    temperature=0.3,  # Lower temperature for more accurate transcription
                    max_tokens=2000,  # Allow plenty of tokens for longer audio
                    response_format=ResponseType.TEXT,
                    media=[{
                        "type": "audio",
                        "data": audio_base64
                    }]
                )
                
                # Extract and clean the transcription
                if response and response.content and len(response.content) > 0:
                    transcription_text = response.content[0].text.strip()
                    logger.info(f"Successfully transcribed with Claude: {transcription_text[:50]}..." if len(transcription_text) > 50 else f"Successfully transcribed with Claude: {transcription_text}")
                    
                    # Clean up temporary mp3 file
                    if os.path.exists(mp3_path) and mp3_path != audio_file_path:
                        try:
                            os.remove(mp3_path)
                            logger.info(f"Removed temporary MP3 file: {mp3_path}")
                        except Exception as cleanup_err:
                            logger.warning(f"Failed to remove temporary MP3 file: {cleanup_err}")
                    
                    return transcription_text
                else:
                    logger.warning("Claude returned empty transcription, falling back to OpenAI")
            except Exception as claude_err:
                logger.error(f"Error with Claude transcription: {claude_err}")
                # Fall through to OpenAI
        else:
            logger.info("Amurex/Claude not available for transcription, checking OpenAI")
    except ImportError:
        logger.warning("Amurex module not found for Claude transcription, checking OpenAI")
    except Exception as e:
        logger.error(f"Unexpected error with amurex/Claude: {e}")
    
    # Try Grok AI as a second option if available
    try:
        from xai import is_available as is_grok_available, transcribe_audio_with_grok
        
        if is_grok_available():
            logger.info("Attempting to transcribe audio with Grok AI")
            language_hint = f"The audio is in {lang_code} language." if lang_code else ""
            prompt = f"Please transcribe this audio file accurately. {language_hint} Provide only the transcription without any commentary."
            
            grok_transcription = transcribe_audio_with_grok(transcription_file, prompt=prompt)
            if grok_transcription:
                logger.info(f"Successfully transcribed with Grok AI: {grok_transcription[:50]}..." if len(grok_transcription) > 50 else f"Successfully transcribed with Grok AI: {grok_transcription}")
                
                # Clean up the mp3 file if we created one
                if os.path.exists(mp3_path) and mp3_path != audio_file_path:
                    try:
                        os.remove(mp3_path)
                        logger.info(f"Removed temporary MP3 file: {mp3_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary MP3 file: {e}")
                
                return grok_transcription
            else:
                logger.warning("Grok AI returned no transcription")
        else:
            logger.info("Grok AI not available for transcription, trying OpenAI")
    except ImportError:
        logger.warning("xAI module not found for Grok transcription, trying OpenAI")
    except Exception as e:
        logger.error(f"Error with Grok AI transcription: {e}")
    
    # Fall back to OpenAI Whisper API if available
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            
            with open(transcription_file, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=lang_code if lang_code else None
                )
            
            # Clean up the mp3 file if we created one
            if os.path.exists(mp3_path) and mp3_path != audio_file_path:
                try:
                    os.remove(mp3_path)
                    logger.info(f"Removed temporary MP3 file: {mp3_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary MP3 file: {e}")
            
            if transcription and hasattr(transcription, 'text'):
                logger.info(f"Successfully transcribed with OpenAI: {transcription.text[:50]}..." if len(transcription.text) > 50 else f"Successfully transcribed with OpenAI: {transcription.text}")
                return transcription.text.strip()
            else:
                logger.warning("OpenAI Whisper API returned no transcription text")
        
        except Exception as e:
            logger.error(f"Error using OpenAI Whisper API: {e}")
    else:
        logger.warning("OpenAI API key not available for transcription")
    
    # Try using SpeechRecognition as a fallback if available
    try:
        import speech_recognition as sr
        
        logger.info("Attempting to transcribe audio with SpeechRecognition")
        r = sr.Recognizer()
        
        # Set better parameters for pocketsphinx
        r.operation_timeout = 10  # Set a reasonable timeout
        
        # Set the language model if language hint is available
        language_hint_model = None
        if lang_code:
            if lang_code.startswith('en'):
                language_hint_model = "en-US"
            elif lang_code.startswith('es'):
                language_hint_model = "es-ES"
            elif lang_code.startswith('fr'):
                language_hint_model = "fr-FR"
            elif lang_code.startswith('de'):
                language_hint_model = "de-DE"
                
        with sr.AudioFile(transcription_file) as source:
            # Adjust for ambient noise for better recognition
            try:
                r.adjust_for_ambient_noise(source)
            except Exception as noise_err:
                logger.warning(f"Failed to adjust for ambient noise: {noise_err}")
                
            # Record the audio data
            audio_data = r.record(source)
            
            # Try to recognize with Google's free service first
            try:
                # Use language hint if available to improve recognition
                language_arg = language_hint_model if language_hint_model else None
                text = r.recognize_google(audio_data, language=language_arg)
                logger.info(f"Successfully transcribed with Google Speech Recognition: {text[:50]}..." if len(text) > 50 else f"Successfully transcribed with Google Speech Recognition: {text}")
                
                # Clean up the mp3 file if we created one
                if os.path.exists(mp3_path) and mp3_path != audio_file_path:
                    try:
                        os.remove(mp3_path)
                    except:
                        pass
                
                return text
            except Exception as google_err:
                logger.warning(f"Google Speech Recognition failed: {google_err}")
                
                # Try with Sphinx (offline engine using pocketsphinx) as last resort
                try:
                    # Configure Sphinx for better performance with the installed pocketsphinx
                    sphinx_config = {
                        "verbose": False,  # Less verbose output
                        "phrase_threshold": 0.1,  # Lower threshold for word detection
                        "keyword_entries": None  # No specific keywords
                    }
                    
                    # Add language model if available
                    if language_hint_model:
                        sphinx_config["language"] = language_hint_model.split('-')[0]  # Just use the first part (e.g., 'en')
                        
                    text = r.recognize_sphinx(audio_data, **sphinx_config)
                    logger.info(f"Successfully transcribed with Sphinx: {text[:50]}..." if len(text) > 50 else f"Successfully transcribed with Sphinx: {text}")
                    return text
                except Exception as sphinx_err:
                    logger.warning(f"Sphinx Recognition failed: {sphinx_err}")
    except ImportError:
        logger.warning("SpeechRecognition library not available")
    except Exception as e:
        logger.error(f"Error with SpeechRecognition: {e}")
    
    # If all transcription methods failed, return a friendly message
    return "I received your voice message, but couldn't transcribe it. Please send your message as text for translation."