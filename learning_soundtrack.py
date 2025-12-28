"""
Personalized Language Learning Soundtrack Generator.
Creates customized music to enhance language learning sessions.
"""

import os
import logging
import uuid
import tempfile
import json
from typing import Optional, Dict, List, Tuple, Any
import random
import time
import subprocess
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directory for generated soundtracks
os.makedirs("audio/soundtrack", exist_ok=True)

# Soundtrack themes with corresponding parameters
SOUNDTRACK_THEMES = {
    "focus": {
        "tempo": "medium",
        "instruments": ["piano", "soft strings", "ambient pads"],
        "frequency_range": "alpha waves (8-12 Hz)",
        "mood": "calm concentration",
        "dynamics": "consistent"
    },
    "study": {
        "tempo": "slow",
        "instruments": ["piano", "cello", "gentle synth"],
        "frequency_range": "theta waves (4-8 Hz)",
        "mood": "deep focus",
        "dynamics": "minimal variation"
    },
    "energetic": {
        "tempo": "medium-fast",
        "instruments": ["piano", "electronic beats", "uplifting strings"],
        "frequency_range": "beta waves (12-30 Hz)",
        "mood": "energetic motivation",
        "dynamics": "builds and resolves"
    },
    "ambient": {
        "tempo": "very slow",
        "instruments": ["ambient pads", "atmospheric sounds", "gentle piano"],
        "frequency_range": "delta waves (0.5-4 Hz)",
        "mood": "background immersion",
        "dynamics": "extremely subtle"
    },
    "rhythm": {
        "tempo": "moderate",
        "instruments": ["drums", "percussion", "bass tones"],
        "frequency_range": "gamma waves (30-100 Hz)",
        "mood": "rhythmic entrainment",
        "dynamics": "pulsing"
    }
}

# Language-specific musical elements
LANGUAGE_MUSICAL_ELEMENTS = {
    "en": {  # English
        "scale": "major pentatonic",
        "rhythm": "4/4 time signature, moderate syncopation",
        "instruments": ["piano", "acoustic guitar", "light percussion"]
    },
    "es": {  # Spanish
        "scale": "phrygian mode",
        "rhythm": "3/4 time with flamenco influences",
        "instruments": ["spanish guitar", "light percussion", "strings"]
    },
    "fr": {  # French
        "scale": "impressionist harmonies",
        "rhythm": "flowing 4/4 with rubato",
        "instruments": ["piano", "accordion", "strings"]
    },
    "de": {  # German
        "scale": "minor and major scales",
        "rhythm": "strict 4/4 time signature",
        "instruments": ["piano", "brass", "strings"]
    },
    "it": {  # Italian
        "scale": "lyrical major scale",
        "rhythm": "flowing 6/8 time signature",
        "instruments": ["mandolin", "acoustic guitar", "light strings"]
    },
    "ja": {  # Japanese
        "scale": "pentatonic minor",
        "rhythm": "free-flowing time",
        "instruments": ["koto", "shakuhachi", "taiko drums"]
    },
    "zh": {  # Chinese
        "scale": "Chinese pentatonic",
        "rhythm": "flowing with space",
        "instruments": ["erhu", "guzheng", "bamboo flute"]
    },
    "ko": {  # Korean
        "scale": "Korean pentatonic",
        "rhythm": "jangdan rhythmic patterns",
        "instruments": ["gayageum", "daegeum", "janggu"]
    },
    "ru": {  # Russian
        "scale": "minor with modal inflections",
        "rhythm": "strong pulse with occasional asymmetric measures",
        "instruments": ["balalaika", "strings", "chorus"]
    },
    "ar": {  # Arabic
        "scale": "maqam scales",
        "rhythm": "complex rhythmic cycles",
        "instruments": ["oud", "qanun", "darbuka"]
    }
}

def generate_learning_soundtrack(language_code: str, 
                              theme: str = "focus", 
                              duration: int = 180, 
                              vocabulary: Optional[List[str]] = None) -> Optional[str]:
    """
    Generate a personalized language learning soundtrack.
    
    Args:
        language_code: ISO language code for the target language
        theme: Soundtrack theme (focus, study, energetic, ambient, rhythm)
        duration: Duration in seconds
        vocabulary: Optional list of vocabulary words to incorporate
        
    Returns:
        Path to the generated audio file or None if generation failed
    """
    try:
        logger.info(f"Generating {theme} soundtrack for {language_code} language learning")
        
        # Get theme parameters
        theme_params = SOUNDTRACK_THEMES.get(theme, SOUNDTRACK_THEMES["focus"])
        
        # Get language-specific musical elements
        lang_elements = LANGUAGE_MUSICAL_ELEMENTS.get(
            language_code, 
            LANGUAGE_MUSICAL_ELEMENTS.get("en")  # Default to English if language not found
        )
        
        # Generate soundtrack using AI if OpenAI, Claude, or XAI is available
        soundtrack_path = generate_soundtrack_with_ai(language_code, theme, lang_elements, theme_params, duration)
        
        # If AI generation failed, fall back to basic audio synthesis
        if not soundtrack_path:
            soundtrack_path = generate_basic_soundtrack(language_code, theme, duration)
        
        return soundtrack_path
    except Exception as e:
        logger.error(f"Error generating learning soundtrack: {str(e)}")
        return None

def generate_soundtrack_with_ai(language_code: str, 
                             theme: str, 
                             lang_elements: Dict[str, Any],
                             theme_params: Dict[str, Any],
                             duration: int) -> Optional[str]:
    """
    Generate a soundtrack using AI services (OpenAI, Claude, or XAI).
    
    Args:
        language_code: ISO language code
        theme: Soundtrack theme
        lang_elements: Language-specific musical elements
        theme_params: Theme-specific parameters
        duration: Duration in seconds
        
    Returns:
        Path to the generated audio file or None if generation failed
    """
    try:
        # Try to import the necessary modules
        import os
        
        # Check if any AI services are available
        openai_key = os.environ.get("OPENAI_API_KEY")
        claude_key = os.environ.get("ANTHROPIC_API_KEY") 
        xai_key = os.environ.get("XAI_API_KEY")
        
        # Try OpenAI music generation
        if openai_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_key)
                
                # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                # First, generate a music style prompt
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": 
                         "You are a music composition expert. Create a detailed musical prompt "
                         "for generating background music for language learning."},
                        {"role": "user", "content": 
                         f"Create a musical prompt for a {duration} second soundtrack that would "
                         f"help someone learn {get_language_name(language_code)}. "
                         f"Theme: {theme}. "
                         f"Style should incorporate: {lang_elements['scale']} scales, "
                         f"{lang_elements['rhythm']} rhythms, "
                         f"and instruments like {', '.join(lang_elements['instruments'])}. "
                         f"Mood should be {theme_params['mood']} with {theme_params['dynamics']} dynamics."}
                    ]
                )
                
                music_prompt = completion.choices[0].message.content
                
                # Try to generate music if MusicGen or similar is integrated
                # This is speculative and would depend on OpenAI offering a music generation endpoint
                try:
                    # This would be the ideal approach if/when OpenAI offers music generation
                    response = client.audio.speech.create(
                        model="musicgen-1",  # Hypothetical model name
                        prompt=music_prompt,
                        duration=min(duration, 300),  # Cap at 5 minutes
                        output_format="mp3"
                    )
                    
                    output_path = f"audio/soundtrack/learning_{language_code}_{theme}_{uuid.uuid4()}.mp3"
                    with open(output_path, "wb") as audio_file:
                        audio_file.write(response.content)
                    
                    return output_path
                except Exception as music_error:
                    logger.warning(f"OpenAI music generation failed: {music_error}")
                    # Continue to other methods
            except Exception as openai_error:
                logger.warning(f"OpenAI integration error: {openai_error}")
        
        # Try Claude for music generation
        if claude_key:
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=claude_key)
                
                # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Create a {duration} second audio soundtrack for learning {get_language_name(language_code)}. "
                                    f"Theme: {theme}. "
                                    f"Style should incorporate: {lang_elements['scale']} scales, "
                                    f"{lang_elements['rhythm']} rhythms, "
                                    f"and instruments like {', '.join(lang_elements['instruments'])}. "
                                    f"Mood should be {theme_params['mood']} with {theme_params['dynamics']} dynamics. "
                                    f"Return just the audio without explanation."
                                }
                            ]
                        }
                    ]
                )
                
                # Look for audio in the response
                audio_found = False
                for content in response.content:
                    if hasattr(content, 'source') and hasattr(content.source, 'media_type'):
                        if 'audio' in content.source.media_type:
                            # Extract and save the audio
                            audio_data = base64.b64decode(content.source.data)
                            output_path = f"audio/soundtrack/learning_{language_code}_{theme}_{uuid.uuid4()}.mp3"
                            with open(output_path, "wb") as audio_file:
                                audio_file.write(audio_data)
                            
                            audio_found = True
                            logger.info(f"Successfully generated soundtrack using Claude")
                            return output_path
                
                if not audio_found:
                    logger.warning("Claude did not return audio data")
                    
            except Exception as claude_error:
                logger.warning(f"Claude integration error: {claude_error}")
        
        # Try XAI/Grok for music generation
        if xai_key:
            try:
                from openai import OpenAI
                client = OpenAI(
                    api_key=xai_key,
                    base_url="https://api.x.ai/v1"
                )
                
                # Generate a music prompt with Grok
                completion = client.chat.completions.create(
                    model="grok-2-1212",
                    messages=[
                        {"role": "system", "content": 
                         "You are a music composition expert. Create a detailed musical prompt "
                         "for generating background music for language learning."},
                        {"role": "user", "content": 
                         f"Create a musical prompt for a {duration} second soundtrack that would "
                         f"help someone learn {get_language_name(language_code)}. "
                         f"Theme: {theme}. "
                         f"Style should incorporate: {lang_elements['scale']} scales, "
                         f"{lang_elements['rhythm']} rhythms, "
                         f"and instruments like {', '.join(lang_elements['instruments'])}. "
                         f"Mood should be {theme_params['mood']} with {theme_params['dynamics']} dynamics."}
                    ]
                )
                
                music_prompt = completion.choices[0].message.content
                
                # Try to generate music
                # This is speculative and would depend on XAI offering a music generation endpoint
                try:
                    # Hypothetical XAI music generation
                    response = client.audio.speech.create(
                        model="musicgen-1",  # Hypothetical model name
                        prompt=music_prompt,
                        duration=min(duration, 300),  # Cap at 5 minutes
                        output_format="mp3"
                    )
                    
                    output_path = f"audio/soundtrack/learning_{language_code}_{theme}_{uuid.uuid4()}.mp3"
                    with open(output_path, "wb") as audio_file:
                        audio_file.write(response.content)
                    
                    return output_path
                except Exception as music_error:
                    logger.warning(f"XAI music generation failed: {music_error}")
                    # Continue to other methods
            except Exception as xai_error:
                logger.warning(f"XAI integration error: {xai_error}")
        
        # If all AI methods fail, return None to fall back to basic synthesis
        return None
    except Exception as e:
        logger.error(f"Error in AI soundtrack generation: {str(e)}")
        return None

def generate_basic_soundtrack(language_code: str, theme: str, duration: int) -> Optional[str]:
    """
    Generate a basic soundtrack using pydub.
    
    Args:
        language_code: ISO language code
        theme: Soundtrack theme
        duration: Duration in seconds
        
    Returns:
        Path to the generated audio file or None if generation failed
    """
    try:
        # Define basic parameters based on theme
        if theme == "focus":
            bpm = 60
            base_freq = 100
            wave_type = "sine"
        elif theme == "study":
            bpm = 50
            base_freq = 80
            wave_type = "sine"
        elif theme == "energetic":
            bpm = 80
            base_freq = 120
            wave_type = "triangle"
        elif theme == "ambient":
            bpm = 40
            base_freq = 60
            wave_type = "sine"
        elif theme == "rhythm":
            bpm = 70
            base_freq = 140
            wave_type = "square"
        else:
            bpm = 60
            base_freq = 100
            wave_type = "sine"
        
        # Language influence (subtle variation)
        language_factor = ord(language_code[0]) % 7  # Simple way to get variation based on language
        bpm += language_factor
        base_freq += language_factor * 5
        
        # Create base track
        duration_ms = duration * 1000
        base_track = AudioSegment.silent(duration=duration_ms)
        
        # Add main tone
        if wave_type == "sine":
            tone = Sine(base_freq).to_audio_segment(duration=500)
        else:
            # Fallback to sine if other wave types are not implemented
            tone = Sine(base_freq).to_audio_segment(duration=500)
        
        # Create pattern
        pattern_duration = int(60000 / bpm)  # Convert BPM to ms per beat
        pattern = AudioSegment.silent(duration=pattern_duration)
        
        # Add tone at the beginning of pattern
        pattern = pattern.overlay(tone, position=0)
        
        # Add tone at 1/3 of pattern for some themes
        if theme in ["focus", "rhythm", "energetic"]:
            pattern = pattern.overlay(tone._spawn(tone.raw_data, overrides={
                "frame_rate": int(tone.frame_rate * 1.5)
            }).set_frame_rate(tone.frame_rate), position=pattern_duration // 3)
        
        # Repeat pattern to fill duration
        num_patterns = int(duration_ms / pattern_duration) + 1
        for i in range(num_patterns):
            position = i * pattern_duration
            # Vary the volume to create dynamics
            volume_factor = 1.0
            if theme == "energetic":
                # Increase volume over time for energetic theme
                volume_factor = min(1.0, 0.7 + (i / num_patterns) * 0.5)
            elif theme == "ambient":
                # Random subtle variations for ambient theme
                volume_factor = 0.7 + random.random() * 0.3
            
            # Apply volume adjustment
            adjusted_pattern = pattern - (20 * (1 - volume_factor))  # dB adjustment
            
            # Overlay at position
            if position + pattern_duration <= duration_ms:
                base_track = base_track.overlay(adjusted_pattern, position=position)
        
        # Add background ambience
        ambience = WhiteNoise().to_audio_segment(duration=duration_ms)
        # Lower volume of ambience significantly
        ambience = ambience - 30  # -30 dB
        base_track = base_track.overlay(ambience)
        
        # Export the final track
        output_path = f"audio/soundtrack/learning_{language_code}_{theme}_{uuid.uuid4()}.mp3"
        base_track.export(output_path, format="mp3")
        
        return output_path
    except Exception as e:
        logger.error(f"Error in basic soundtrack generation: {str(e)}")
        return None

def get_language_name(lang_code: str) -> str:
    """
    Get the full name of a language from its code.
    
    Args:
        lang_code: ISO language code
        
    Returns:
        Language name
    """
    # Map common language codes to full names
    name_map = {
        'en': 'English',
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'ja': 'Japanese',
        'zh': 'Chinese',
        'ko': 'Korean',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'nl': 'Dutch',
        'tr': 'Turkish',
        'pl': 'Polish',
        'sv': 'Swedish',
        'fi': 'Finnish',
        'da': 'Danish',
        'no': 'Norwegian',
        'cs': 'Czech',
        'hu': 'Hungarian',
        'el': 'Greek',
        'he': 'Hebrew',
        'th': 'Thai',
        'vi': 'Vietnamese',
        'uk': 'Ukrainian',
        'id': 'Indonesian',
        'ms': 'Malay',
        'fa': 'Persian',
        'tl': 'Filipino',
    }
    
    # Return the language name or the code itself if not found
    return name_map.get(lang_code, lang_code)

def get_available_themes() -> List[str]:
    """
    Get a list of available soundtrack themes.
    
    Returns:
        List of theme names
    """
    return list(SOUNDTRACK_THEMES.keys())