"""
Enhanced audio generation for the language learning bot.
Provides multiple fallback TTS options for better quality voice output.
"""

import logging
import os
import tempfile
import time
import base64
from typing import Optional

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Try to import gTTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    logger.warning("gTTS (Google Text-to-Speech) not available.")

# Constants for voice selections
OPENAI_VOICES = {
    "alloy": "Neutral, versatile voice",
    "echo": "Clear, crisp voice",
    "fable": "British accent, authoritative",
    "onyx": "Deep, resonant voice",
    "nova": "Warm, friendly female voice",
    "shimmer": "Cheerful, bright voice"
}

DEFAULT_VOICE = "nova"

def select_voice_for_language(lang_code: str) -> str:
    """
    Select an appropriate voice for the language.
    This is a simplified version; in a more advanced system,
    we would map specific voices to each language.
    
    Args:
        lang_code (str): Language code
        
    Returns:
        str: Voice identifier
    """
    # For now, just use the default voice for all languages
    # In a more sophisticated version, we would have specific voices 
    # for different language families
    return DEFAULT_VOICE

def generate_audio_with_openai(text: str, lang_code: str) -> Optional[str]:
    """
    Generate audio using OpenAI's TTS API.
    
    Args:
        text (str): Text to convert to speech
        lang_code (str): Language code
        
    Returns:
        str: Path to audio file or None if generation fails
    """
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("OpenAI API key not available. Cannot use TTS API.")
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        # Select appropriate voice
        voice = select_voice_for_language(lang_code)
        
        # Create audio directory if it doesn't exist
        os.makedirs('audio', exist_ok=True)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            # Generate speech with OpenAI
            response = client.audio.speech.create(
                model="tts-1", 
                voice=voice,
                input=text
            )
            
            # Save to file
            response.stream_to_file(temp_file.name)
            
            logger.info(f"Generated audio with OpenAI TTS: {temp_file.name}")
            return temp_file.name
            
    except Exception as e:
        logger.error(f"Error generating audio with OpenAI: {e}")
        return None

def generate_audio_with_grok(text: str, lang_code: str) -> Optional[str]:
    """
    Generate audio using Grok TTS capabilities.
    This uses the synchronous version of the API to avoid async issues.
    
    Args:
        text (str): Text to convert to speech
        lang_code (str): Language code
        
    Returns:
        str: Path to audio file or None if generation fails
    """
    try:
        # Import xAI module directly to access client
        import xai
        
        if not xai.is_available() or not xai.xai_client:
            logger.warning("Grok not available. Cannot use for TTS.")
            return None
        
        # Prepare system prompt
        system_prompt = (
            "You are a text-to-speech engine. Given the following text, "
            "please convert it to base64-encoded audio data. "
            "The response should be formatted as a JSON object with a single key 'audio_base64' "
            "containing the base64-encoded audio data."
        )
        
        # Prepare user prompt
        user_prompt = f"Convert this text to speech in {lang_code} language: {text}"
        
        # Create messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Use synchronous API client directly
        try:
            completion = xai.xai_client.chat.completions.create(
                model=xai.GROK_TEXT_MODEL,  # Use constant from xai module
                messages=messages,
                temperature=0.1
            )
            # Get response text from completion
            response = completion.choices[0].message.content
        except Exception as client_error:
            logger.error(f"Error calling Grok API: {client_error}")
            return None
        
        if not response:
            logger.warning("No response from Grok for TTS")
            return None
        
        # Try to extract base64 data from response
        try:
            import json
            import re
            
            # Try to parse as JSON first
            try:
                data = json.loads(response)
                if "audio_base64" in data:
                    audio_base64 = data["audio_base64"]
                else:
                    # Look for other possible keys containing audio data
                    for key in data:
                        if "audio" in key.lower() and isinstance(data[key], str):
                            audio_base64 = data[key]
                            break
                    else:
                        raise ValueError("No audio data found in JSON response")
            except json.JSONDecodeError:
                # Try to extract base64 data using regex
                base64_pattern = r'base64,((?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?)'
                match = re.search(base64_pattern, response)
                if match:
                    audio_base64 = match.group(1)
                else:
                    raise ValueError("Could not extract base64 data from response")
            
            # Decode base64 data and save to file
            audio_data = base64.b64decode(audio_base64)
            
            # Create audio directory if it doesn't exist
            os.makedirs('audio', exist_ok=True)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_file.write(audio_data)
                logger.info(f"Generated audio with Grok: {temp_file.name}")
                return temp_file.name
                
        except Exception as e:
            logger.error(f"Error decoding audio data from Grok: {e}")
            return None
            
    except ImportError:
        logger.warning("xAI module not available. Cannot use Grok for TTS.")
        return None
    except Exception as e:
        logger.error(f"Error generating audio with Grok: {e}")
        return None

def generate_audio_with_gtts(text: str, lang_code: str) -> Optional[str]:
    """
    Generate audio using Google Text-to-Speech (gTTS).
    
    Args:
        text (str): Text to convert to speech
        lang_code (str): Language code
        
    Returns:
        str: Path to audio file or None if generation fails
    """
    if not GTTS_AVAILABLE:
        logger.warning("gTTS not available. Cannot generate audio.")
        return None
        
    try:
        # Adjust language code for gTTS if needed
        tts_lang = lang_code
        if lang_code == 'zh-CN':
            tts_lang = 'zh-cn'
        
        # Create audio directory if it doesn't exist
        os.makedirs('audio', exist_ok=True)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            # Generate audio
            tts = gTTS(text=text, lang=tts_lang, slow=False)
            tts.save(temp_file.name)
            
            logger.info(f"Generated audio with gTTS: {temp_file.name}")
            return temp_file.name
    except Exception as e:
        logger.error(f"Error generating audio with gTTS: {e}")
        return None

def generate_layered_audio(text: str, layers: int = 3, duration: int = 30, 
                       genre: Optional[str] = None, mood: Optional[str] = None) -> Optional[str]:
    """
    Generate layered audio for music creation.
    This creates multiple audio tracks and combines them for a richer music experience.
    
    Args:
        text (str): Text to inspire the music
        layers (int): Number of audio layers to generate
        duration (int): Approximate duration in seconds
        genre (str, optional): Music genre
        mood (str, optional): Emotional mood
        
    Returns:
        str: Path to the generated audio file or None if failed
    """
    try:
        import os
        import uuid
        import subprocess
        import tempfile
        from openai import OpenAI
        
        # Check if OpenAI API key is available
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("OpenAI API key not available. Cannot generate layered audio.")
            return None
        
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        # Create unique ID for this music file
        music_id = str(uuid.uuid4())
        
        # Ensure directories exist
        os.makedirs("audio/music_cache", exist_ok=True)
        
        # Output path for final mixed audio
        output_path = f"audio/music_cache/layered_{music_id}.mp3"
        
        # Generate layer prompts using GPT
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": 
                 "You are a music composition expert. Create layered musical instructions for a multi-track composition. "
                 "For each track, provide specific musical elements including: instrument, rhythm pattern, "
                 "melodic motif, emotional quality, and technical specifications. Make sure the tracks complement each other."
                },
                {"role": "user", "content": 
                 f"Create {layers} complementary audio layers for music inspired by: '{text}'. "
                 f"Genre: {genre or 'suitable genre based on the text'}. "
                 f"Mood: {mood or 'appropriate mood based on the text'}. "
                 f"Create a JSON array with {layers} objects, each containing: 'instrument', 'pattern', 'description'."}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse layer specifications
        import json
        layer_specs = json.loads(completion.choices[0].message.content)
        
        # List to store paths to temporary audio files for each layer
        temp_files = []
        
        # Generate audio for each layer
        for i, layer_spec in enumerate(layer_specs.get("layers", [])):
            if i >= layers:  # Respect the requested number of layers
                break
                
            instrument = layer_spec.get("instrument", "piano")
            pattern = layer_spec.get("pattern", "simple melody")
            description = layer_spec.get("description", f"Layer {i+1}")
            
            # Create a TTS prompt for this layer
            layer_prompt = f"Musical {instrument} playing a {pattern}. {description}"
            
            # Adjust voice based on instrument type
            voice = "alloy"  # default
            if "bass" in instrument.lower() or "cello" in instrument.lower():
                voice = "onyx"  # deeper voice for bass instruments
            elif "flute" in instrument.lower() or "violin" in instrument.lower():
                voice = "nova"  # higher voice for treble instruments
            elif "drum" in instrument.lower() or "percussion" in instrument.lower():
                voice = "echo"  # crisp voice for percussion
                
            # Generate the layer audio
            try:
                with tempfile.NamedTemporaryFile(suffix=f'_layer_{i}.mp3', delete=False) as temp_file:
                    # Generate speech with musical qualities
                    response = client.audio.speech.create(
                        model="tts-1-hd",  # using high definition model for better audio quality
                        voice=voice,
                        input=layer_prompt,
                        speed=0.8,  # slower for more musical quality
                    )
                    
                    # Save to file
                    response.stream_to_file(temp_file.name)
                    temp_files.append(temp_file.name)
                    logger.info(f"Generated layer {i+1} with {instrument}")
            
            except Exception as e:
                logger.error(f"Error generating layer {i+1}: {e}")
        
        # If we have at least one layer, we can proceed
        if temp_files:
            try:
                # Use ffmpeg to mix the layers
                # Import temp files as inputs
                inputs = []
                for file in temp_files:
                    inputs.extend(["-i", file])
                
                # Create filter_complex for mixing
                filter_complex = ""
                for i in range(len(temp_files)):
                    filter_complex += f"[{i}:a]"
                filter_complex += f"amix=inputs={len(temp_files)}:duration=longest[aout]"
                
                # Combine command
                cmd = ["ffmpeg", *inputs, "-filter_complex", filter_complex, "-map", "[aout]", output_path]
                
                # Run ffmpeg to mix the audio
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Clean up temporary files
                for file in temp_files:
                    try:
                        os.remove(file)
                    except:
                        pass
                
                logger.info(f"Successfully generated layered audio: {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"Error mixing audio layers: {e}")
                
                # If mixing fails but we have individual tracks, return the first one
                if temp_files:
                    logger.warning("Mixing failed, returning first layer only")
                    return temp_files[0]
        
        # If we couldn't generate any layers, return None
        logger.warning("Could not generate any audio layers")
        return None
        
    except Exception as e:
        logger.error(f"Error in layered audio generation: {e}")
        return None

def generate_audio_with_anthropic(text: str, lang_code: str) -> Optional[str]:
    """
    Generate audio using Anthropic's Claude capabilities.
    
    Args:
        text (str): Text to convert to speech
        lang_code (str): Language code
        
    Returns:
        str: Path to audio file or None if generation fails
    """
    try:
        # Check if Anthropic API key is available
        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("Anthropic API key not available. Cannot use Claude for audio.")
            return None
            
        from anthropic import Anthropic
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        
        # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please convert the following text to spoken audio in {lang_code} language. Return the audio file only without any explanation:\n\n{text}"
                        }
                    ]
                }
            ]
        )
        
        # Check for audio in the response
        for content in response.content:
            if hasattr(content, 'source') and hasattr(content.source, 'media_type'):
                if 'audio' in content.source.media_type:
                    # Create audio directory if it doesn't exist
                    os.makedirs('audio', exist_ok=True)
                    
                    # Extract and save the audio to a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                        audio_data = base64.b64decode(content.source.data)
                        temp_file.write(audio_data)
                        logger.info(f"Generated audio with Claude: {temp_file.name}")
                        return temp_file.name
                        
        logger.warning("Claude did not return audio data")
        return None
        
    except ImportError:
        logger.warning("Anthropic Python SDK not available.")
        return None
    except Exception as e:
        logger.error(f"Error generating audio with Claude: {e}")
        return None

def generate_audio(text: str, lang_code: str) -> Optional[str]:
    """
    Generate audio using ElevenLabs with fallback mechanisms.
    Try ElevenLabs first (best quality), then gTTS as fallback.
    
    Args:
        text (str): Text to convert to speech
        lang_code (str): Language code
        
    Returns:
        str: Path to audio file or None if all methods fail
    """
    # Try ElevenLabs first (highest quality)
    try:
        from elevenlabs_voice import generate_elevenlabs_audio
        audio_path = generate_elevenlabs_audio(text, lang_code)
        if audio_path:
            logger.info(f"Successfully generated audio with ElevenLabs")
            return audio_path
    except Exception as e:
        logger.warning(f"ElevenLabs TTS failed: {e}")
    
    # Fall back to gTTS
    return generate_audio_with_gtts(text, lang_code)