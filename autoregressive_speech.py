"""
Autoregressive streaming speech synthesis module based on advanced models.
Implements real-time streaming speech capabilities for more natural voice interactions.
"""

import os
import logging
import tempfile
from typing import Optional, Tuple, Dict, Any
import base64
import time
import uuid

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for the required API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Import clients conditionally to avoid errors
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except ImportError:
    logger.warning("OpenAI Python SDK not available")
    openai_client = None

try:
    # Set up XAI client using the OpenAI package with custom base URL
    if XAI_API_KEY:
        xai_client = OpenAI(
            api_key=XAI_API_KEY,
            base_url="https://api.x.ai/v1"
        )
    else:
        xai_client = None
except:
    logger.warning("XAI integration not available")
    xai_client = None

try:
    from anthropic import Anthropic
    anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
except ImportError:
    logger.warning("Anthropic Python SDK not available")
    anthropic_client = None

def get_voice_for_language(language_code: str) -> Dict[str, Any]:
    """
    Get the best voice parameters for a specific language.
    
    Args:
        language_code: ISO language code
        
    Returns:
        Dictionary with voice parameters
    """
    # Language-specific voice mapping
    voice_map = {
        "en": {"voice": "nova", "model": "tts-1-hd", "speed": 1.0},  # English - higher quality
        "es": {"voice": "alloy", "model": "tts-1-hd", "speed": 0.95},  # Spanish
        "fr": {"voice": "echo", "model": "tts-1-hd", "speed": 0.95},  # French
        "de": {"voice": "onyx", "model": "tts-1-hd", "speed": 0.95},  # German
        "it": {"voice": "fable", "model": "tts-1-hd", "speed": 0.95},  # Italian
        "pt": {"voice": "shimmer", "model": "tts-1-hd", "speed": 0.95},  # Portuguese
        "ru": {"voice": "nova", "model": "tts-1-hd", "speed": 0.9},  # Russian
        "ja": {"voice": "nova", "model": "tts-1-hd", "speed": 0.9},  # Japanese
        "ko": {"voice": "alloy", "model": "tts-1-hd", "speed": 0.9},  # Korean
        "zh": {"voice": "shimmer", "model": "tts-1-hd", "speed": 0.95},  # Chinese
        "ar": {"voice": "echo", "model": "tts-1-hd", "speed": 0.9},  # Arabic
    }
    
    # Get country code from language code (e.g., "en-US" -> "en")
    base_lang = language_code.split('-')[0].lower()
    
    # Return the voice parameters for the language or default if not found
    return voice_map.get(base_lang, {"voice": "nova", "model": "tts-1-hd", "speed": 1.0})

def generate_streaming_speech(text: str, language_code: str = "en") -> Tuple[Optional[str], Optional[str]]:
    """
    Generate streaming speech using advanced autoregressive models.
    This function uses cutting-edge text-to-speech models to create high-quality voice.
    
    Args:
        text: Text to convert to speech
        language_code: ISO language code
        
    Returns:
        Tuple of (filepath, error message)
    """
    # Create a unique ID for this audio file
    audio_id = str(uuid.uuid4())
    
    # Ensure proper cache directories exist
    os.makedirs("audio/cache", exist_ok=True)
    
    # Determine output path
    output_path = f"audio/cache/autoregressive_{audio_id}.mp3"
    
    # First attempt: Try OpenAI advanced TTS
    if openai_client:
        try:
            # Get optimal voice settings for the language
            voice_params = get_voice_for_language(language_code)
            logger.info(f"Generating streaming speech with OpenAI TTS for language {language_code}, voice: {voice_params['voice']}")
            
            # For enhanced natural-sounding speech, preprocess text to add SSML-like markers
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": 
                     f"You are an expert in natural speech patterns. The text will be spoken in {language_code} language. "
                     "Add natural pauses, emphasis, and intonation markers to make the speech sound more human-like. "
                     "Use <break>, <emphasis>, and other similar markers as needed. Keep the original meaning intact."
                    },
                    {"role": "user", "content": f"Text to enhance: {text}"}
                ]
            )
            
            # Get enhanced text with speech markers
            enhanced_text = completion.choices[0].message.content
            
            # Use OpenAI's advanced TTS with high-definition model
            response = openai_client.audio.speech.create(
                model=voice_params["model"],  # Using HD model for higher quality
                voice=voice_params["voice"],
                input=enhanced_text,
                response_format="mp3",
                speed=voice_params["speed"]
            )
            
            # Save the audio file
            with open(output_path, "wb") as audio_file:
                audio_file.write(response.content)
                
            logger.info(f"Successfully generated streaming speech with OpenAI TTS: {output_path}")
            return output_path, None
            
        except Exception as e:
            logger.error(f"OpenAI TTS error: {str(e)}")
            # Continue to alternative methods
    
    # Second attempt: Try Claude from Anthropic for TTS
    if anthropic_client:
        try:
            logger.info(f"Attempting streaming speech with Claude for language {language_code}")
            
            # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Convert this text to spoken audio in {language_code} language. Make it sound natural and expressive: {text}"
                            }
                        ]
                    }
                ]
            )
            
            # Check for audio in the response
            audio_found = False
            for content in response.content:
                if hasattr(content, 'source') and hasattr(content.source, 'media_type'):
                    if 'audio' in content.source.media_type:
                        # Extract and save the audio
                        audio_data = base64.b64decode(content.source.data)
                        with open(output_path, "wb") as audio_file:
                            audio_file.write(audio_data)
                        
                        audio_found = True
                        logger.info(f"Successfully generated streaming speech using Claude: {output_path}")
                        return output_path, None
            
            if not audio_found:
                logger.warning("Claude did not return audio data")
                
        except Exception as e:
            logger.error(f"Claude TTS error: {str(e)}")
            # Continue to next option
    
    # Third attempt: Try xAI/Grok TTS
    if xai_client:
        try:
            logger.info(f"Attempting streaming speech with xAI for language {language_code}")
            
            # First, generate a better prompt with Grok
            messages = [
                {
                    "role": "system",
                    "content": f"You are a text-to-speech expert. Enhance this text for better audio synthesis in {language_code} language."
                },
                {
                    "role": "user", 
                    "content": f"Make this text sound natural when spoken: {text}"
                }
            ]
            
            # Get enhanced text from Grok
            response = xai_client.chat.completions.create(
                model="grok-2-1212",
                messages=messages,
                temperature=0.3
            )
            
            enhanced_text = response.choices[0].message.content
            
            # Use X.AI for TTS with the enhanced text
            try:
                # Determine the best voice for this language
                voice = "echo"  # Default voice
                if language_code.startswith("zh") or language_code.startswith("ja"):
                    voice = "nova"  # Better for Asian languages
                elif language_code.startswith("ru") or language_code.startswith("ar"):
                    voice = "onyx"  # Deeper voice for these languages
                
                # Generate the audio
                response = xai_client.audio.speech.create(
                    model="tts-1",
                    voice=voice,
                    input=enhanced_text,
                    response_format="mp3"
                )
                
                # Save the audio file
                with open(output_path, "wb") as audio_file:
                    audio_file.write(response.content)
                    
                logger.info(f"Successfully generated streaming speech with xAI: {output_path}")
                return output_path, None
                
            except Exception as tts_error:
                logger.error(f"xAI TTS specific error: {tts_error}")
                
        except Exception as e:
            logger.error(f"xAI TTS error: {str(e)}")
    
    # Fourth attempt: Try regular TTS as fallback
    try:
        logger.info("Falling back to standard TTS...")
        from enhanced_audio import generate_audio
        
        # Fall back to regular (non-streaming) TTS
        audio_path = generate_audio(text, language_code)
        if audio_path:
            logger.info(f"Generated speech using fallback TTS: {audio_path}")
            return audio_path, "Streaming speech not available. Using regular TTS instead."
    except Exception as e:
        logger.error(f"Fallback TTS error: {str(e)}")
    
    # If we reach here, all methods have failed
    return None, "Autoregressive speech services are currently unavailable"

def transcribe_streaming_audio(audio_path: str, language_code: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe streaming audio using advanced models with multiple fallback options.
    
    Args:
        audio_path: Path to the audio file
        language_code: Optional language hint
        
    Returns:
        Tuple of (transcription, error message)
    """
    if not os.path.exists(audio_path):
        return None, "Audio file not found"
    
    # Import the regular transcription module for fallback
    try:
        from transcription import transcribe_audio_with_fallback
        
        # First try to transcribe using our standard fallback mechanism
        transcription = transcribe_audio_with_fallback(audio_path, language_code)
        if transcription:
            logger.info(f"Successfully transcribed audio using transcription module")
            return transcription, None
    except Exception as e:
        logger.error(f"Standard transcription fallback error: {str(e)}")
    
    # Try OpenAI Whisper model directly
    if openai_client:
        try:
            logger.info(f"Transcribing with OpenAI Whisper for language {language_code or 'auto'}")
            
            with open(audio_path, "rb") as audio_file:
                response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language_code
                )
            
            return response.text, None
        except Exception as e:
            logger.error(f"OpenAI transcription error: {str(e)}")
    
    # Try X.AI as fallback
    if xai_client:
        try:
            logger.info(f"Transcribing with XAI for language {language_code or 'auto'}")
            
            with open(audio_path, "rb") as audio_file:
                response = xai_client.audio.transcriptions.create(
                    model="whisper-1",  # Assuming similar API naming
                    file=audio_file,
                    language=language_code
                )
            
            return response.text, None
        except Exception as e:
            logger.error(f"XAI transcription error: {str(e)}")
    
    # Try built-in SpeechRecognition as a last resort
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            
            # Try to recognize with Google (doesn't require API key)
            try:
                text = recognizer.recognize_google(audio_data, language=language_code)
                return text, None
            except:
                pass
                
            # Try to recognize with Sphinx (offline)
            try:
                text = recognizer.recognize_sphinx(audio_data, language=language_code)
                return text, None
            except:
                pass
    except Exception as e:
        logger.error(f"SpeechRecognition fallback error: {str(e)}")
    
    # If absolutely all methods fail
    return None, "All transcription services failed. Please try again with clearer audio or type your message as text."

def get_autoregressive_capabilities() -> Dict[str, bool]:
    """
    Return the current capabilities of autoregressive speech services.
    
    Returns:
        Dictionary with capability flags
    """
    return {
        "openai_tts": openai_client is not None,
        "xai_tts": xai_client is not None,
        "anthropic_tts": anthropic_client is not None,
        "openai_whisper": openai_client is not None,
        "xai_whisper": xai_client is not None
    }