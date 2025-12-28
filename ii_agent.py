"""
Intelligent Internet (II) Agent for voice message processing in the Enhanced Language Bot.
This module provides integration with II-Agent for improved voice transcription and handling.
"""

import logging
import os
import json
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Check if we have the required AI API keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")

# Try to import necessary libraries
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not found. Some II-Agent features may not be available.")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic library not found. Some II-Agent features may not be available.")

# Initialize clients
openai_client = None
anthropic_client = None
xai_client = None

if OPENAI_API_KEY and OPENAI_AVAILABLE:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized for II-Agent")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")

if ANTHROPIC_API_KEY and ANTHROPIC_AVAILABLE:
    try:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        logger.info("Anthropic client initialized for II-Agent")
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic client: {e}")

if XAI_API_KEY and OPENAI_AVAILABLE:
    try:
        xai_client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)
        logger.info("XAI client initialized for II-Agent")
    except Exception as e:
        logger.error(f"Failed to initialize XAI client: {e}")

def is_available() -> bool:
    """Check if any II-Agent service is available."""
    return any([openai_client, anthropic_client, xai_client])

async def process_voice_message(audio_file_path: str, language_hint: Optional[str] = None) -> Tuple[Optional[str], List[str]]:
    """
    Process a voice message using the II-Agent approach.
    Tries multiple AI models with fallback mechanisms and unified error handling.
    
    Args:
        audio_file_path: Path to the audio file
        language_hint: Optional language code to help with transcription
        
    Returns:
        Tuple containing (transcription, error_messages)
    """
    transcription = None
    error_messages = []
    
    # Try to convert audio to appropriate format for better compatibility
    try:
        import os
        import subprocess
        from pydub import AudioSegment
        import tempfile
        
        # First attempt to convert OGG to WAV for better compatibility
        wav_path = None
        mp3_path = None
        try:
            # Create temporary WAV file
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            
            # Convert using pydub (more reliable than ffmpeg direct call)
            audio = AudioSegment.from_file(audio_file_path)
            audio.export(wav_path, format="wav")
            logger.info(f"Successfully converted audio to WAV: {wav_path}")
            
            # Also create an MP3 version for additional compatibility
            fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            audio.export(mp3_path, format="mp3")
            logger.info(f"Successfully converted audio to MP3: {mp3_path}")
        except Exception as e:
            logger.warning(f"Could not convert audio format: {e}")
            # Continue with original file if conversion fails
            wav_path = audio_file_path
            mp3_path = None
    except ImportError:
        logger.warning("pydub not available, using original audio format")
        wav_path = audio_file_path
        mp3_path = None
    
    # 1. Try Whisper through OpenAI (best audio model)
    if openai_client:
        try:
            logger.info(f"Processing voice with OpenAI Whisper: {wav_path}")
            with open(wav_path, "rb") as audio_file:
                response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language_hint if language_hint else None
                )
            transcription = response.text
            if transcription:
                logger.info(f"OpenAI Whisper transcription successful: {transcription[:50]}...")
                return transcription, []
        except Exception as e:
            error_messages.append(f"OpenAI Whisper error: {str(e)}")
            logger.error(f"OpenAI Whisper transcription failed: {e}")
    
    # 2. Try Whisper through XAI as fallback
    if xai_client:
        try:
            logger.info(f"Processing voice with XAI Whisper: {wav_path}")
            with open(wav_path, "rb") as audio_file:
                response = xai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language_hint if language_hint else None
                )
            transcription = response.text
            if transcription:
                logger.info(f"XAI Whisper transcription successful: {transcription[:50]}...")
                return transcription, []
        except Exception as e:
            error_messages.append(f"XAI Whisper error: {str(e)}")
            logger.error(f"XAI Whisper transcription failed: {e}")
    
    # 3. Try to use local Whisper model as fallback (if installed)
    try:
        import whisper
        logger.info(f"Attempting to use local Whisper model: {wav_path}")
        
        model = whisper.load_model("base")
        result = model.transcribe(wav_path)
        transcription = result["text"]
        
        if transcription:
            logger.info(f"Local Whisper transcription successful: {transcription[:50]}...")
            return transcription, []
    except Exception as e:
        error_messages.append(f"Local Whisper error: {str(e)}")
        logger.warning(f"Local Whisper not available or failed: {e}")
    
    # 4. Try Claude for audio analysis (using a different approach)
    # Due to limitations with audio input, we'll try a different approach
    if anthropic_client and mp3_path:
        try:
            import base64
            import requests
            logger.info(f"Processing voice with Claude via image conversion: {mp3_path}")
            
            # Create a spectrogram image from the audio
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                from scipy.io import wavfile
                
                # Generate a spectrogram image
                fd, img_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                
                # Load audio file
                audio = AudioSegment.from_file(mp3_path)
                samples = np.array(audio.get_array_of_samples())
                
                # Generate spectrogram
                plt.figure(figsize=(10, 4))
                plt.specgram(samples, Fs=audio.frame_rate)
                plt.title("Audio Spectrogram (For Transcription)")
                plt.xlabel("Time (s)")
                plt.ylabel("Frequency (Hz)")
                plt.savefig(img_path)
                plt.close()
                
                # Now use the spectrogram with Claude
                with open(img_path, "rb") as img_file:
                    img_data = img_file.read()
                    img_base64 = base64.b64encode(img_data).decode("utf-8")
                
                # Create prompt for Claude
                prompt = (
                    "This is a spectrogram of an audio recording. Based on this visual representation, "
                    "can you identify any spoken words or phrases? Only provide the transcription text, no explanations."
                )
                if language_hint:
                    prompt += f" The audio is likely in {language_hint} language."
                
                # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                # do not change this unless explicitly requested by the user
                claude_response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    temperature=0.3,
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_base64}}
                        ]}
                    ]
                )
                
                transcription = claude_response.content[0].text
                if transcription:
                    # Clean up any explanatory text Claude might include
                    for prefix in ["Transcription:", "Here's the transcription:", "The transcription is:"]:
                        if transcription.startswith(prefix):
                            transcription = transcription[len(prefix):].strip()
                    
                    logger.info(f"Claude transcription successful: {transcription[:50]}...")
                    return transcription, []
                
                # Clean up the temporary image file
                os.remove(img_path)
            except Exception as e:
                logger.warning(f"Spectrogram creation failed: {e}")
                # Fall through to next attempt
                
        except Exception as e:
            error_messages.append(f"Claude error: {str(e)}")
            logger.error(f"Claude transcription failed: {e}")
    
    # 5. Try SpeechRecognition library as a fallback
    try:
        import speech_recognition as sr
        logger.info(f"Attempting to use SpeechRecognition: {wav_path}")
        
        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = r.record(source)
            # Try Google Speech Recognition
            transcription = r.recognize_google(audio_data)
            
        if transcription:
            logger.info(f"SpeechRecognition successful: {transcription[:50]}...")
            return transcription, []
    except Exception as e:
        error_messages.append(f"SpeechRecognition error: {str(e)}")
        logger.warning(f"SpeechRecognition failed: {e}")
    
    # Clean up temporary files
    try:
        if wav_path and wav_path != audio_file_path and os.path.exists(wav_path):
            os.remove(wav_path)
        if mp3_path and os.path.exists(mp3_path):
            os.remove(mp3_path)
    except Exception as e:
        logger.warning(f"Error cleaning up temporary files: {e}")
    
    # If all methods fail, return None with error messages
    if not error_messages:
        error_messages.append("All transcription services failed without specific errors")
    
    return None, error_messages

async def chat_with_ii_agent(messages: List[Dict[str, str]], 
                       audio_file_path: Optional[str] = None,
                       language_hint: Optional[str] = None) -> Optional[str]:
    """
    Chat with the II-Agent using available AI services.
    Can process both regular chat and audio messages.
    
    Args:
        messages: List of message dictionaries with role and content
        audio_file_path: Optional path to audio file for voice processing
        language_hint: Optional language code to help with processing
        
    Returns:
        str: AI response or None if all services fail
    """
    # First, check if this is an audio/pronunciation request
    is_audio_request = False
    audio_text = None
    
    # Only check the last message (user's request)
    if messages and messages[-1]["role"] == "user":
        user_query = messages[-1]["content"].lower()
        audio_keywords = [
            "pronounce", "pronunciation", "say", "speak", "audio", 
            "listen", "sound", "hear", "pronouncing", "speech",
            "generate audio", "text to speech", "tts", "voice"
        ]
        
        if any(keyword in user_query for keyword in audio_keywords):
            is_audio_request = True
            
            # Try to extract the text they want to hear
            import re
            # Look for quoted text
            quoted = re.findall(r'["\']([^"\']+)["\']', user_query)
            
            if quoted:
                # Take the first quoted string
                audio_text = quoted[0]
            else:
                # Look for text after common phrases
                patterns = [
                    r'(?:pronounce|say|speak|audio for|generate audio for|voice for|speech for|hear) [""]?([^.!?]+)[.!?]?',
                    r'(?:the word|the phrase|the sentence) [""]?([^.!?]+)[.!?]?',
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, user_query, re.IGNORECASE)
                    if matches:
                        audio_text = matches[0].strip()
                        break
    
    # For audio requests, send a special response
    if is_audio_request and audio_text:
        return (f"I'd be happy to help you hear the pronunciation of '{audio_text}'. "
                f"While I can't generate audio directly in chat mode, "
                f"I've copied this text and you can use the /audio command or switch to translation mode "
                f"to hear it pronounced correctly.\n\n"
                f"Text for audio: {audio_text}\n\n"
                f"You can also try the dedicated /audio command which works with your last message or translation.")
    
    # Try OpenAI (Grok) first
    if xai_client:
        try:
            model = "grok-2-1212"
            
            # Adding a system message if not present
            if not any(msg.get("role") == "system" for msg in messages):
                messages.insert(0, {
                    "role": "system",
                    "content": ("You are an AI assistant that helps with language learning and translation. "
                                "When asked to pronounce words or generate audio, explain that you can't "
                                "generate audio directly in chat mode, but suggest using the /audio command "
                                "or switching to translation mode.")
                })
            
            response = xai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error using XAI for chat: {e}")
    
    # Try Claude as fallback
    if anthropic_client:
        try:
            # Format messages for Claude
            claude_messages = []
            system_prompt = ("You are an AI assistant that helps with language learning and translation. "
                            "When asked to pronounce words or generate audio, explain that you can't "
                            "generate audio directly in chat mode, but suggest using the /audio command "
                            "or switching to translation mode.")
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    # Claude uses system prompt differently
                    system_prompt = content
                else:
                    claude_role = "user" if role == "user" else "assistant"
                    claude_messages.append({"role": claude_role, "content": content})
            
            # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            # do not change this unless explicitly requested by the user
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                system=system_prompt,
                messages=claude_messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error using Claude for chat: {e}")
    
    # If all else fails, return a fallback message
    return "I'm sorry, I'm having trouble connecting to AI services right now. Please try again later."