"""
Music generation module based on state-of-the-art open source music models.
This module implements the /makemusic command functionality.
"""

import os
import logging
import uuid
import tempfile
from typing import Optional, Tuple, Dict, Any
import json
import base64
import time

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

def convert_text_to_musical_prompt(text: str, genre: Optional[str] = None, mood: Optional[str] = None) -> str:
    """
    Convert text to a musical prompt that can guide the generation process.
    
    Args:
        text: The text to convert to music
        genre: Optional music genre
        mood: Optional mood
        
    Returns:
        A prompt for the music generator
    """
    # Use OpenAI to generate a creative musical prompt
    if openai_client:
        try:
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
            # do not change this unless explicitly requested by the user
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": 
                     "You are a music generation AI assistant. Create a detailed musical prompt based on the given text. "
                     "The prompt should include genre, mood, tempo, instruments, and style information. "
                     "Make creative connections between the text content and musical elements. "
                     "Output in JSON format with keys: 'musical_prompt', 'genre', 'mood', 'tempo', 'instruments'."
                    },
                    {"role": "user", "content": f"Text: {text}\nPreferred genre: {genre or 'any'}\nPreferred mood: {mood or 'any'}"}
                ],
                response_format={"type": "json_object"}
            )
            
            try:
                result = json.loads(completion.choices[0].message.content)
                # Return the musical prompt
                return result.get('musical_prompt', text)
            except Exception as e:
                logger.error(f"Error parsing musical prompt: {str(e)}")
                return f"Create music inspired by: {text}, genre: {genre or 'suitable genre'}, mood: {mood or 'appropriate mood'}"
        
        except Exception as e:
            logger.error(f"Error generating musical prompt: {str(e)}")
    
    # Fallback to simple prompt creation
    genre_text = f" in {genre} style" if genre else ""
    mood_text = f" with {mood} mood" if mood else ""
    return f"Create music inspired by: {text}{genre_text}{mood_text}"

def generate_music_from_text(text: str, genre: Optional[str] = None, 
                            mood: Optional[str] = None, duration: int = 30) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate music based on the provided text, genre, and mood.
    
    Args:
        text: Text to inspire the music generation
        genre: Optional music genre
        mood: Optional mood for the music
        duration: Desired duration in seconds (may not be respected by all backends)
        
    Returns:
        Tuple of (file path, error message)
    """
    # Create a unique ID for this music file
    music_id = str(uuid.uuid4())
    
    # Ensure proper cache directories exist
    os.makedirs("audio/music_cache", exist_ok=True)
    
    # Determine output path
    output_path = f"audio/music_cache/generated_{music_id}.mp3"
    
    # Generate a musical prompt from the text
    musical_prompt = convert_text_to_musical_prompt(text, genre, mood)
    
    # First attempt: Try using OpenAI for direct music generation with musicgen capability
    if openai_client:
        try:
            logger.info(f"Generating music with OpenAI Musicgen based on prompt: {musical_prompt[:50]}...")
            
            # Create a musicgen-optimized prompt
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": 
                     "You're a music production expert that creates detailed prompts for audio generation AI. "
                     "Create a specific, technical music generation prompt based on the user's request. "
                     "Include genre, instruments, tempo, mood, and structure information. Make it concise and effective for music AI."
                    },
                    {"role": "user", "content": musical_prompt}
                ]
            )
            
            refined_prompt = completion.choices[0].message.content
            
            # Use musicgen to create the audio (implemented via MusicGen capability in OpenAI)
            try:
                # Attempt to use MusicGen feature (requires token granting MusicGen access)
                musicgen_response = openai_client.audio.speech.create(
                    model="musicgen-1",  # Using model id for MusicGen
                    voice="music",  # Special voice parameter for music
                    input=refined_prompt,
                    response_format="mp3",
                    speed=1.0,
                    # Custom parameters for music generation:
                    duration=min(duration, 120),  # Cap at 2 minutes max
                    genre=genre if genre else None,
                    mood=mood if mood else None
                )
                
                # Save the generated music
                with open(output_path, "wb") as audio_file:
                    audio_file.write(musicgen_response.content)
                
                logger.info(f"Successfully generated music using OpenAI MusicGen: {output_path}")
                return output_path, None  # No error
            except Exception as musicgen_error:
                logger.warning(f"OpenAI MusicGen failed, trying alternative method: {str(musicgen_error)}")
                # Continue to alternative methods
        except Exception as e:
            logger.error(f"OpenAI music generation preparation error: {str(e)}")
    
    # Second attempt: Try using XAI (Grok) for music generation
    if xai_client:
        try:
            logger.info(f"Attempting to generate music with Grok Audio Generation based on: {musical_prompt[:50]}...")
            
            # Format the prompt for Grok
            messages = [
                {
                    "role": "system",
                    "content": "You are a music generation assistant. Create music based on text descriptions."
                },
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": f"Generate music with the following characteristics:\n\n{musical_prompt}\n\nDuration: {duration} seconds\nGenre: {genre or 'appropriate genre'}\nMood: {mood or 'suitable mood'}"
                        }
                    ]
                }
            ]
            
            # Request music generation from Grok
            response = xai_client.chat.completions.create(
                model="grok-2-audio-1215",  # Using Grok's audio generation model
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            
            # Process the response - extract audio content
            generated_audio = response.choices[0].message.content
            
            # The model may return a base64-encoded audio or a URL
            if "base64," in generated_audio:
                # Extract the base64 data and save as audio file
                try:
                    base64_data = generated_audio.split("base64,")[1].split("\"")[0]
                    audio_bytes = base64.b64decode(base64_data)
                    
                    with open(output_path, "wb") as audio_file:
                        audio_file.write(audio_bytes)
                    
                    logger.info(f"Successfully generated music using Grok: {output_path}")
                    return output_path, None  # No error
                except Exception as decode_error:
                    logger.error(f"Failed to decode Grok audio: {str(decode_error)}")
            
            # If we reach here, we couldn't extract audio from Grok's response
            logger.warning("Grok did not return usable audio data")
        except Exception as e:
            logger.error(f"Grok music generation error: {str(e)}")
    
    # Third attempt: Use Anthropic Claude for multi-modal generation
    if anthropic_client:
        try:
            logger.info(f"Attempting to generate music with Claude based on: {musical_prompt[:50]}...")
            
            # Format the prompt for Claude
            # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4000,
                temperature=0.7,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Generate music based on this description. Create something that matches the following characteristics:\n\n{musical_prompt}\n\nDuration: {duration} seconds\nGenre: {genre or 'appropriate to the content'}\nMood: {mood or 'fitting to the description'}\n\nRespond with both a description of the music and the actual audio file in a format I can use."
                            }
                        ]
                    }
                ]
            )
            
            # Check for audio in the response
            for content in response.content:
                if hasattr(content, 'source') and hasattr(content.source, 'media_type'):
                    if 'audio' in content.source.media_type:
                        # Extract and save the audio
                        audio_data = base64.b64decode(content.source.data)
                        with open(output_path, "wb") as audio_file:
                            audio_file.write(audio_data)
                        
                        logger.info(f"Successfully generated music using Claude: {output_path}")
                        return output_path, None  # No error
            
            logger.warning("Claude did not return audio data")
        except Exception as e:
            logger.error(f"Claude music generation error: {str(e)}")
    
    # Fourth attempt: Generate real music using enhanced_audio multi-track synthesis
    try:
        logger.info("Attempting to generate music using enhanced audio synthesis...")
        
        # Import here to avoid circular imports
        from enhanced_audio import generate_layered_audio, generate_audio
        
        # Try to use the multi-track generation feature
        try:
            audio_path = generate_layered_audio(
                text=musical_prompt,
                layers=3,  # Generate multiple layers for richer music
                duration=duration,
                genre=genre,
                mood=mood
            )
            
            if audio_path:
                logger.info(f"Successfully generated music using layered audio synthesis: {audio_path}")
                return audio_path, None
            
        except Exception as layered_error:
            logger.warning(f"Layered audio synthesis failed: {str(layered_error)}")
        
        # If multi-track generation fails, use simple TTS with musical tone
        logger.warning("Falling back to musical TTS...")
        if openai_client:
            try:
                # Use TTS with musical voice
                explanation = f"This is a musical interpretation of: {musical_prompt}"
                tts_response = openai_client.audio.speech.create(
                    model="tts-1-hd",
                    voice="alloy",  # Melodic voice
                    input=explanation,
                    speed=0.9,  # Slightly slower for musical feel
                )
                
                # Save the audio
                with open(output_path, "wb") as audio_file:
                    audio_file.write(tts_response.content)
                
                return output_path, "Music generation capability limited. Generated a musical interpretation instead."
                
            except Exception as tts_error:
                logger.error(f"TTS music fallback failed: {str(tts_error)}")
    except Exception as e:
        logger.error(f"Enhanced audio synthesis failed: {str(e)}")
    
    # Final fallback: generate basic audio explanation
    try:
        from enhanced_audio import generate_audio
        
        explanation = f"I tried to create music based on: {musical_prompt}. Unfortunately, music generation is currently limited. I'll improve this capability soon."
        audio_path = generate_audio(explanation, "en")
        
        if audio_path:
            return audio_path, "Full music generation is currently limited. Generated explanatory audio instead."
    except Exception as e:
        logger.error(f"Fallback audio generation failed: {str(e)}")
    
    return None, "Music generation is not currently available"

def get_music_generation_capabilities() -> Dict[str, bool]:
    """
    Return the current capabilities of music generation services.
    
    Returns:
        Dictionary with capability flags
    """
    return {
        "openai_available": openai_client is not None,
        "xai_available": xai_client is not None,
        "full_music_generation": False,  # Currently not fully implemented
        "concept_generation": openai_client is not None,
    }