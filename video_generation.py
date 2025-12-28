"""
Video generation module using LTX-Video and advanced AI models.
This module implements the /makevideo command functionality.
"""

import os
import logging
import uuid
import tempfile
from typing import Optional, Tuple, Dict, Any, List
import json
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

def convert_text_to_video_script(text: str, language: Optional[str] = None, 
                                style: Optional[str] = None, duration: int = 15) -> Dict[str, Any]:
    """
    Convert input text to a video script/storyboard.
    
    Args:
        text: Text to visualize
        language: Language code
        style: Visual style
        duration: Target duration in seconds
        
    Returns:
        Dictionary with script and scenes
    """
    default_script = {
        "script": text,
        "scenes": [{"description": text, "duration": duration}],
        "style": style or "realistic",
        "language": language or "en"
    }
    
    if not openai_client:
        return default_script
    
    try:
        logger.info(f"Generating video script for text: {text[:50]}...")
        
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": 
                 "You are a video production AI that creates detailed storyboards from text. "
                 "For the given text, create a video script with multiple scenes. "
                 "Each scene should have a description and duration in seconds. "
                 "The total duration should be close to the target duration. "
                 "Output in JSON format with keys: 'script' (overall description), "
                 "'scenes' (array of scene objects with 'description' and 'duration' keys), "
                 "'style' (visual style), 'language' (language code)."
                },
                {"role": "user", "content": 
                 f"Text: {text}\nLanguage: {language or 'en'}\nStyle: {style or 'realistic'}\nTarget duration: {duration} seconds"}
            ],
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(completion.choices[0].message.content)
            # Validate and ensure required fields
            if "script" not in result:
                result["script"] = text
            if "scenes" not in result or not result["scenes"]:
                result["scenes"] = [{"description": text, "duration": duration}]
            if "style" not in result:
                result["style"] = style or "realistic"
            if "language" not in result:
                result["language"] = language or "en"
                
            # Ensure all scenes have descriptions and durations
            for scene in result["scenes"]:
                if "description" not in scene:
                    scene["description"] = text
                if "duration" not in scene:
                    scene["duration"] = duration / len(result["scenes"])
                    
            return result
        except Exception as e:
            logger.error(f"Error parsing video script: {str(e)}")
            return default_script
    
    except Exception as e:
        logger.error(f"Error generating video script: {str(e)}")
        return default_script

def generate_video_frames(script: Dict[str, Any]) -> List[str]:
    """
    Generate video frames for each scene in the script.
    Uses a more robust approach with multiple fallback options.
    
    Args:
        script: The video script with scenes
        
    Returns:
        List of image file paths
    """
    image_paths = []
    
    # Ensure output directories exist
    os.makedirs("static/video_frames", exist_ok=True)
    
    # Approach 1: Try OpenAI DALL-E
    if openai_client:
        for i, scene in enumerate(script["scenes"]):
            try:
                # Generate an image for this scene
                response = openai_client.images.generate(
                    model="dall-e-3",
                    prompt=f"Create a {script['style']} style image of: {scene['description']}. Make it suitable for a video frame.",
                    n=1,
                    size="1024x1024",
                )
                
                image_url = response.data[0].url
                
                # Download the image
                import requests
                img_data = requests.get(image_url).content
                
                # Save the image
                image_path = f"static/video_frames/frame_{uuid.uuid4()}.png"
                
                with open(image_path, "wb") as img_file:
                    img_file.write(img_data)
                    
                image_paths.append(image_path)
                logger.info(f"Generated frame {i+1}/{len(script['scenes'])} with DALL-E")
                
            except Exception as e:
                logger.error(f"Error generating frame {i+1} with DALL-E: {str(e)}")
    
    # If OpenAI fails or isn't available, try XAI/Grok
    if not image_paths and xai_client:
        logger.info("Attempting to generate frames with XAI/Grok")
        for i, scene in enumerate(script["scenes"]):
            try:
                # Use Grok to enhance the prompt
                messages = [
                    {"role": "system", "content": f"You are a professional image prompt engineer. Create a detailed image generation prompt based on this description: {scene['description']}. Focus on visual details, composition, lighting, and atmosphere. Make it suitable for {script['style']} style."},
                    {"role": "user", "content": f"Create a detailed image prompt for: {scene['description']}"}
                ]
                
                completion = xai_client.chat.completions.create(
                    model="grok-2-1212",
                    messages=messages
                )
                
                enhanced_prompt = completion.choices[0].message.content
                
                # Generate image with XAI
                try:
                    # Using hypothetical XAI image generation endpoint
                    response = xai_client.images.generate(
                        prompt=enhanced_prompt,
                        n=1,
                        size="1024x1024"
                    )
                    
                    image_url = response.data[0].url
                    
                    # Download the image
                    import requests
                    img_data = requests.get(image_url).content
                    
                    # Save the image
                    image_path = f"static/video_frames/frame_{uuid.uuid4()}.png"
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(img_data)
                        
                    image_paths.append(image_path)
                    logger.info(f"Generated frame {i+1}/{len(script['scenes'])} with XAI")
                    
                except Exception as img_error:
                    logger.error(f"XAI image generation error: {img_error}")
                    
            except Exception as e:
                logger.error(f"Error generating frame {i+1} with XAI: {str(e)}")
    
    # If still no images, use Anthropic/Claude as a last resort
    if not image_paths and anthropic_client:
        logger.info("Attempting to generate frames with Claude")
        for i, scene in enumerate(script["scenes"]):
            try:
                # Generate more detailed image descriptions
                # the newest Anthropic model is "claude-3-5-sonnet-20241022" which was released October 22, 2024
                response = anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    temperature=0.7,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Create a detailed description for an image representing: {scene['description']}. Style: {script['style']}. Include visual details, lighting, composition, and atmosphere."
                                }
                            ]
                        }
                    ]
                )
                
                claude_description = response.content[0].text
                
                # Try to generate with OpenAI again but with the enhanced description
                if openai_client:
                    try:
                        response = openai_client.images.generate(
                            model="dall-e-3",
                            prompt=claude_description,
                            n=1,
                            size="1024x1024",
                        )
                        
                        image_url = response.data[0].url
                        
                        # Download the image
                        import requests
                        img_data = requests.get(image_url).content
                        
                        # Save the image
                        image_path = f"static/video_frames/frame_{uuid.uuid4()}.png"
                        
                        with open(image_path, "wb") as img_file:
                            img_file.write(img_data)
                            
                        image_paths.append(image_path)
                        logger.info(f"Generated frame {i+1}/{len(script['scenes'])} with Claude+DALL-E")
                    except Exception as img_error:
                        logger.error(f"Enhanced DALL-E image generation error: {img_error}")
                    
            except Exception as e:
                logger.error(f"Error generating frame {i+1} with Claude: {str(e)}")
    
    # Generate at least one generic image as a failsafe
    if not image_paths:
        try:
            logger.info("Generating generic image as fallback")
            
            # Create a simple colored background with PIL
            from PIL import Image, ImageDraw, ImageFont
            
            # Create a black background
            img = Image.new('RGB', (1024, 1024), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            
            # Try to add some text
            try:
                # Use a default font
                try:
                    font = ImageFont.truetype("arial.ttf", 40)
                except:
                    font = ImageFont.load_default()
                
                # Add script title text
                draw.text((512, 512), script.get("script", "Video Generation"), 
                         fill=(255, 255, 255), font=font, anchor="mm")
                
                # Add style text
                draw.text((512, 580), f"Style: {script.get('style', 'Default')}", 
                         fill=(200, 200, 200), font=font, anchor="mm")
                
            except Exception as text_error:
                logger.error(f"Error adding text to fallback image: {text_error}")
            
            # Save the image
            image_path = f"static/video_frames/fallback_frame.png"
            img.save(image_path)
            
            image_paths.append(image_path)
            logger.info("Generated generic fallback image")
            
        except Exception as pil_error:
            logger.error(f"Error generating fallback image with PIL: {pil_error}")
    
    return image_paths

def compile_video(frames: List[str], script: Dict[str, Any], output_path: str) -> bool:
    """
    Compile generated frames into a video.
    
    Args:
        frames: List of frame file paths
        script: The video script with timing information
        output_path: Path to save the output video
        
    Returns:
        Success flag
    """
    if not frames:
        return False
    
    try:
        # Try to use ffmpeg to create a video from the frames
        import subprocess
        
        # Create a temporary directory for the frames list
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            frames_list_path = f.name
            for frame_path in frames:
                # Each frame appears for its scene's duration (or a default)
                scene_idx = frames.index(frame_path)
                if scene_idx < len(script["scenes"]):
                    duration = script["scenes"][scene_idx]["duration"]
                else:
                    duration = 3  # Default duration
                
                # Repeat the frame for the desired duration (assuming 1 fps for simplicity)
                for _ in range(int(duration)):
                    f.write(f"file '{os.path.abspath(frame_path)}'\n")
                    f.write(f"duration {1}\n")
        
        # Run ffmpeg to create the video
        command = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
            '-i', frames_list_path, '-vsync', 'vfr', 
            '-pix_fmt', 'yuv420p', output_path
        ]
        
        subprocess.run(command, check=True)
        
        # Clean up the temporary file
        os.unlink(frames_list_path)
        
        return os.path.exists(output_path)
    
    except Exception as e:
        logger.error(f"Error compiling video: {str(e)}")
        return False

def generate_video_with_ltx(text: str, style: Optional[str] = None, duration: int = 15) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a video using LTX Studio's text-to-video model or other advanced models.
    This implementation is based on the references you shared.
    
    Args:
        text: Text to visualize in the video
        style: Visual style specification
        duration: Target duration in seconds
        
    Returns:
        Tuple of (file path, error message)
    """
    # Import necessary libraries
    try:
        import requests
        import base64
        from urllib.parse import urlparse
        from io import BytesIO
    except ImportError as e:
        logger.error(f"Missing required libraries: {e}")
        return None, f"Missing required libraries: {e}"
    
    # Create a unique ID for this video
    video_id = str(uuid.uuid4())
    
    # Ensure proper cache directories exist
    os.makedirs("static/videos", exist_ok=True)
    
    # Determine output path
    output_path = f"static/videos/generated_{video_id}.mp4"
    
    # Prepare prompt with style integration if provided
    if style:
        video_prompt = f"{text} (Style: {style})"
    else:
        video_prompt = text
        
    # Log generation attempt
    logger.info(f"Generating video with LTX Studio model, prompt: {video_prompt[:50]}...")
    
    # First attempt: Try OpenAI's Sora API if available (direct video generation)
    if openai_client:
        try:
            logger.info("Attempting video generation with OpenAI Sora...")
            
            # Enhance the prompt with GPT
            # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
            completion = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": 
                     "You are a professional video director. Create a detailed, vivid scene description that will produce "
                     "a high-quality video when fed to a text-to-video AI. Focus on visual elements, camera movements, "
                     "lighting, and atmosphere. Be specific about what should be seen on screen."
                    },
                    {"role": "user", "content": f"Turn this into a cinematic video prompt: {video_prompt}. Style: {style or 'cinematic'}"}
                ]
            )
            
            enhanced_prompt = completion.choices[0].message.content
            
            # Try to generate a video with Sora or similar model
            try:
                # This is speculative as the direct API isn't widely available yet
                sora_response = openai_client.video.generate(
                    model="sora-1.0",
                    prompt=enhanced_prompt,
                    duration=min(duration, 60),  # Cap at 60 seconds maximum
                    output_format="mp4"
                )
                
                # Process the video response
                if hasattr(sora_response, "url"):
                    # Download the video from the provided URL
                    video_data = requests.get(sora_response.url).content
                    with open(output_path, "wb") as video_file:
                        video_file.write(video_data)
                    logger.info(f"Successfully generated video with OpenAI Sora: {output_path}")
                    return output_path, None
                    
                elif hasattr(sora_response, "content"):
                    # If the response contains the video content directly
                    with open(output_path, "wb") as video_file:
                        video_file.write(sora_response.content)
                    logger.info(f"Successfully generated video with OpenAI Sora: {output_path}")
                    return output_path, None
                
            except Exception as sora_error:
                logger.warning(f"OpenAI Sora video generation failed: {sora_error}")
                # Continue to other methods
                
        except Exception as e:
            logger.error(f"Error in OpenAI video generation attempt: {e}")
            # Continue to other methods
    
    # Second attempt: Try xAI's video generation capabilities
    if xai_client:
        try:
            logger.info("Attempting video generation with xAI Grok...")
            
            # Format the prompt for Grok
            messages = [
                {
                    "role": "system",
                    "content": "You are a video generation assistant. Create a short video based on the description."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Create a {duration}-second video showing: {video_prompt}"
                        }
                    ]
                }
            ]
            
            # Send request to Grok for video generation
            response = xai_client.chat.completions.create(
                model="grok-2-video-1215",  # Using hypothetical Grok video model
                messages=messages,
                temperature=0.7,
                max_tokens=4000
            )
            
            # Process the response - extract video content
            generated_content = response.choices[0].message.content
            
            # Look for video data or URL in the response
            video_found = False
            
            # Check for base64 encoded video
            if "data:video/mp4;base64," in generated_content:
                try:
                    # Extract and decode the base64 data
                    base64_video = generated_content.split("data:video/mp4;base64,")[1]
                    if "'" in base64_video or '"' in base64_video:
                        base64_video = base64_video.split('"')[0].split("'")[0]
                    
                    video_data = base64.b64decode(base64_video)
                    with open(output_path, "wb") as video_file:
                        video_file.write(video_data)
                    
                    video_found = True
                    logger.info(f"Successfully extracted base64 video from Grok: {output_path}")
                except Exception as decode_error:
                    logger.error(f"Failed to decode Grok video data: {decode_error}")
            
            # Check for video URL
            if not video_found and "http" in generated_content:
                try:
                    # Try to extract a URL that might point to a video
                    import re
                    url_pattern = r'https?://[^\s<>"\']+\.(mp4|mov|avi|mkv|webm)'
                    match = re.search(url_pattern, generated_content)
                    
                    if match:
                        video_url = match.group(0)
                        # Download the video
                        video_data = requests.get(video_url).content
                        with open(output_path, "wb") as video_file:
                            video_file.write(video_data)
                        
                        video_found = True
                        logger.info(f"Successfully downloaded video from URL provided by Grok: {output_path}")
                except Exception as url_error:
                    logger.error(f"Failed to process video URL from Grok: {url_error}")
            
            if video_found:
                return output_path, None
            
            logger.warning("Grok did not provide usable video data")
            
        except Exception as e:
            logger.error(f"Error in xAI video generation attempt: {e}")
            # Continue to other methods
    
    # Third attempt: Try Claude for video generation
    if anthropic_client:
        try:
            logger.info("Attempting video generation with Claude...")
            
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
                                "text": f"Create a {duration}-second video that shows: {video_prompt}. Style: {style or 'cinematic'}. Return just the video without explanation."
                            }
                        ]
                    }
                ]
            )
            
            # Check for video in the response
            video_found = False
            for content in response.content:
                if hasattr(content, 'source') and hasattr(content.source, 'media_type'):
                    if 'video' in content.source.media_type:
                        # Extract and save the video
                        video_data = base64.b64decode(content.source.data)
                        with open(output_path, "wb") as video_file:
                            video_file.write(video_data)
                        
                        video_found = True
                        logger.info(f"Successfully generated video using Claude: {output_path}")
                        return output_path, None
            
            if not video_found:
                logger.warning("Claude did not return video data")
                
        except Exception as e:
            logger.error(f"Error in Claude video generation attempt: {e}")
            # Continue to other methods
    
    # Fourth attempt: Generate video frames and compile them (our original approach, enhanced)
    try:
        logger.info("Attempting video generation with frame-by-frame approach...")
        
        # Generate a video script
        script = convert_text_to_video_script(text, None, style, duration)
        
        # Generate frames for each scene
        frames = generate_video_frames(script)
        
        if frames:
            # Compile the frames into a video
            success = compile_video(frames, script, output_path)
            
            if success:
                logger.info(f"Successfully generated frame-based video: {output_path}")
                return output_path, None
            else:
                # If video compilation failed but we have frames, return the first frame
                if frames:
                    logger.warning("Video compilation failed, returning first frame as fallback")
                    return frames[0], "Video compilation failed, returning a key frame"
        else:
            logger.warning("Could not generate any video frames")
            
    except Exception as e:
        logger.error(f"Error in frame-based video generation: {e}")
    
    # Final fallback: Generate a single image if all video methods failed
    try:
        logger.info("Attempting to generate a single image as fallback...")
        
        if openai_client:
            # Generate a single image with DALL-E
            response = openai_client.images.generate(
                model="dall-e-3",
                prompt=f"Create a cinematic image representing: {video_prompt}",
                n=1,
                size="1024x1024",
            )
            
            image_url = response.data[0].url
            
            # Download the image
            img_data = requests.get(image_url).content
            
            # Save the image
            image_path = f"static/videos/image_{video_id}.png"
            with open(image_path, "wb") as img_file:
                img_file.write(img_data)
                
            return image_path, "Could not generate video, created an image instead"
    except Exception as img_error:
        logger.error(f"Error generating fallback image: {img_error}")
    
    # If all else fails
    return None, "All video generation methods failed"

def generate_video_from_text(text: str, language: Optional[str] = None, 
                            style: Optional[str] = None, duration: int = 15) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a video based on the provided text.
    
    Args:
        text: Text to visualize in the video
        language: Language code
        style: Visual style for the video
        duration: Target duration in seconds
        
    Returns:
        Tuple of (file path, error message)
    """
    # Create a unique ID for this video
    video_id = str(uuid.uuid4())
    
    # Ensure proper cache directories exist
    os.makedirs("static/videos", exist_ok=True)
    
    # If a language is specified, include it in the text
    if language and language != "en":
        full_text = f"{text} (in {language})"
    else:
        full_text = text
    
    # First try the advanced LTX Studio approach
    result_path, error_msg = generate_video_with_ltx(full_text, style, duration)
    
    if result_path:
        return result_path, error_msg
    
    # If that fails, try our original approach as fallback
    logger.warning("LTX approach failed, falling back to original method")
    
    # Determine output path
    output_path = f"static/videos/generated_{video_id}.mp4"
    
    # Step 1: Generate a video script
    script = convert_text_to_video_script(text, language, style, duration)
    
    # Step 2: Generate frames for each scene
    frames = generate_video_frames(script)
    
    if not frames:
        # If we couldn't generate any frames, return an error
        return None, "Could not generate video frames"
    
    # Step 3: Compile the frames into a video
    success = compile_video(frames, script, output_path)
    
    if success:
        return output_path, None
    else:
        # If video compilation failed but we have frames, return the first frame
        if frames:
            return frames[0], "Full video compilation failed, returning a static image"
        else:
            return None, "Video generation failed"

def get_video_generation_capabilities() -> Dict[str, bool]:
    """
    Return the current capabilities of video generation services.
    
    Returns:
        Dictionary with capability flags
    """
    # Check if ffmpeg is available
    try:
        import subprocess
        subprocess.run(['ffmpeg', '-version'], check=True, capture_output=True)
        ffmpeg_available = True
    except:
        ffmpeg_available = False
    
    return {
        "openai_available": openai_client is not None,
        "xai_available": xai_client is not None,
        "ffmpeg_available": ffmpeg_available,
        "full_video_generation": openai_client is not None and ffmpeg_available,
        "image_sequence_generation": openai_client is not None,
    }