"""
Augmented Reality Translation Overlay Feature Module.
This module provides AR-like translation capabilities via Telegram.
"""

import os
import logging
from typing import Optional, Dict, Any, List, Tuple
import uuid
import tempfile
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io
import base64
import numpy as np
import cv2
import textwrap

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directory for temporary files
os.makedirs("static/ar_overlay", exist_ok=True)

def create_augmented_translation_image(image_path: str, 
                                       source_text: str, 
                                       translated_text: Dict[str, str], 
                                       source_regions: Optional[List[Tuple[int, int, int, int]]] = None) -> Optional[str]:
    """
    Create an augmented reality-like translation overlay on an image.
    
    Args:
        image_path: Path to the input image
        source_text: The original text being translated
        translated_text: Dictionary mapping language codes to translated text
        source_regions: Optional list of bounding boxes where text appears in the image
        
    Returns:
        Path to the generated image with AR overlay or None if failed
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Create a drawing context
        draw = ImageDraw.Draw(img)
        
        # Try to load a nice font, falling back to default if not available
        try:
            # Try to find a font that supports multiple languages
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
            font_small = ImageFont.truetype("DejaVuSans.ttf", 16)
            font_large = ImageFont.truetype("DejaVuSans.ttf", 24)
        except IOError:
            try:
                font = ImageFont.truetype("Arial.ttf", 20)
                font_small = ImageFont.truetype("Arial.ttf", 16)
                font_large = ImageFont.truetype("Arial.ttf", 24)
            except IOError:
                # Fall back to default
                font = ImageFont.load_default()
                font_small = font
                font_large = font
        
        # If we have source regions, highlight them
        if source_regions:
            for x, y, w, h in source_regions:
                # Create semi-transparent overlay for the text region
                draw.rectangle([x, y, x+w, y+h], outline=(255, 255, 0), width=2)
                
                # Add a semi-transparent fill
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([x, y, x+w, y+h], fill=(255, 255, 0, 80))
                
                if img.mode == 'RGBA':
                    img = Image.alpha_composite(img, overlay)
                else:
                    img = img.convert('RGBA')
                    img = Image.alpha_composite(img, overlay)
                    img = img.convert('RGB')
                
                draw = ImageDraw.Draw(img)  # Need to recreate draw after changing img
        else:
            # If no regions provided, make a best guess based on image analysis
            # Convert PIL image to OpenCV format for text detection
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Simple text detection approach
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours that might be text
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours that might be text regions
            possible_text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                # Basic heuristics for text-like regions
                if 0.1 < aspect_ratio < 15 and w > 20 and h > 10 and w * h > 200:
                    possible_text_regions.append((x, y, w, h))
            
            # Use up to 3 most promising regions
            possible_text_regions = sorted(possible_text_regions, key=lambda r: r[2] * r[3], reverse=True)[:3]
            
            for x, y, w, h in possible_text_regions:
                # Create semi-transparent overlay for the text region
                draw.rectangle([x, y, x+w, y+h], outline=(255, 215, 0), width=2)
                
                # Add a semi-transparent fill
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle([x, y, x+w, y+h], fill=(255, 215, 0, 60))
                
                if img.mode == 'RGBA':
                    img = Image.alpha_composite(img, overlay)
                else:
                    img = img.convert('RGBA')
                    img = Image.alpha_composite(img, overlay)
                    img = img.convert('RGB')
                
                draw = ImageDraw.Draw(img)  # Need to recreate draw after changing img
        
        # Add a semi-transparent overlay at the bottom for translations
        height, width = img.height, img.width
        
        # Calculate the height needed for translations
        num_translations = len(translated_text)
        overlay_height = 40 + (30 * num_translations)  # Base height + height per translation
        
        # Create bottom overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([0, height - overlay_height, width, height], 
                              fill=(0, 0, 0, 180))
        
        if img.mode == 'RGBA':
            img = Image.alpha_composite(img, overlay)
        else:
            img = img.convert('RGBA')
            img = Image.alpha_composite(img, overlay)
            img = img.convert('RGB')
        
        draw = ImageDraw.Draw(img)  # Need to recreate draw
        
        # Add source text
        y_position = height - overlay_height + 10
        draw.text((10, y_position), "Original: " + source_text[:50] + ("..." if len(source_text) > 50 else ""), 
                 fill=(255, 255, 255), font=font)
        
        # Add translations
        y_position += 30
        for lang_code, text in translated_text.items():
            # Get flag emoji and language name
            flag = get_flag_emoji(lang_code)
            lang_name = get_language_name(lang_code)
            
            # Format and draw the translation
            translation_text = f"{flag} {lang_name}: {text[:50]}" + ("..." if len(text) > 50 else "")
            draw.text((10, y_position), translation_text, fill=(255, 255, 255), font=font)
            y_position += 30
        
        # Add AR label in top-right corner
        draw.text((width - 120, 10), "AR TRANSLATION", fill=(0, 255, 255), font=font_large)
        
        # Add cyberpunk-style decoration
        draw.line([(width - 125, 10), (width - 125, 40), (width - 180, 40)], 
                 fill=(0, 255, 255), width=2)
        
        # Save the output image
        output_path = f"static/ar_overlay/ar_overlay_{uuid.uuid4()}.png"
        img.save(output_path)
        
        return output_path
    except Exception as e:
        logger.error(f"Error creating AR translation overlay: {str(e)}")
        return None

def extract_text_from_image(image_path: str) -> Optional[str]:
    """
    Attempt to extract text from an image using OCR.
    This is a placeholder for actual OCR integration.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Extracted text or None if extraction failed
    """
    try:
        # This is a placeholder - in a real implementation, this would use
        # tesseract OCR or a cloud OCR service like Google Vision
        
        # For now, return a mock result
        return "Text extracted from image would appear here"
    except Exception as e:
        logger.error(f"Error extracting text from image: {str(e)}")
        return None

def get_flag_emoji(lang_code: str) -> str:
    """
    Get flag emoji for a language code.
    
    Args:
        lang_code: ISO language code
        
    Returns:
        Flag emoji string
    """
    # Map common language codes to flag emojis
    flag_map = {
        'en': '🇺🇸',  # English
        'es': '🇪🇸',  # Spanish
        'fr': '🇫🇷',  # French
        'de': '🇩🇪',  # German
        'it': '🇮🇹',  # Italian
        'pt': '🇵🇹',  # Portuguese
        'ru': '🇷🇺',  # Russian
        'ja': '🇯🇵',  # Japanese
        'zh': '🇨🇳',  # Chinese
        'ko': '🇰🇷',  # Korean
        'ar': '🇸🇦',  # Arabic
        'hi': '🇮🇳',  # Hindi
        'bn': '🇧🇩',  # Bengali
        'nl': '🇳🇱',  # Dutch
        'tr': '🇹🇷',  # Turkish
        'pl': '🇵🇱',  # Polish
        'sv': '🇸🇪',  # Swedish
        'fi': '🇫🇮',  # Finnish
        'da': '🇩🇰',  # Danish
        'no': '🇳🇴',  # Norwegian
        'cs': '🇨🇿',  # Czech
        'hu': '🇭🇺',  # Hungarian
        'el': '🇬🇷',  # Greek
        'he': '🇮🇱',  # Hebrew
        'th': '🇹🇭',  # Thai
        'vi': '🇻🇳',  # Vietnamese
        'uk': '🇺🇦',  # Ukrainian
        'id': '🇮🇩',  # Indonesian
        'ms': '🇲🇾',  # Malay
        'fa': '🇮🇷',  # Persian
        'tl': '🇵🇭',  # Filipino/Tagalog
    }
    
    # Return the flag emoji or a question mark flag
    return flag_map.get(lang_code, '🏳️')

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

def detect_text_regions(image_path: str) -> List[Tuple[int, int, int, int]]:
    """
    Detect regions in an image that might contain text.
    
    Args:
        image_path: Path to the image
        
    Returns:
        List of bounding boxes (x, y, width, height)
    """
    try:
        # Open image with OpenCV
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours that might be text
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that might be text regions
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # Basic heuristics for text-like regions
            if 0.1 < aspect_ratio < 15 and w > 20 and h > 10 and w * h > 200:
                text_regions.append((x, y, w, h))
        
        return text_regions
    except Exception as e:
        logger.error(f"Error detecting text regions: {str(e)}")
        return []