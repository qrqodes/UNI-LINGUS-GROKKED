"""
Cyberpunk Progress Animation for Language Learning.
Provides animated visual feedback for learning progress in a cyberpunk aesthetic.
"""

import os
import logging
import time
import json
import random
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import math
import io

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directory for animation frames
os.makedirs("static/animations", exist_ok=True)

# Cyberpunk color palettes
CYBERPUNK_PALETTES = {
    "neon": [
        (255, 0, 128),    # Neon pink
        (0, 255, 255),    # Cyan
        (255, 255, 0),    # Yellow
        (128, 0, 255),    # Purple
        (0, 255, 128)     # Neon green
    ],
    "hacker": [
        (0, 255, 0),      # Matrix green
        (32, 255, 32),    # Bright green
        (0, 128, 0),      # Dark green
        (192, 255, 192),  # Light green
        (0, 64, 0)        # Very dark green
    ],
    "future": [
        (0, 128, 255),    # Blue
        (255, 64, 64),    # Red
        (64, 0, 128),     # Dark purple
        (0, 196, 255),    # Light blue
        (255, 0, 64)      # Magenta
    ],
    "synthwave": [
        (255, 0, 128),    # Neon pink
        (0, 0, 255),      # Deep blue
        (255, 0, 255),    # Magenta
        (128, 0, 255),    # Purple
        (255, 128, 0)     # Orange
    ]
}

def generate_cyberpunk_text_animation(text: str, frames: int = 5) -> List[str]:
    """
    Generate cyberpunk-styled text animation frames.
    
    Args:
        text: The text to animate
        frames: Number of animation frames to generate
        
    Returns:
        List of animated text frames as strings
    """
    animation_frames = []
    
    # Cyberpunk text decorations
    decorations = [
        ":: {} ::",               # Simple brackets
        "> {} <",                 # Arrows
        "[ {} ]",                 # Square brackets
        "⟪ {} ⟫",                # Double angle brackets
        "⟦ {} ⟧",                # Double square brackets
        "⟨ {} ⟩",                # Angle brackets
        "《 {} 》",              # Double angle brackets (CJK)
        "【 {} 】",              # CJK brackets
        "⎡ {} ⎤",                # Upper-left and upper-right bracket
        "⧫ {} ⧫"                  # Diamond
    ]
    
    # Cyberpunk text prefixes/suffixes
    prefixes = [
        "SYS://",
        "CORE>",
        "NET::",
        "EXEC:",
        "AI://",
        "VR::",
        "CYBER:",
        "NEURO::",
        "SYN://",
        "DATA::"
    ]
    
    # Create varied animation frames
    for i in range(frames):
        # Choose decoration and prefix
        decoration = random.choice(decorations)
        prefix = random.choice(prefixes) if random.random() > 0.5 else ""
        
        # Create base frame
        frame = decoration.format(text)
        
        # Add prefix sometimes
        if prefix:
            frame = f"{prefix} {frame}"
        
        # Add glitch effects randomly
        if random.random() > 0.7:
            # Randomly replace a character with a glitchy symbol
            glitch_chars = "!@#$%^&*<>[]{}|;:/"
            if len(text) > 5:
                pos = random.randint(0, len(text) - 1)
                glitched_text = text[:pos] + random.choice(glitch_chars) + text[pos+1:]
                frame = decoration.format(glitched_text)
                if prefix:
                    frame = f"{prefix} {frame}"
        
        animation_frames.append(frame)
    
    return animation_frames

def generate_cyberpunk_progress_bar(progress: float, style: str = "neon") -> List[str]:
    """
    Generate cyberpunk-styled progress bar frames.
    
    Args:
        progress: Progress value between 0 and 1
        style: Style name (neon, hacker, future, synthwave)
        
    Returns:
        List of progress bar frames as strings
    """
    # Progress bar length
    bar_length = 20
    filled_length = int(bar_length * progress)
    
    # Create base animation frames
    animation_frames = []
    
    # Progress bar styles
    styles = {
        "neon": {
            "filled": "█",
            "empty": "▒",
            "left": "[",
            "right": "]",
            "prefix": "PROGRESS::"
        },
        "hacker": {
            "filled": "■",
            "empty": "□",
            "left": "<",
            "right": ">",
            "prefix": "SYS://LOAD:"
        },
        "future": {
            "filled": "▓",
            "empty": "░",
            "left": "《",
            "right": "》",
            "prefix": "NEURAL//"
        },
        "synthwave": {
            "filled": "◆",
            "empty": "◇",
            "left": "【",
            "right": "】",
            "prefix": "DATA::"
        }
    }
    
    # Use requested style or default to neon
    bar_style = styles.get(style, styles["neon"])
    
    # Create multiple animation frames with slight variations
    for i in range(5):  # Number of different animation frames
        # Simulate animation of the progress bar's active cell
        if i == 0:
            # Normal state
            filled = bar_style["filled"] * filled_length
            empty = bar_style["empty"] * (bar_length - filled_length)
        elif i == 1 and filled_length < bar_length:
            # Pulse at the edge of progress
            filled = bar_style["filled"] * (filled_length)
            pulse = "▣"  # Special character for pulse
            empty = bar_style["empty"] * (bar_length - filled_length - (1 if filled_length < bar_length else 0))
            if filled_length < bar_length:
                filled = filled + pulse
        else:
            # Other variations
            filled = bar_style["filled"] * filled_length
            empty = bar_style["empty"] * (bar_length - filled_length)
            
        # Assemble the frame
        bar = f"{bar_style['left']}{filled}{empty}{bar_style['right']} {int(progress * 100)}%"
        
        # Add cyberpunk flair with different messages based on progress
        if progress < 0.3:
            prefix = f"{bar_style['prefix']} INITIALIZING"
        elif progress < 0.6:
            prefix = f"{bar_style['prefix']} PROCESSING"
        elif progress < 0.9:
            prefix = f"{bar_style['prefix']} ACCELERATING"
        else:
            prefix = f"{bar_style['prefix']} FINALIZING"
        
        frame = f"{prefix} {bar}"
        
        animation_frames.append(frame)
    
    return animation_frames

def generate_language_level_animation(level: str, experience_points: int, next_level_points: int) -> List[str]:
    """
    Generate a cyberpunk animation showing language proficiency level.
    
    Args:
        level: Current language level (e.g., "A1", "B2")
        experience_points: Current experience points
        next_level_points: Points needed for next level
        
    Returns:
        List of animation frames as strings
    """
    # Calculate progress to next level
    progress = min(1.0, experience_points / next_level_points)
    
    # Create animation frames
    animation_frames = []
    
    # CEFR level descriptions
    cefr_descriptions = {
        "A1": "Beginner",
        "A2": "Elementary",
        "B1": "Intermediate",
        "B2": "Upper Intermediate",
        "C1": "Advanced",
        "C2": "Proficient"
    }
    
    # Get level description
    level_desc = cefr_descriptions.get(level, "Custom Level")
    
    # Create base frame with progress bar
    bar_length = 15
    filled_length = int(bar_length * progress)
    
    # Create multiple animation frames with cyberpunk aesthetic
    for i in range(5):
        # Create progress bar with different styles
        if i == 0:
            bar = f"[{'█' * filled_length}{'▒' * (bar_length - filled_length)}]"
        elif i == 1:
            bar = f"<{'=' * filled_length}{' ' * (bar_length - filled_length)}>"
        elif i == 2:
            bar = f"《{'■' * filled_length}{'□' * (bar_length - filled_length)}》"
        elif i == 3:
            bar = f"({filled_length * '|'}{'·' * (bar_length - filled_length)})"
        else:
            bar = f"[{'▓' * filled_length}{'░' * (bar_length - filled_length)}]"
        
        # Create frame with cyberpunk aesthetics
        frame = f"LANG/LEVEL: {level} - {level_desc}\n"
        frame += f"XP: {experience_points}/{next_level_points}\n"
        frame += f"PROGRESS: {bar} {int(progress * 100)}%\n"
        
        # Add cyberpunk flair
        if progress < 0.3:
            frame += "STATUS:: DEVELOPING NEURAL PATHWAYS"
        elif progress < 0.6:
            frame += "STATUS:: LINGUISTIC MATRIX EXPANDING"
        elif progress < 0.9:
            frame += "STATUS:: LANGUAGE CORTEX OPTIMIZING"
        else:
            frame += "STATUS:: BREAKTHROUGH IMMINENT"
        
        animation_frames.append(frame)
    
    return animation_frames

def generate_achievement_unlock_animation(achievement_name: str, reward: str = None) -> List[str]:
    """
    Generate a cyberpunk animation for achievement unlocking.
    
    Args:
        achievement_name: Name of the achievement
        reward: Optional reward description
        
    Returns:
        List of animation frames as strings
    """
    animation_frames = []
    
    # Frame 1: Initial notification
    frame = f">> ACHIEVEMENT DETECTED <<\n"
    frame += f">> SCANNING... <<\n"
    animation_frames.append(frame)
    
    # Frame 2: Processing
    frame = f">> ACHIEVEMENT DETECTED <<\n"
    frame += f">> ANALYZING... <<\n"
    frame += f">> {achievement_name[:5]}... <<\n"
    animation_frames.append(frame)
    
    # Frame 3: More processing
    frame = f">> ACHIEVEMENT DETECTED <<\n"
    frame += f">> PROCESSING... <<\n"
    frame += f">> {achievement_name[:10]}... <<\n"
    animation_frames.append(frame)
    
    # Frame 4: Almost done
    frame = f">> ACHIEVEMENT VERIFIED <<\n"
    frame += f">> FINALIZING... <<\n"
    frame += f">> {achievement_name} <<\n"
    animation_frames.append(frame)
    
    # Frame 5: Complete
    frame = f">> ACHIEVEMENT UNLOCKED <<\n"
    frame += f">> {achievement_name} <<\n"
    if reward:
        frame += f">> REWARD: {reward} <<\n"
    animation_frames.append(frame)
    
    # Frame 6: Flashy celebration
    frame = f">> ✨ ACHIEVEMENT UNLOCKED ✨ <<\n"
    frame += f">> ✨ {achievement_name} ✨ <<\n"
    if reward:
        frame += f">> REWARD: {reward} <<\n"
    animation_frames.append(frame)
    
    return animation_frames

def generate_cyberpunk_image_animation(
    text: str,
    progress: float = None,
    level: str = None,
    achievement: str = None,
    style: str = "neon"
) -> Optional[str]:
    """
    Generate a cyberpunk-styled animated GIF with text, progress, and other information.
    
    Args:
        text: Main text to display
        progress: Optional progress value (0-1)
        level: Optional level indication (e.g., "A1", "B2")
        achievement: Optional achievement name
        style: Visual style (neon, hacker, future, synthwave)
        
    Returns:
        Path to the generated animated GIF or None if generation failed
    """
    try:
        # Select color palette
        palette = CYBERPUNK_PALETTES.get(style, CYBERPUNK_PALETTES["neon"])
        primary_color = palette[0]
        secondary_color = palette[1]
        accent_color = palette[2]
        
        # Create a list to store frames
        frames = []
        
        # Parameters
        width, height = 800, 400
        bg_color = (10, 10, 20)
        font_path = None  # Will use default if not found
        
        # Try to find a suitable font
        try:
            # Try to find a font that looks tech/cyber
            for font_name in ["Courier", "Courier New", "Consolas", "Liberation Mono"]:
                try:
                    font_path = ImageFont.truetype(font_name, 24)
                    break
                except IOError:
                    continue
        except:
            pass
        
        # Fall back to default if no font found
        if not font_path:
            font_path = ImageFont.load_default()
        
        # Create 5 animation frames
        for frame_idx in range(5):
            # Create base image
            img = Image.new('RGB', (width, height), bg_color)
            draw = ImageDraw.Draw(img)
            
            # Draw a grid pattern in the background
            grid_spacing = 50
            grid_color = (30, 30, 50)
            
            # Horizontal grid lines
            for y in range(0, height, grid_spacing):
                draw.line([(0, y), (width, y)], fill=grid_color, width=1)
            
            # Vertical grid lines
            for x in range(0, width, grid_spacing):
                draw.line([(x, 0), (x, height)], fill=grid_color, width=1)
            
            # Add random circuit-like lines
            for _ in range(10):
                start_x = random.randint(0, width)
                start_y = random.randint(0, height)
                
                # Create zigzag lines
                points = [(start_x, start_y)]
                current_x, current_y = start_x, start_y
                
                for _ in range(random.randint(3, 7)):
                    # Decide direction (horizontal or vertical)
                    if random.random() > 0.5:
                        # Horizontal movement
                        current_x += random.randint(-100, 100)
                        current_x = max(0, min(width, current_x))
                    else:
                        # Vertical movement
                        current_y += random.randint(-100, 100)
                        current_y = max(0, min(height, current_y))
                    
                    points.append((current_x, current_y))
                
                # Draw the circuit line
                line_color = palette[random.randint(0, len(palette) - 1)]
                draw.line(points, fill=line_color, width=2)
            
            # Add header
            header_text = "CYBER.LANG::SYSTEM" if not achievement else "ACHIEVEMENT.SYSTEM"
            header_font = ImageFont.truetype(font_path.path, 30) if hasattr(font_path, 'path') else font_path
            header_width = draw.textlength(header_text, font=header_font)
            draw.text(((width - header_width) / 2, 20), header_text, fill=primary_color, font=header_font)
            
            # Draw a line under the header
            draw.line([(50, 60), (width - 50, 60)], fill=primary_color, width=2)
            
            # Add main text
            main_text = text
            main_font = ImageFont.truetype(font_path.path, 24) if hasattr(font_path, 'path') else font_path
            
            # Wrap text to fit width
            words = main_text.split()
            lines = []
            current_line = []
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                test_width = draw.textlength(test_line, font=main_font)
                
                if test_width <= (width - 100):
                    current_line.append(word)
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw wrapped text
            y_position = 100
            for line in lines:
                line_width = draw.textlength(line, font=main_font)
                draw.text(((width - line_width) / 2, y_position), line, fill=secondary_color, font=main_font)
                y_position += 30
            
            # Add progress bar if specified
            if progress is not None:
                y_position += 20
                
                # Draw progress bar label
                label = "PROGRESS STATUS:"
                label_width = draw.textlength(label, font=main_font)
                draw.text(((width - label_width) / 2, y_position), label, fill=accent_color, font=main_font)
                y_position += 30
                
                # Draw progress bar
                bar_width = 600
                bar_height = 30
                bar_x = (width - bar_width) / 2
                bar_y = y_position
                
                # Draw outer rectangle
                draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], 
                              outline=secondary_color, width=2)
                
                # Draw filled portion
                filled_width = int(bar_width * progress)
                draw.rectangle([(bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height)], 
                              fill=secondary_color)
                
                # Add percentage text
                percent_text = f"{int(progress * 100)}%"
                percent_font = ImageFont.truetype(font_path.path, 18) if hasattr(font_path, 'path') else font_path
                percent_width = draw.textlength(percent_text, font=percent_font)
                draw.text((bar_x + (bar_width - percent_width) / 2, bar_y + 5), 
                         percent_text, fill=bg_color, font=percent_font)
                
                y_position += bar_height + 20
            
            # Add level information if specified
            if level:
                level_text = f"LANGUAGE LEVEL: {level}"
                level_width = draw.textlength(level_text, font=main_font)
                draw.text(((width - level_width) / 2, y_position), level_text, fill=accent_color, font=main_font)
                y_position += 30
            
            # Add achievement information if specified
            if achievement:
                # Make achievement text blink in some frames
                if frame_idx % 2 == 0:
                    # Draw a highlight box
                    box_padding = 20
                    achievement_width = draw.textlength(achievement, font=main_font)
                    box_x1 = (width - achievement_width) / 2 - box_padding
                    box_y1 = y_position - 5
                    box_x2 = (width + achievement_width) / 2 + box_padding
                    box_y2 = y_position + 35
                    
                    # Draw glowing box
                    draw.rectangle([(box_x1, box_y1), (box_x2, box_y2)], 
                                  outline=primary_color, width=3)
                    
                # Achievement text with different color in different frames
                text_color = palette[frame_idx % len(palette)]
                achievement_width = draw.textlength(achievement, font=main_font)
                draw.text(((width - achievement_width) / 2, y_position), 
                         achievement, fill=text_color, font=main_font)
                y_position += 30
            
            # Add animation-specific elements based on frame index
            if frame_idx == 0:
                # Add corner brackets
                draw.text((20, 20), ">", fill=accent_color, font=main_font)
                draw.text((width - 30, 20), "<", fill=accent_color, font=main_font)
                draw.text((20, height - 30), ">", fill=accent_color, font=main_font)
                draw.text((width - 30, height - 30), "<", fill=accent_color, font=main_font)
            elif frame_idx == 1:
                # Add scanning lines
                for y in range(0, height, 100):
                    draw.line([(0, y + frame_idx * 10), (width, y + frame_idx * 10)], 
                             fill=accent_color, width=1)
            elif frame_idx == 2:
                # Add corner decorations
                draw.line([(10, 10), (50, 10), (50, 50), (10, 50), (10, 10)], 
                         fill=primary_color, width=2)
                draw.line([(width - 10, 10), (width - 50, 10), (width - 50, 50), (width - 10, 50), (width - 10, 10)], 
                         fill=primary_color, width=2)
                draw.line([(10, height - 10), (50, height - 10), (50, height - 50), (10, height - 50), (10, height - 10)], 
                         fill=primary_color, width=2)
                draw.line([(width - 10, height - 10), (width - 50, height - 10), (width - 50, height - 50), (width - 10, height - 50), (width - 10, height - 10)], 
                         fill=primary_color, width=2)
            elif frame_idx == 3:
                # Add random data-looking text in the corners
                data_font = ImageFont.truetype(font_path.path, 12) if hasattr(font_path, 'path') else font_path
                data_text = "SYS://DATA.1337.XF"
                draw.text((20, height - 20), data_text, fill=secondary_color, font=data_font)
                draw.text((width - 150, height - 20), f"ID::{random.randint(10000, 99999)}", 
                         fill=secondary_color, font=data_font)
            else:
                # Add binary-looking decoration at the bottom
                binary = "".join([random.choice(["0", "1"]) for _ in range(30)])
                binary_font = ImageFont.truetype(font_path.path, 14) if hasattr(font_path, 'path') else font_path
                binary_width = draw.textlength(binary, font=binary_font)
                draw.text(((width - binary_width) / 2, height - 30), binary, fill=secondary_color, font=binary_font)
            
            # Add the frame to our list
            frames.append(img)
        
        # Save as animated GIF
        output_path = f"static/animations/cyberpunk_{uuid.uuid4()}.gif"
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=200,  # 200ms per frame
            loop=0  # Loop forever
        )
        
        return output_path
    except Exception as e:
        logger.error(f"Error generating cyberpunk image animation: {str(e)}")
        return None