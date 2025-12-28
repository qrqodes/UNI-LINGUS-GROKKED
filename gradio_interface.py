"""
Gradio web interface for Enhanced Language Learning Bot with MCP (Model Control Protocol) support.
This allows the bot to be accessible through a web interface as well and implements
the MCP server for standardized AI model control.

References:
- MCP Documentation: https://huggingface.co/blog/gradio-mcp
"""

import os
import sys
import logging
try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Please install with: pip install gradio")
    sys.exit(1)
from typing import Dict, Any, Optional, List

# Local imports - may need to adjust these
from enhanced_translator import detect_language, translate_text
import ai_services_simplified as ai
import vocabulary_data
from enhanced_audio import generate_audio

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define language options
LANGUAGES = {
    "en": "English 🇬🇧",
    "es": "Spanish 🇪🇸",
    "pt": "Portuguese 🇵🇹",
    "it": "Italian 🇮🇹",
    "fr": "French 🇫🇷",
    "ru": "Russian 🇷🇺",
    "zh-CN": "Chinese 🇨🇳",
    "de": "German 🇩🇪",
    "ja": "Japanese 🇯🇵",
    "ko": "Korean 🇰🇷",
    "ar": "Arabic 🇸🇦",
    "hi": "Hindi 🇮🇳"
}

def translate_interface(text: str, target_languages: List[str]) -> Dict[str, Any]:
    """
    Handle translation requests from the Gradio interface.
    
    Args:
        text: Text to translate
        target_languages: List of language codes to translate to
        
    Returns:
        Dict with translations and detected language
    """
    if not text or not target_languages:
        return {"error": "Please enter text and select at least one target language."}
    
    try:
        # Detect input language
        source_lang = detect_language(text)
        if not source_lang:
            return {"error": "Could not detect language of input text."}
        
        # Get language name
        source_lang_name = LANGUAGES.get(source_lang, source_lang)
        
        # Translate to all selected languages
        translations = {}
        for lang in target_languages:
            if lang == source_lang:
                translations[lang] = text  # No need to translate if same language
            else:
                result = translate_text(text, source_lang=source_lang, target_lang=lang)
                if result:
                    translations[lang] = result
                else:
                    translations[lang] = f"Translation to {LANGUAGES.get(lang, lang)} failed."
        
        # Format results
        formatted_results = {
            "source_language": source_lang,
            "source_language_name": source_lang_name,
            "translations": translations
        }
        
        return formatted_results
    
    except Exception as e:
        logger.error(f"Error in translation: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}

def vocabulary_interface(text: str, level: str = "intermediate") -> Dict[str, Any]:
    """
    Extract vocabulary from text.
    
    Args:
        text: Text to analyze
        level: Learning level (beginner, intermediate, advanced)
        
    Returns:
        Dict with vocabulary information
    """
    if not text:
        return {"error": "Please enter text to analyze."}
    
    try:
        # Use detected language or default to English
        detected_lang = detect_language(text) or 'en'
        vocab_data = vocabulary_data.generate_thematic_vocabulary("general", detected_lang, level)
        return {"vocabulary": vocab_data}
    
    except Exception as e:
        logger.error(f"Error in vocabulary extraction: {str(e)}")
        return {"error": f"An error occurred: {str(e)}"}

def chat_with_ai_interface(text: str) -> str:
    """
    Chat with AI.
    
    Args:
        text: User message
        
    Returns:
        AI response
    """
    if not text:
        return "Please enter a message to chat with AI."
    
    try:
        response = ai.query_claude(text)
        if response:
            return response
        else:
            return "AI service is currently unavailable. Please try again later."
    
    except Exception as e:
        logger.error(f"Error in AI chat: {str(e)}")
        return f"An error occurred while communicating with AI: {str(e)}"

def setup_gradio_app():
    """Set up and return the Gradio application with MCP support."""
    
    # Define MCP models for translation, vocabulary, and chat
    # This follows the Model Control Protocol standard
    mcp_models = {
        "translation-model": {
            "display_name": "Translation Model",
            "description": "Multilingual translation with 12 supported languages",
            "tags": ["translation", "multilingual"],
            "input_schema": {
                "text": {"display_name": "Text", "type": "string", "description": "Text to translate"},
                "target_languages": {"display_name": "Target Languages", "type": "list", "description": "Languages to translate to"}
            },
            "output_schema": {
                "translations": {"display_name": "Translations", "type": "json", "description": "Translation results"}
            }
        },
        "vocabulary-model": {
            "display_name": "Vocabulary Extraction",
            "description": "Extract and analyze vocabulary from text",
            "tags": ["vocabulary", "learning", "education"],
            "input_schema": {
                "text": {"display_name": "Text", "type": "string", "description": "Text to analyze"},
                "level": {"display_name": "Level", "type": "string", "description": "Learning level"}
            },
            "output_schema": {
                "vocabulary": {"display_name": "Vocabulary", "type": "json", "description": "Extracted vocabulary items"}
            }
        },
        "ai-chat-model": {
            "display_name": "AI Language Assistant",
            "description": "Converse with an AI language assistant",
            "tags": ["chat", "ai", "assistant"],
            "input_schema": {
                "text": {"display_name": "Message", "type": "string", "description": "Your message to the AI"}
            },
            "output_schema": {
                "response": {"display_name": "Response", "type": "string", "description": "AI response"}
            }
        }
    }
    
    # Translation tab
    with gr.Blocks() as translation_tab:
        gr.Markdown("# Enhanced Language Learning Bot - Translation")
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Text to translate", 
                    lines=5, 
                    placeholder="Enter text here...",
                    elem_id="translation-model-text"  # MCP element ID
                )
                
                # Create checkboxes for each language
                lang_checkboxes = {}
                with gr.Row():
                    with gr.Column():
                        for i, (code, name) in enumerate(list(LANGUAGES.items())[:6]):
                            lang_checkboxes[code] = gr.Checkbox(label=name, value=(code == "en"))
                    
                    with gr.Column():
                        for i, (code, name) in enumerate(list(LANGUAGES.items())[6:]):
                            lang_checkboxes[code] = gr.Checkbox(label=name, value=(code == "en"))
                
                translate_btn = gr.Button(
                    "Translate",
                    elem_id="translation-model-submit"  # MCP element ID
                )
            
            with gr.Column():
                output_lang = gr.Textbox(label="Detected Language")
                output_translations = gr.JSON(
                    label="Translations",
                    elem_id="translation-model-translations"  # MCP element ID
                )
        
        # We need to handle the selected languages directly in the function
        def translate_with_selected_langs(text):
            selected_langs = [code for code, checkbox in lang_checkboxes.items() 
                             if checkbox.value]
            return translate_interface(text, selected_langs)
        
        translate_btn.click(
            fn=translate_with_selected_langs,
            inputs=[input_text],
            outputs=[output_translations],
            api_name="translation-model"  # MCP API endpoint
        )
    
    # Vocabulary tab
    with gr.Blocks() as vocab_tab:
        gr.Markdown("# Enhanced Language Learning Bot - Vocabulary")
        
        with gr.Row():
            with gr.Column():
                vocab_text = gr.Textbox(
                    label="Text to analyze", 
                    lines=5, 
                    placeholder="Enter text here...",
                    elem_id="vocabulary-model-text"  # MCP element ID
                )
                level_radio = gr.Radio(
                    choices=["beginner", "intermediate", "advanced"],
                    label="Learning Level",
                    value="intermediate",
                    elem_id="vocabulary-model-level"  # MCP element ID
                )
                vocab_btn = gr.Button(
                    "Extract Vocabulary",
                    elem_id="vocabulary-model-submit"  # MCP element ID
                )
            
            with gr.Column():
                vocab_output = gr.JSON(
                    label="Vocabulary",
                    elem_id="vocabulary-model-vocabulary"  # MCP element ID
                )
        
        vocab_btn.click(
            fn=vocabulary_interface,
            inputs=[vocab_text, level_radio],
            outputs=[vocab_output],
            api_name="vocabulary-model"  # MCP API endpoint
        )
    
    # AI Chat tab
    with gr.Blocks() as chat_tab:
        gr.Markdown("# Enhanced Language Learning Bot - AI Chat")
        
        with gr.Row():
            with gr.Column():
                chat_input = gr.Textbox(
                    label="Your message", 
                    lines=3, 
                    placeholder="Enter your message here...",
                    elem_id="ai-chat-model-text"  # MCP element ID
                )
                chat_btn = gr.Button(
                    "Send",
                    elem_id="ai-chat-model-submit"  # MCP element ID
                )
            
            with gr.Column():
                chat_output = gr.Textbox(
                    label="AI Response", 
                    lines=10,
                    elem_id="ai-chat-model-response"  # MCP element ID
                )
        
        chat_btn.click(
            fn=chat_with_ai_interface,
            inputs=[chat_input],
            outputs=[chat_output],
            api_name="ai-chat-model"  # MCP API endpoint
        )
    
    # Combine all tabs
    demo = gr.TabbedInterface(
        [translation_tab, vocab_tab, chat_tab],
        ["Translation", "Vocabulary", "AI Chat"]
    )
    
    # Set MCP metadata
    demo.mcp_config = {
        "metadata": {
            "name": "Enhanced Language Bot MCP",
            "version": "1.0.0",
            "description": "Enhanced Language Learning Bot with translation, vocabulary extraction, and AI chat",
            "tags": ["language-learning", "translation", "vocabulary", "chat"],
            "models": mcp_models
        }
    }
    
    return demo

if __name__ == "__main__":
    # Create and launch the Gradio interface with MCP support
    demo = setup_gradio_app()
    demo.launch(server_name="0.0.0.0", 
                server_port=7860, 
                share=True,
                mcp_server=True)  # Enable MCP server