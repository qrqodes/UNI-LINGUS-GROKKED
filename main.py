#!/usr/bin/env python
"""
Main entry point for the Enhanced Language Learning and Translation Bot.
This module initializes the web server and Telegram bot.

Two modes of operation:
1. When run directly (python main.py) - runs the Telegram bot
2. When imported by gunicorn (main:app) - provides the Flask app
"""
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Import Flask app for web interface
try:
    from app import app
except ImportError as e:
    logger.error(f"Error importing Flask app: {e}")
    app = None

def run_telegram_bot():
    """Start the enhanced Telegram bot with advanced features."""
    logger.info("Starting enhanced language learning and translation bot...")
    
    try:
        # Import and run the enhanced bot
        import enhanced_bot
        logger.info("Starting enhanced bot with transcription, database and AI fallback features")
        enhanced_bot.main()
    except Exception as e:
        logger.error(f"Error starting the bot: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point that decides which service to run."""
    # Check if we need to create cache directories
    try:
        from utils import ensure_cache_dirs
        ensure_cache_dirs()
    except ImportError:
        logger.warning("Utils module not found, skipping cache directory creation")
    
    # Run the Telegram bot when executed directly
    run_telegram_bot()

if __name__ == '__main__':
    main()
