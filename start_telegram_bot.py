#!/usr/bin/env python3
"""
Telegram Bot Startup Script for Replit Deployment
This script ensures the Telegram bot runs continuously on Replit servers.
"""

import os
import sys
import time
import logging
import asyncio
from threading import Thread

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('telegram_bot.log')
    ]
)

logger = logging.getLogger(__name__)

def keep_alive():
    """Keep the bot alive by running a simple web server"""
    from flask import Flask
    alive_app = Flask(__name__)
    
    @alive_app.route('/')
    def home():
        return "🤖 UNI LINGUS Telegram Bot is ONLINE! 🌍"
    
    @alive_app.route('/health')
    def health():
        return {"status": "healthy", "service": "telegram_bot"}
    
    # Run on a different port than the main web app
    alive_app.run(host='0.0.0.0', port=8080, debug=False)

def start_telegram_bot():
    """Start the enhanced Telegram bot"""
    try:
        logger.info("🚀 Starting UNI LINGUS Telegram Bot...")
        
        # Import and run the enhanced bot
        from enhanced_bot import main
        
        # Run the bot
        main()
        
    except KeyboardInterrupt:
        logger.info("🛑 Telegram bot stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting Telegram bot: {e}")
        # Restart after error
        time.sleep(5)
        start_telegram_bot()

def main():
    """Main function to start both keep-alive server and Telegram bot"""
    logger.info("🎯 UNI LINGUS Telegram Bot - Replit Deployment")
    logger.info("🌟 Starting always-on Telegram bot service...")
    
    # Start keep-alive server in a separate thread
    keep_alive_thread = Thread(target=keep_alive, daemon=True)
    keep_alive_thread.start()
    
    # Small delay to let the keep-alive server start
    time.sleep(2)
    
    # Start the Telegram bot (this will run continuously)
    start_telegram_bot()

if __name__ == "__main__":
    main()