# Enhanced Language Learning and Translation Bot - Setup Guide

This guide provides step-by-step instructions to set up and run your Enhanced Language Learning and Translation Bot, even if you're not a programmer. By following these instructions, you'll have a fully functional language learning platform with a Telegram bot interface and optional web application.

## Quick Start (Easiest Method)

1. **Open a terminal** in the bot's directory

2. **Run the installation script**:
   ```
   ./install.sh
   ```
   
   This script will:
   - Check for Python
   - Create a virtual environment (if possible)
   - Install all required packages
   - Ask for your Telegram token and set it up
   - Create necessary directories
   - Provide instructions for running the bot

3. **Start the bot**:
   After installation, you can start the bot by running:
   ```
   python main.py
   ```

## Manual Setup (Alternative Method)

If you prefer to do the setup manually or the install script doesn't work, follow these steps:

1. **Ensure Python 3.7+ is installed**:
   You can check by running:
   ```
   python --version
   ```
   If the version is below 3.7 or Python is not installed, download and install Python from [python.org](https://www.python.org/downloads/)

2. **Install required packages**:
   ```
   pip install python-telegram-bot deep-translator gtts pypinyin langdetect googletrans-py flask gunicorn
   ```

3. **Set up your Telegram token**:
   You need a Telegram bot token from [@BotFather](https://t.me/BotFather)
   
   Set it as an environment variable:
   ```
   # On Linux/Mac:
   export TELEGRAM_TOKEN="your_telegram_bot_token"
   
   # On Windows (Command Prompt):
   set TELEGRAM_TOKEN=your_telegram_bot_token
   
   # On Windows (PowerShell):
   $env:TELEGRAM_TOKEN="your_telegram_bot_token"
   ```

4. **Run the setup script**:
   ```
   python setup.py
   ```

5. **Start the bot**:
   ```
   python main.py
   ```

## Bot Features

### Supported Languages (16 languages)

The bot now supports these languages:
- 🇬🇧 English
- 🇪🇸 Spanish
- 🇫🇷 French
- 🇮🇹 Italian
- 🇵🇹 Portuguese
- 🇷🇺 Russian
- 🇨🇳 Chinese
- 🇩🇪 German
- 🇯🇵 Japanese
- 🇰🇷 Korean
- 🇸🇦 Arabic
- 🇮🇳 Hindi
- 🇹🇷 Turkish
- 🇳🇱 Dutch
- 🇵🇱 Polish
- 🇸🇪 Swedish

### Enhanced Learning Features

- 🔊 **Pronunciation Audio**: Listen to translations in any supported language
- 🎮 **Vocabulary Matching Game**: Test your vocabulary knowledge at different levels
- 📚 **Word of the Day**: Learn new words with definitions, examples, and translations
- 🔄 **Spaced Repetition System**: Efficiently memorize vocabulary
- 🍎 **Thematic Learning**: Learn vocabulary organized by themes (travel, food, work, etc.)
- 📊 **Learning Statistics**: Track your progress with detailed statistics
- 🏆 **Achievements**: Unlock achievements as you progress in your learning journey
- 🔔 **Customizable Notifications**: Set up reminders to practice regularly

## Troubleshooting

- **Problem**: Bot doesn't start
  **Solution**: Make sure your TELEGRAM_TOKEN is set correctly and your internet connection is working

- **Problem**: Missing module error
  **Solution**: Run `pip install <module_name>` for any missing modules

- **Problem**: Audio doesn't work
  **Solution**: Ensure you have gtts installed: `pip install gtts`

- **Problem**: Translation fails
  **Solution**: Check your internet connection and try again later

For more help, check the full documentation or submit an issue in the repository.

## Starting the Web Application (Optional)

If you want to use the web interface:

```
gunicorn --bind 0.0.0.0:5000 main:app
```

Then access it in your browser at `http://localhost:5000`

## Enjoy Learning!

Start a chat with your bot on Telegram and begin your language learning journey!