# Installation Guide

This document provides detailed instructions for installing and setting up the Enhanced Language Learning Bot.

## Prerequisites

- Python 3.9+ installed
- PostgreSQL database
- Telegram bot token (from BotFather)
- API keys for AI services (Grok, Claude, and optionally OpenAI)

## Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/enhanced-language-bot.git
cd enhanced-language-bot
```

## Step 2: Set Up a Virtual Environment (Optional but Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Step 3: Install Dependencies

```bash
# Install required packages
pip install python-telegram-bot anthropic openai flask flask-wtf flask-sqlalchemy
pip install gunicorn googletrans-py gtts psycopg2-binary langdetect
pip install pocketsphinx SpeechRecognition python-dotenv deep-translator
pip install email-validator flask-limiter pypinyin requests amurex
```

## Step 4: Set Up Environment Variables

Create a `.env` file in the root directory with the following variables:

```
TELEGRAM_TOKEN=your_telegram_bot_token
XAI_API_KEY=your_grok_api_key
ANTHROPIC_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=your_postgresql_connection_string
```

## Step 5: Database Setup

1. Create a PostgreSQL database
2. Update the `DATABASE_URL` in your `.env` file
3. The application will automatically create necessary tables on first run

## Step 6: Run the Bot

```bash
# Run the Telegram bot
python main.py
```

## Step 7: Run the Web Interface (Optional)

```bash
# Start the web interface
gunicorn --bind 0.0.0.0:5000 main:app
```

## Additional Configuration Options

### Customizing Languages

Default languages are set in `enhanced_bot.py`. Modify the `DEFAULT_LANGUAGES` list to change the default languages.

### Speech Synthesis Settings

Audio generation settings can be configured in `enhanced_audio.py`.

### AI Service Priorities

By default, the bot prioritizes Grok, then Claude, then Google Translate. This can be adjusted in the `ai_services_simplified.py` file.

## Troubleshooting

### Database Connection Issues

If you encounter database connection issues, verify:
- Your PostgreSQL server is running
- The `DATABASE_URL` is correctly formatted
- Your firewall allows connections to the PostgreSQL port

### API Key Errors

If the bot cannot connect to AI services:
- Verify your API keys are valid and not expired
- Check that you've spelled the environment variable names correctly
- Ensure you have sufficient quota/credits with the API providers

### Audio Generation Problems

If audio generation fails:
- Check that the required directories (e.g., `audio/`) exist and are writable
- Verify internet connection as some TTS services require online access

## Deployment Options

See the following files for deployment-specific instructions:
- `HEROKU_DEPLOYMENT.md` - For deploying to Heroku
- `RENDER_DEPLOYMENT.md` - For deploying to Render
- `README_SETUP.md` - For general setup instructions