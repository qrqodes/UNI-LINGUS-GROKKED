# Enhanced Language Learning Bot

An advanced Telegram bot delivering intelligent multilingual translation services with interactive language learning capabilities. The bot transforms cross-language communication into an educational and engaging experience through state-of-the-art AI technologies.

## Features

- **Automatic Language Detection**: Detects the language of input text without requiring commands
- **Multi-Language Translation**: Translates to multiple languages simultaneously
- **Audio Pronunciation**: Generates audio pronunciation for translations
- **AI-Powered Chat**: Direct interaction with AI for language learning assistance
- **Transliteration**: Provides Latin transcription for non-Latin scripts (including pinyin for Chinese)
- **Vocabulary Extraction**: Extract vocabulary with definitions, examples, and contextual usage
- **Voice Message Support**: Transcribes and translates voice messages
- **Language Learning Tools**: Interactive vocabulary building and learning games

## Supported Languages

The bot focuses on these core languages:
- English
- Spanish
- Portuguese
- Italian
- French
- Russian
- Chinese
- German
- Japanese
- Korean
- Arabic
- Hindi

## Technical Stack

- **Python Telegram Bot Framework**: Core bot functionality
- **Grok AI**: Primary AI model for translations and language detection
- **Claude AI**: Fallback AI model for when Grok is unavailable
- **Google Translate**: Additional fallback for translations
- **Text-to-Speech**: Multiple providers with 5-tier fallback system
- **PostgreSQL**: Database for user preferences and vocabulary
- **Flask**: Web interface for browser-based access

## Setup Requirements

### Step 1: Set Up Your Telegram Bot

1. Open Telegram and search for [@BotFather](https://t.me/BotFather)
2. Start a chat with BotFather and send the command `/newbot`
3. Follow the prompts to create your bot:
   - Provide a name for your bot (e.g., "My Language Assistant")
   - Provide a username ending with "bot" (e.g., "my_language_assistant_bot")
4. BotFather will generate a token that looks like `123456789:ABCDefGhIJKlmNoPQRsTUVwxyZ`
5. Copy this token for use in your environment variables

### Step 2: Get Required API Keys

You need at least ONE of these AI service API keys:

#### Option A: X.AI (Grok) API Key
1. Go to [x.ai](https://x.ai) and create an account
2. Navigate to your API settings or developer dashboard
3. Generate a new API key and copy it

#### Option B: Anthropic (Claude) API Key
1. Go to [Anthropic Console](https://console.anthropic.com/) and create an account
2. Navigate to the API Keys section
3. Create a new API key and copy it

#### Option C: OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys) and create an account
2. Navigate to the API Keys section
3. Create a new API key and copy it

### Step 3: Environment Variables

Duplicate the `.env.example` file and rename it to `.env`, then fill in your keys:

```
TELEGRAM_TOKEN=your_telegram_bot_token
XAI_API_KEY=your_grok_api_key
ANTHROPIC_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

> **Note for Replit Users:** The database URL is automatically configured on Replit.

### Step 4: Python Dependencies

On Replit, dependencies are automatically installed from the pyproject.toml file.

For local development, install required packages:
```bash
pip install -r requirements.txt
```

## Running the Bot

### On Replit

1. Make sure all environment variables are set in the Secrets tab
2. Click the Run button at the top of the Replit interface
3. The bot will start automatically and begin responding to messages in Telegram

### Running Locally

To run the main Telegram bot:
```bash
python main.py
```

To run the web interface:
```bash
gunicorn --bind 0.0.0.0:5000 main:app
```

### 24/7 Hosting Options

For 24/7 operation, you have several options:

1. **Replit Pro**: Use Replit's Always On feature to keep your bot running continuously
2. **Heroku**: Deploy using the provided Procfile (see HEROKU_DEPLOYMENT.md for details)
3. **Render**: Deploy using the provided render.yaml configuration (see RENDER_DEPLOYMENT.md for details)

Once running, send `/start` to your bot in Telegram to begin using it!

## Bot Commands

- `/start` - Start the bot
- `/languages` - Select languages to translate to
- `/audio` - Generate speech from last message
- `/askai` - Extract vocabulary from your last message
- `/chat` - Chat directly with AI
- `/translator` - Switch back to translation mode
- `/flags` - Toggle language flags and names in translations
- `/stats` - View your usage statistics

## Usage Tips

1. **Translation Mode**: Simply send any text to get translations (default mode)
2. **Chat Mode**: Use `/chat` to talk directly with AI for language learning help
3. **Vocabulary Extraction**: Use `/askai` to extract vocabulary from text
4. **Audio Pronunciation**: Click the "Hear Audio" button after translations
5. **Language Selection**: Use `/languages` to choose which languages to translate to

## Deployment

This bot can be deployed to:
- Replit: For 24/7 operation via Replit Pro
- Heroku: Using the provided Procfile
- Render: Using the provided render.yaml configuration

## Fallback Systems

The bot uses a multi-tier fallback system:
1. Primary: Grok AI for translations and language detection
2. Secondary: Claude AI if Grok is unavailable
3. Tertiary: Google Translate for translations
4. Audio: 5-tier fallback system (OpenAI TTS → Grok → gTTS → Manual Synthesis → Text)

## Database Schema

- **users**: User preferences, selected languages
- **vocabulary**: User vocabulary items for learning
- **translations**: Translation history
- **statistics**: User usage statistics
- **achievements**: User achievements and streaks

## License

This project is licensed under the MIT License - see the LICENSE file for details.