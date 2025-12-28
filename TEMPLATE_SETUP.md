# Getting Started with Your Telegram Language Bot

This document will guide you through setting up your new Language Bot after creating it from the template.

## 1. Set Up Your Environment Secrets

First, you need to add your API tokens as Replit Secrets:

1. Click on the **Tools** tab in the left sidebar
2. Select **Secrets**
3. Add the following secrets one by one:

- `TELEGRAM_TOKEN` - Your Telegram bot token from BotFather
- `XAI_API_KEY` - Your X.AI (Grok) API key (optional)
- `ANTHROPIC_API_KEY` - Your Anthropic Claude API key (optional)
- `OPENAI_API_KEY` - Your OpenAI API key (optional)

Note: You need at least ONE of the AI API keys for the bot to function.

## 2. Create Your Telegram Bot

If you don't already have a Telegram Bot token:

1. Open Telegram and search for "@BotFather"
2. Start a chat and send the command `/newbot`
3. Follow the instructions to name your bot and create a username
4. Copy the token provided by BotFather
5. Add it as the `TELEGRAM_TOKEN` secret in Replit

## 3. Start the Bot

1. Click the **Run** button at the top of the Replit interface
2. Wait for the bot to start up (this may take a minute)
3. Once you see "Bot started successfully" in the console, your bot is ready!

## 4. Test Your Bot

1. Open Telegram and start a chat with your new bot
2. Send the `/start` command
3. Try sending a message in any language to see it translated
4. Explore other commands like `/languages` to customize your experience

## 5. Ensure 24/7 Operation (Optional)

For your bot to run continuously, you need to:

1. Upgrade to Replit Pro
2. Enable the "Always On" feature for your repl

## 6. Next Steps

- Customize the language list in `enhanced_bot.py` if desired
- Modify the welcome message or keyboard layout
- Add new features or commands

Enjoy your new Language Learning Bot!