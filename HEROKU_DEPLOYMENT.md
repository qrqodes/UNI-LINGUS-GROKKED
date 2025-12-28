# Heroku Deployment Guide for Your Telegram Bot

This guide will help you deploy your Telegram translation bot to Heroku for 24/7 operation.

## Prerequisites

- [Heroku account](https://signup.heroku.com/) (which you already have)
- [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
- Git installed on your computer

## Step 1: Prepare Your Environment

1. Install the Heroku CLI on your computer if you haven't already:
   - macOS: `brew install heroku/brew/heroku`
   - Windows: Download the installer from [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)

2. Log in to your Heroku account via CLI:
   ```
   heroku login
   ```

## Step 2: Prepare Your Project

1. Download your project files from Replit:
   - Use the "Download as zip" option in Replit or clone the repository
   - Unzip the files to a local folder on your computer

2. Navigate to your project folder in the terminal:
   ```
   cd path/to/your/project
   ```

3. Initialize a Git repository if it's not already a Git repo:
   ```
   git init
   git add .
   git commit -m "Initial commit for Heroku deployment"
   ```

## Step 3: Create and Configure Heroku App

1. Create a new Heroku app:
   ```
   heroku create your-telegram-bot-name
   ```

2. Add PostgreSQL to your app:
   ```
   heroku addons:create heroku-postgresql:mini
   ```

3. Configure your environment variables:
   ```
   heroku config:set TELEGRAM_TOKEN=your_telegram_token
   heroku config:set ANTHROPIC_API_KEY=your_anthropic_key
   heroku config:set OPENAI_API_KEY=your_openai_key
   ```
   (Add all other required environment variables from your Replit)

## Step 4: Deploy to Heroku

1. Push your code to Heroku:
   ```
   git push heroku main
   ```
   (If you're using a branch other than main, use `git push heroku your-branch:main`)

2. Ensure both processes start:
   ```
   heroku ps:scale web=1 worker=1
   ```

3. Check the logs to make sure everything is running:
   ```
   heroku logs --tail
   ```

## Step 5: Verify Deployment

1. Your web interface should be accessible at:
   ```
   https://your-telegram-bot-name.herokuapp.com
   ```

2. Send a message to your Telegram bot to ensure it's responding

## Troubleshooting

If your app isn't working:

1. Check the logs:
   ```
   heroku logs --tail
   ```

2. Ensure environment variables are set correctly:
   ```
   heroku config
   ```

3. Restart the dynos if needed:
   ```
   heroku restart
   ```

## Keeping Your App Alive

Heroku free tier apps go to sleep after 30 minutes of inactivity. To keep it running:

1. Upgrade to a paid dyno ($7/month per dyno)
2. Use a service like [UptimeRobot](https://uptimerobot.com/) to ping your app every 25 minutes

With these steps, your bot should run continuously as long as your Heroku account remains active.