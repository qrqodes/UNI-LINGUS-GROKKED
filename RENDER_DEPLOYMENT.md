# Render Deployment Guide for Your Telegram Bot

This guide will help you deploy your Telegram translation bot to Render for 24/7 operation, even when your personal computer is turned off.

## Prerequisites

- A [Render account](https://render.com/) (sign up for free)
- Your Telegram bot token and API keys (for Anthropic and OpenAI)
- Your project files (already prepared in Replit)

## Step 1: Prepare Your Project for Deployment

1. **Download your project from Replit:**
   - Use the "Download as zip" option in Replit
   - Unzip the files to a folder on your computer

2. **Create a GitHub repository:**
   - Create a new repository on GitHub
   - Push your code to this repository
   ```
   git init
   git add .
   git commit -m "Initial commit for Render deployment"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repository-name.git
   git push -u origin main
   ```

## Step 2: Deploy to Render Using the Blueprint

Render provides two easy ways to deploy your application:

### Option 1: Deploy Using render.yaml (Recommended)

1. **Log in to your Render account**

2. **Create a new "Blueprint":**
   - Go to the Dashboard and click on "New +"
   - Select "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` file
   - Click "Apply"

3. **Configure your services:**
   - Render will create all the necessary services defined in the `render.yaml` file
   - You'll be prompted to enter your environment variables:
     - `TELEGRAM_TOKEN`: Your Telegram bot token
     - `ANTHROPIC_API_KEY`: Your Anthropic API key
     - `OPENAI_API_KEY`: Your OpenAI API key
   - Review and confirm

### Option 2: Manual Setup

If you prefer to set up services manually:

1. **Create a PostgreSQL database:**
   - Go to the Dashboard and click on "New +"
   - Select "PostgreSQL"
   - Name it "telegram-bot-db"
   - Choose the free plan
   - Click "Create Database"
   - Copy the internal connection string (you'll need it for the next steps)

2. **Create a Web Service:**
   - Go to the Dashboard and click on "New +"
   - Select "Web Service"
   - Connect your GitHub repository
   - Name: "telegram-bot-web"
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --log-file=-`
   - Add the following environment variables:
     - `TELEGRAM_TOKEN`: Your Telegram bot token
     - `ANTHROPIC_API_KEY`: Your Anthropic API key
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `DATABASE_URL`: The internal connection string from step 1
   - Choose the free plan
   - Click "Create Web Service"

3. **Create a Background Worker:**
   - Go to the Dashboard and click on "New +"
   - Select "Background Worker"
   - Connect your GitHub repository
   - Name: "telegram-bot-worker"
   - Runtime: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python main.py`
   - Add the same environment variables as the web service
   - Choose the free plan
   - Click "Create Background Worker"

## Step 3: Verify Your Deployment

1. **Check the deployment logs:**
   - Navigate to your services in the Render dashboard
   - Look at the logs to make sure everything is starting correctly
   - If there are any errors, troubleshoot based on the log messages

2. **Test your bot:**
   - Send a message to your Telegram bot
   - It should now respond as expected

## Important Notes on Render's Free Tier

1. **Free PostgreSQL databases on Render:**
   - Are deleted after 90 days
   - Have a storage limit of 256 MB
   - Consider upgrading to a paid plan ($7/month) for persistent storage

2. **Free Web Services and Background Workers:**
   - Spin down after 15 minutes of inactivity
   - Have a startup time when accessed again (this can cause slight delays)
   - Have limited resources (512 MB RAM, 0.1 CPU)

3. **To keep your bot running 24/7:**
   - Upgrade to a paid plan (starts at $7/month)
   - Or set up a service like UptimeRobot to ping your web service every 14 minutes

## Maintaining Multiple Deployments

Since you're deploying to both Heroku and Render:

1. **Using the same database:**
   - Consider using a single database for both deployments
   - This ensures consistent data regardless of which deployment is active

2. **Webhook vs. Polling:**
   - If using webhook mode, ensure only one deployment has the webhook active
   - If using polling mode (as your current implementation does), both can run simultaneously
   - The first deployment to poll will receive the updates

3. **Synchronizing code:**
   - When you update your bot, be sure to update both deployments
   - Using a GitHub repository makes this easier