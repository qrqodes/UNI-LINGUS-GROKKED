# Deploying Your Bot to Both Heroku and Render

This guide provides step-by-step instructions to deploy your enhanced Telegram translation bot to both Heroku and Render, ensuring it runs 24/7 even when your computer is off.

## Preparation Checklist

Before starting deployment, make sure you have:

- [ ] Your Telegram bot token
- [ ] Your Anthropic API key
- [ ] Your OpenAI API key (if used)
- [ ] Accounts on both Heroku and Render
- [ ] Heroku CLI installed on your computer
- [ ] Git installed on your computer

## Step 1: Download Your Project From Replit

1. In your Replit project, click "..." in the top-right corner and select "Download as zip"
2. Extract the zip file to a folder on your computer

## Step 2: Prepare Your Project Files

1. Make sure you have these files in your project:
   - `Procfile` (for Heroku)
   - `render.yaml` (for Render)
   - `runtime.txt` (Python version)
   - `deployment-requirements.txt` (rename to `requirements.txt` after downloading)

2. Create a GitHub repository for your project:
   ```
   git init
   git add .
   git commit -m "Initial commit for deployment"
   git branch -M main
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git push -u origin main
   ```

## Step 3: Deploy to Heroku

Follow the detailed steps in the `HEROKU_DEPLOYMENT.md` document, but here's a summary:

1. Login to Heroku:
   ```
   heroku login
   ```

2. Create a new Heroku app:
   ```
   heroku create your-bot-name
   ```

3. Add PostgreSQL database:
   ```
   heroku addons:create heroku-postgresql:mini
   ```

4. Configure environment variables:
   ```
   heroku config:set TELEGRAM_TOKEN=your_telegram_token
   heroku config:set ANTHROPIC_API_KEY=your_anthropic_key
   heroku config:set OPENAI_API_KEY=your_openai_key
   ```

5. Deploy your application:
   ```
   git push heroku main
   ```

6. Ensure both processes start:
   ```
   heroku ps:scale web=1 worker=1
   ```

7. Check logs:
   ```
   heroku logs --tail
   ```

## Step 4: Deploy to Render

Follow the detailed steps in the `RENDER_DEPLOYMENT.md` document, but here's a summary:

1. Login to your Render account

2. Create a new Blueprint:
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` file
   - Enter your environment variables when prompted
   
   OR
   
   Manually create:
   - A PostgreSQL database
   - A web service (for the Flask app)
   - A background worker (for the Telegram bot)

3. Configure services with the same environment variables:
   - `TELEGRAM_TOKEN`
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
   - `DATABASE_URL` (automatically set when using PostgreSQL on Render)

4. Deploy and verify the services are running

## Step 5: Test Your Deployments

1. Send a message to your Telegram bot to confirm it's working
2. Check the logs on both platforms to ensure everything is running correctly

## Keeping Your Bot Running 24/7

### On Heroku:
- Free tier dynos sleep after 30 minutes of inactivity
- To prevent this:
  - Use a paid dyno ($7/month)
  - Use UptimeRobot to ping your app every 25 minutes

### On Render:
- Free tier services spin down after 15 minutes of inactivity
- To prevent this:
  - Use a paid service ($7/month)
  - Use UptimeRobot to ping your web service every 14 minutes

## Maintaining Both Deployments

1. When you make changes to your bot:
   - Update your GitHub repository first
   - Deploy to both platforms

2. Consider using a single external database:
   - To ensure consistent data between both deployments
   - ElephantSQL offers a free tier PostgreSQL database

3. Monitor both deployments regularly:
   - Check logs for errors
   - Verify the bot is responding correctly

## Troubleshooting

If your bot stops responding:

1. Check the logs on both platforms
2. Verify all environment variables are set correctly
3. Restart the services if necessary
4. Check if your API keys are still valid

By deploying to both Heroku and Render, you have a backup if one service experiences issues, ensuring your bot remains available 24/7.