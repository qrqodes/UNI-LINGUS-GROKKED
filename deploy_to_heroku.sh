#!/bin/bash
# Script to prepare and deploy your Telegram bot to Heroku
# Run this script after downloading your project from Replit

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================${NC}"
echo -e "${BLUE}   Telegram Bot Heroku Deployment Helper         ${NC}"
echo -e "${BLUE}==================================================${NC}"
echo

# Check if required tools are installed
echo -e "${YELLOW}Checking required tools...${NC}"

# Check for Heroku CLI
if ! command -v heroku &> /dev/null; then
    echo -e "${RED}❌ Heroku CLI not found!${NC}"
    echo -e "Please install the Heroku CLI first:"
    echo -e "  macOS: brew install heroku/brew/heroku"
    echo -e "  Other systems: https://devcenter.heroku.com/articles/heroku-cli"
    exit 1
else
    echo -e "${GREEN}✅ Heroku CLI found${NC}"
fi

# Check for Git
if ! command -v git &> /dev/null; then
    echo -e "${RED}❌ Git not found!${NC}"
    echo -e "Please install Git first:"
    echo -e "  macOS: brew install git"
    echo -e "  Other systems: https://git-scm.com/downloads"
    exit 1
else
    echo -e "${GREEN}✅ Git found${NC}"
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python not found!${NC}"
    echo -e "Please install Python first:"
    echo -e "  macOS: brew install python"
    echo -e "  Other systems: https://www.python.org/downloads/"
    exit 1
else
    echo -e "${GREEN}✅ Python found${NC}"
    echo -e "   $(python3 --version)"
fi

echo

# Check for existing files
echo -e "${YELLOW}Checking project files...${NC}"
if [ ! -f "main.py" ] || [ ! -f "app.py" ]; then
    echo -e "${RED}❌ Essential project files not found!${NC}"
    echo -e "Make sure you're running this in the directory containing your Telegram bot files."
    exit 1
else
    echo -e "${GREEN}✅ Essential project files found${NC}"
fi

# Check for required Heroku files
echo -e "${YELLOW}Checking Heroku configuration files...${NC}"

# Check/create Procfile
if [ ! -f "Procfile" ]; then
    echo -e "${YELLOW}⚠️ Procfile not found, creating it...${NC}"
    echo "web: gunicorn app:app --log-file=-" > Procfile
    echo "worker: python main.py" >> Procfile
    echo -e "${GREEN}✅ Created Procfile${NC}"
else
    echo -e "${GREEN}✅ Procfile exists${NC}"
fi

# Check/create runtime.txt
if [ ! -f "runtime.txt" ]; then
    echo -e "${YELLOW}⚠️ runtime.txt not found, creating it...${NC}"
    echo "python-3.11.x" > runtime.txt
    echo -e "${GREEN}✅ Created runtime.txt${NC}"
else
    echo -e "${GREEN}✅ runtime.txt exists${NC}"
fi

echo

# Login to Heroku
echo -e "${YELLOW}You need to log in to Heroku. A browser window will open for authentication.${NC}"
read -p "Press Enter to continue..."
heroku login

# Create a new Heroku app
echo
echo -e "${YELLOW}Creating a new Heroku app for your bot...${NC}"
read -p "Enter a name for your Heroku app (letters, numbers, and dashes only): " app_name

if ! heroku create $app_name; then
    echo -e "${RED}❌ Failed to create app '$app_name'. The name might be taken or invalid.${NC}"
    read -p "Would you like to try a different name? (y/n): " try_again
    if [ "$try_again" = "y" ]; then
        read -p "Enter a different name for your Heroku app: " app_name
        if ! heroku create $app_name; then
            echo -e "${RED}❌ Failed again. Please run the script later and choose a unique name.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Exiting. Please run the script again when ready.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✅ Heroku app '$app_name' created successfully${NC}"

# Add PostgreSQL
echo
echo -e "${YELLOW}Adding PostgreSQL database to your app...${NC}"
if ! heroku addons:create heroku-postgresql:mini -a $app_name; then
    echo -e "${RED}❌ Failed to add PostgreSQL. This might be because you need to add a payment method.${NC}"
    echo -e "${YELLOW}You can add it later with: heroku addons:create heroku-postgresql:mini -a $app_name${NC}"
else
    echo -e "${GREEN}✅ PostgreSQL added to your app${NC}"
fi

# Configure environment variables
echo
echo -e "${YELLOW}Setting up environment variables...${NC}"
echo -e "Let's set up the required environment variables for your bot."

read -p "Enter your TELEGRAM_TOKEN: " telegram_token
heroku config:set TELEGRAM_TOKEN=$telegram_token -a $app_name
echo -e "${GREEN}✅ TELEGRAM_TOKEN set${NC}"

read -p "Enter your ANTHROPIC_API_KEY: " anthropic_key
heroku config:set ANTHROPIC_API_KEY=$anthropic_key -a $app_name
echo -e "${GREEN}✅ ANTHROPIC_API_KEY set${NC}"

read -p "Enter your OPENAI_API_KEY: " openai_key
heroku config:set OPENAI_API_KEY=$openai_key -a $app_name
echo -e "${GREEN}✅ OPENAI_API_KEY set${NC}"

# Ask if there are any other environment variables
read -p "Do you need to set any other environment variables? (y/n): " more_vars
while [ "$more_vars" = "y" ]; do
    read -p "Enter the variable name: " var_name
    read -p "Enter the variable value: " var_value
    heroku config:set $var_name=$var_value -a $app_name
    echo -e "${GREEN}✅ $var_name set${NC}"
    read -p "Add another variable? (y/n): " more_vars
done

# Initialize Git repository (if needed)
echo
echo -e "${YELLOW}Setting up Git repository...${NC}"

if [ ! -d ".git" ]; then
    echo -e "Initializing Git repository..."
    git init
    echo -e "${GREEN}✅ Git repository initialized${NC}"
else
    echo -e "${GREEN}✅ Git repository already exists${NC}"
fi

# Create .gitignore file
if [ ! -f ".gitignore" ]; then
    echo -e "Creating .gitignore file..."
    echo "__pycache__/" > .gitignore
    echo "*.py[cod]" >> .gitignore
    echo "*$py.class" >> .gitignore
    echo "*.so" >> .gitignore
    echo ".env" >> .gitignore
    echo "venv/" >> .gitignore
    echo "ENV/" >> .gitignore
    echo "env/" >> .gitignore
    echo "*.log" >> .gitignore
    echo "local_settings.py" >> .gitignore
    echo -e "${GREEN}✅ Created .gitignore file${NC}"
fi

# Commit files
echo -e "Adding files to Git..."
git add .
git commit -m "Prepare for Heroku deployment"
echo -e "${GREEN}✅ Files committed to Git${NC}"

# Deploy to Heroku
echo
echo -e "${YELLOW}Ready to deploy to Heroku!${NC}"
read -p "Deploy now? (y/n): " deploy_now

if [ "$deploy_now" = "y" ]; then
    echo -e "Deploying to Heroku... This might take a few minutes."
    if ! git push heroku main; then
        echo -e "${YELLOW}Trying alternative push command...${NC}"
        if ! git push heroku master; then
            echo -e "${RED}❌ Deployment failed. Please check the error messages above.${NC}"
            echo -e "You can try manually with: git push heroku main"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}✅ Deployment successful!${NC}"
    
    # Scale dynos
    echo -e "Scaling dynos..."
    heroku ps:scale web=1 worker=1 -a $app_name
    echo -e "${GREEN}✅ Dynos scaled successfully${NC}"
    
    # Show app info
    echo
    echo -e "${BLUE}==================================================${NC}"
    echo -e "${GREEN}Your Telegram bot has been deployed to Heroku!${NC}"
    echo -e "${BLUE}==================================================${NC}"
    echo
    echo -e "Web interface URL: https://$app_name.herokuapp.com"
    echo -e "To check your app logs: heroku logs --tail -a $app_name"
    echo -e "To restart your app: heroku restart -a $app_name"
    
    # Show logs
    echo
    echo -e "${YELLOW}Showing application logs (press Ctrl+C to exit logs):${NC}"
    heroku logs --tail -a $app_name
else
    echo
    echo -e "${BLUE}==================================================${NC}"
    echo -e "${YELLOW}Deployment prepared but not executed.${NC}"
    echo -e "${BLUE}==================================================${NC}"
    echo
    echo -e "When you're ready, deploy with: git push heroku main"
    echo -e "Then scale dynos with: heroku ps:scale web=1 worker=1 -a $app_name"
    echo -e "And check logs with: heroku logs --tail -a $app_name"
fi

echo
echo -e "${GREEN}Deployment preparation complete!${NC}"