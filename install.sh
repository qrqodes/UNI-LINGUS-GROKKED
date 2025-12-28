#!/bin/bash

# Enhanced Language Learning and Translation Bot Installation Script
# This script sets up the environment for the bot to run

echo "=== Starting Installation Process ==="
echo "Setting up Enhanced Language Learning and Translation Bot..."

# Check for Python 3
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    # Check if this is Python 3
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d. -f1)
    if [ "$PYTHON_VERSION" -ge 3 ]; then
        PYTHON_CMD="python"
    else
        echo "Error: Python 3 is required but not found."
        echo "Please install Python 3 before continuing."
        exit 1
    fi
else
    echo "Error: Python is not installed or not in PATH."
    echo "Please install Python 3 before continuing."
    exit 1
fi

echo "Using Python command: $PYTHON_CMD"

# Create virtual environment
echo -e "\n=== Creating Virtual Environment ==="
$PYTHON_CMD -m venv venv || {
    echo "Failed to create virtual environment."
    echo "Continuing without virtual environment..."
}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    else
        echo "Warning: Cannot find activation script for virtual environment."
        echo "Continuing without activation..."
    fi
fi

# Install required packages
echo -e "\n=== Installing Required Packages ==="
$PYTHON_CMD -m pip install --upgrade pip || echo "Warning: Failed to upgrade pip, continuing..."
$PYTHON_CMD -m pip install python-telegram-bot deep-translator gtts pypinyin langdetect googletrans-py flask gunicorn

# Check if Telegram token is set
echo -e "\n=== Checking Environment Variables ==="
if [ -z "$TELEGRAM_TOKEN" ]; then
    echo "Warning: TELEGRAM_TOKEN environment variable is not set."
    echo "You will need to set this before running the bot."
    
    # Ask user for token
    echo -e "\nWould you like to set your Telegram token now? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "\nPlease enter your Telegram token (from BotFather):"
        read -r token
        
        # Write to .env file
        echo "TELEGRAM_TOKEN=$token" > .env
        echo "Token saved to .env file."
        
        # Export it for immediate use
        export TELEGRAM_TOKEN="$token"
    else
        echo "You can set the token later by running: export TELEGRAM_TOKEN=your_token"
        echo "Or by adding it to the .env file."
    fi
else
    echo "TELEGRAM_TOKEN is already set."
fi

# Run setup script
echo -e "\n=== Running Setup Script ==="
$PYTHON_CMD setup.py

# Display instructions
echo -e "\n=== Installation Complete ==="
echo "To start the bot, run:"
echo "  $PYTHON_CMD main.py"
echo ""
echo "To start the web application, run:"
echo "  gunicorn --bind 0.0.0.0:5000 main:app"
echo ""
echo "For development mode with auto-reload:"
echo "  FLASK_APP=app.py FLASK_DEBUG=1 flask run --host=0.0.0.0"
echo ""
echo "Enjoy your Enhanced Language Learning and Translation Bot!"

# Final check
echo -e "\n=== Environment Check ==="
if [ -z "$TELEGRAM_TOKEN" ]; then
    echo "Warning: Please set your TELEGRAM_TOKEN before starting the bot."
    echo "Run: export TELEGRAM_TOKEN=your_token"
else
    echo "Environment looks good! You're ready to start the bot."
fi