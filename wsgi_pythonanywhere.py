# WSGI file for PythonAnywhere deployment
# 
# IMPORTANT: Do NOT add secrets directly to this file!
# Use PythonAnywhere's Environment Variables section in the Web tab instead.
#
# Required environment variables (set in PythonAnywhere Web tab):
# - SESSION_SECRET: Random string for session security
# - VENICE_API_KEY: (Optional) For AI chat feature
# - TELEGRAM_BOT_TOKEN: (Optional) For Telegram bot

import sys
import os

# Add your project directory to the sys.path
# Replace YOUR_USERNAME with your PythonAnywhere username
project_home = '/home/YOUR_USERNAME/uni_lingus'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Generate a default session secret if not provided
if not os.environ.get('SESSION_SECRET'):
    import secrets
    os.environ['SESSION_SECRET'] = secrets.token_hex(32)

# Import Flask app
from app import app as application
