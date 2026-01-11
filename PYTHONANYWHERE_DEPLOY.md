# UNI LINGUS - PythonAnywhere Deployment Guide

## Step 1: Create PythonAnywhere Account
1. Go to https://www.pythonanywhere.com
2. Sign up for a **free account**
3. Your app will be at: `yourusername.pythonanywhere.com`

## Step 2: Upload Your Files
1. Go to **Files** tab in PythonAnywhere
2. Create a folder: `/home/yourusername/uni_lingus`
3. Upload these files:
   - `app.py`
   - `main.py`
   - `models.py`
   - `shared_services.py`
   - `transcription.py`
   - `uni_advanced_translator.py`
   - `updated_uni_grammar.py`
   - `uni_language_development.py`
   - `uni_web_integration.py`
   - `requirements_pythonanywhere.txt` (rename to `requirements.txt`)
   - `templates/` folder (with index.html)
   - `static/` folder

## Step 3: Create Virtual Environment
Open a **Bash console** in PythonAnywhere and run:

```bash
cd ~/uni_lingus
mkvirtualenv --python=/usr/bin/python3.11 uni_lingus
pip install -r requirements.txt
```

## Step 4: Set Up Web App
1. Go to **Web** tab
2. Click **Add a new web app**
3. Choose **Manual configuration** (not Flask)
4. Select **Python 3.11**

## Step 5: Configure WSGI
1. In the Web tab, click on the **WSGI configuration file** link
2. Replace contents with:

```python
import sys
import os

project_home = '/home/yourusername/uni_lingus'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

os.environ['SESSION_SECRET'] = 'your-random-secret-key-here'
os.environ['FLASK_ENV'] = 'production'

from app import app as application
```

## Step 6: Set Virtual Environment Path
In the Web tab, set:
- **Virtualenv**: `/home/yourusername/.virtualenvs/uni_lingus`

## Step 7: Set Up Database (Optional)
PythonAnywhere offers PostgreSQL for paid accounts.
For free tier, use SQLite by changing `app.py`:

```python
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///uni_lingus.db"
```

## Step 8: Add Environment Variables
In the Web tab, go to **Environment variables** section and add:
- `SESSION_SECRET`: (any random string)
- `VENICE_API_KEY`: (your Venice AI key for chat)
- `TELEGRAM_BOT_TOKEN`: (your Telegram bot token)

## Step 9: Reload and Test
1. Click **Reload** button in Web tab
2. Visit `yourusername.pythonanywhere.com`

## Telegram Bot Note
The Telegram bot requires a **separate always-on task** which is only available on paid PythonAnywhere accounts. On free tier, only the web app will work.

## What Works on Free Tier
- Translation (all 6 languages)
- Text-to-Speech (gTTS)
- AI Chat (if you have API keys)
- Web interface

## What Doesn't Work on Free Tier
- Telegram bot (needs always-on task)
- Heavy AI features (removed)
- Advanced audio processing (removed)
