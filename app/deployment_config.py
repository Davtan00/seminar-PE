import os
from app.config import get_settings

settings = get_settings()

HEROKU_CONFIG = {
    "app_name": settings.HEROKU_APP_NAME,
    "region": os.getenv("HEROKU_REGION", "eu"),
    "stack": os.getenv("HEROKU_STACK", "heroku-22"),
    "repo_url": f"https://git.heroku.com/{settings.HEROKU_APP_NAME}.git",
    "web_url": f"https://{settings.HEROKU_APP_NAME}.herokuapp.com"
}

# Using formatted string, this is just my script/template I always use for heroku deployments
DEPLOYMENT_COMMANDS = f"""
# Initial Heroku Setup
heroku login
heroku create {settings.HEROKU_APP_NAME}

# Set environment variables (DO NOT INCLUDE ACTUAL KEYS IN THIS FILE)
heroku config:set OPENAI_API_KEY=$OPENAI_API_KEY
heroku config:set HEROKU_API_KEY=$HEROKU_API_KEY
heroku config:set HEROKU_APP_NAME={settings.HEROKU_APP_NAME}

# Deploy
git init
git add .
git commit -m "Initial commit"
heroku git:remote -a {settings.HEROKU_APP_NAME}
git push heroku main

# View logs
heroku logs --tail
"""
