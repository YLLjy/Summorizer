import os

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv()

# API Keys and other settings
CO_API_KEY = os.getenv("CO_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
