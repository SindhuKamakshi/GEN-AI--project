from pathlib import Path
from dotenv import load_dotenv
import os

# backend/
BASE_DIR = Path(__file__).resolve().parent

# Project root: CodeLens-AI/
PROJECT_ROOT = BASE_DIR.parent

# Load .env from project root
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
REPOSITORIES_DIR = DATA_DIR / "repositories"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Create required directories
REPOSITORIES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")