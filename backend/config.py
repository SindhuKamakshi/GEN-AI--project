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
configured_data_dir = os.getenv("CODELENS_DATA_DIR")
DATA_DIR = (
    Path(configured_data_dir).expanduser()
    if configured_data_dir
    else PROJECT_ROOT / "data"
)
if not DATA_DIR.is_absolute():
    DATA_DIR = PROJECT_ROOT / DATA_DIR

REPOSITORIES_DIR = DATA_DIR / "repositories"
CHROMA_DIR = DATA_DIR / "chroma_db"
configured_origins = os.getenv("CODELENS_ALLOWED_ORIGINS")
ALLOWED_ORIGINS = (
    [
        origin.strip().rstrip("/")
        for origin in configured_origins.split(",")
        if origin.strip()
    ]
    if configured_origins
    else [
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ]
)

# Create required directories
REPOSITORIES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")