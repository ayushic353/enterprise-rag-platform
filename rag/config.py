"""
Central configuration for the RAG platform.
Every tunable value lives here and is pulled from environment variables,
so nothing is hardcoded elsewhere in the codebase.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ---- Storage ----
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", BASE_DIR / "chroma_db"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_knowledge")

# ---- Embeddings ----
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---- Vision / captioning ----
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
VISION_MODEL = os.getenv("VISION_MODEL", "Salesforce/blip-image-captioning-large")

# ---- LLM providers ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ---- Chunking ----
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# ---- Retrieval ----
TOP_K = int(os.getenv("TOP_K", "4"))

# ---- API auth ----
# Comma-separated list of accepted API keys. Generate your own, never reuse examples.
API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
