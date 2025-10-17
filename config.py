import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# ===== CONFIG =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        OPENAI_API_KEY = None

PDF_FOLDER = "docs"
PERSIST_DIR = "faiss_store"

# Optimized settings
CHAT_MODEL_NAME = "gpt-3.5-turbo"
MAX_TOKENS = 512
EMBED_MODEL = "text-embedding-3-small"

# Optimized chunking (smaller chunks = faster retrieval)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Retrieval settings
DEFAULT_TOP_K = 3
SIMILARITY_THRESHOLD = 0.7
