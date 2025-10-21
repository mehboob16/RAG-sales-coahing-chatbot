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
PERSIST_FILE = "clustered_index.pkl"


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

# Clustering settings
NUM_CLUSTERS = 5 # This will now be a fallback/default
PERSIST_FILE = "clustered_index.pkl"

# --- DYNAMIC SETTINGS BASED ON CHUNK COUNT ---
# This dictionary defines tiers for scaling.
# Key: max number of chunks for this tier
# Value: (number_of_clusters, clusters_to_search, top_k_retrieval)
DYNAMIC_SETTINGS = {
    500:   (5, 1, 3),    # For up to 500 chunks
    1500:  (10, 2, 4),   # For up to 1500 chunks
    5000:  (20, 3, 5),   # For up to 5000 chunks
    # Add more tiers for larger document sets
}

# Default settings for very large document sets (if count exceeds all tiers)
DEFAULT_DYNAMIC_SETTING = (25, 3, 6)

def get_dynamic_settings(num_chunks):
    """Gets the appropriate settings tuple based on the number of chunks."""
    for threshold, settings in sorted(DYNAMIC_SETTINGS.items()):
        if num_chunks <= threshold:
            return settings
    return DEFAULT_DYNAMIC_SETTING