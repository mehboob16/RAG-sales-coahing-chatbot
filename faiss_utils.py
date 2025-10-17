from langchain.vectorstores import FAISS
import os
from config import PERSIST_DIR

def try_load_faiss(embeddings):
    """Try to load persisted FAISS index"""
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError("No persisted index found.")
    return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
