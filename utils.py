
import os
import tempfile
import glob
import hashlib
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from config import OPENAI_API_KEY, EMBED_MODEL
import streamlit as st



# ===== STREAMING CALLBACK =====
class StreamingCallback(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# ===== Optimized Helpers =====
@st.cache_data(show_spinner=False)
def load_pdfs_from_folder(folder_path):
    """Cached PDF loading"""
    pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    all_docs = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = os.path.basename(pdf)
        all_docs.extend(docs)
    return all_docs

def load_pdfs_from_uploaded_files(uploaded_files):
    """Load uploaded PDFs"""
    all_docs = []
    tmp_dir = tempfile.mkdtemp(prefix="uploaded_pdfs_")
    for f in uploaded_files:
        tmp_path = os.path.join(tmp_dir, f.name)
        with open(tmp_path, "wb") as out:
            out.write(f.read())
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = f.name
        all_docs.extend(docs)
    return all_docs, tmp_dir

def custom_recursive_splitter(chunk_size, chunk_overlap):
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

# Simplified context building
def build_context(docs_and_scores):
    """Build concise context from documents"""
    pieces = []
    for idx, (doc, score) in enumerate(docs_and_scores, 1):
        meta = doc.metadata or {}
        file_name = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        # Truncate to first 200 chars for speed
        text = doc.page_content.strip()[:200]
        pieces.append(f"[{file_name}, p.{page}]: {text}")
    return "\n\n".join(pieces)

# Simple cache key
def make_cache_key(query: str):
    return hashlib.md5(query.encode()).hexdigest()

@st.cache_resource
def get_embeddings():
    """Cached embeddings instance"""
    return OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
