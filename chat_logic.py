from config import PERSIST_DIR, DEFAULT_TOP_K, SIMILARITY_THRESHOLD, CHAT_MODEL_NAME, MAX_TOKENS, PDF_FOLDER, OPENAI_API_KEY, CHUNK_OVERLAP, CHUNK_SIZE
from langchain_community.chat_models import ChatOpenAI
import os
import hashlib
import streamlit as st
from langchain.vectorstores import FAISS
from utils import get_embeddings, load_pdfs_from_folder, load_pdfs_from_uploaded_files, custom_recursive_splitter


@st.cache_resource
def get_llm(streaming=True):
    """Cached LLM instance"""
    if streaming:
        return ChatOpenAI(
            model_name=CHAT_MODEL_NAME,
            temperature=0,
            max_tokens=MAX_TOKENS,
            openai_api_key=OPENAI_API_KEY,
            streaming=True
        )
    return ChatOpenAI(
        model_name=CHAT_MODEL_NAME,
        temperature=0,
        max_tokens=MAX_TOKENS,
        openai_api_key=OPENAI_API_KEY
    )

def create_faiss_from_chunks(chunks, embeddings, progress_bar=None):
    """Create FAISS index with progress updates"""
    if progress_bar:
        progress_bar.progress(0.5, "Creating embeddings...")
    faiss_db = FAISS.from_documents(chunks, embeddings)
    if progress_bar:
        progress_bar.progress(0.8, "Saving index...")
    faiss_db.save_local(PERSIST_DIR)
    if progress_bar:
        progress_bar.progress(1.0, "Done!")
    return faiss_db

def try_load_faiss(embeddings):
    """Try to load persisted FAISS index"""
    if not os.path.exists(PERSIST_DIR):    
        os.makedirs(PERSIST_DIR, exist_ok=True) # Ensure directory exists
        
        raise FileNotFoundError("No persisted index found.")
    return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)

# Simplified retrieval (no complex normalization)
def get_relevant_chunks(vectorstore, query: str, k: int = DEFAULT_TOP_K):
    """Fast retrieval with simple scoring"""
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
    # FAISS returns distances (lower = better), convert to similarity
    results = []
    for doc, score in docs_and_scores:
        # Simple inversion: similarity = 1 / (1 + distance)
        sim = 1.0 / (1.0 + float(score))
        if sim >= SIMILARITY_THRESHOLD or len(results) < 3:
            results.append((doc, sim))
    return results[:k]

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

# ===== ONE-CLICK REINDEX FUNCTION =====
def perform_reindex(source="local", uploaded_files=None):
    """Single function to handle all reindexing"""
    progress = st.progress(0.0, "Starting reindex...")
    
    try:
        # Step 1: Load documents
        progress.progress(0.2, "Loading PDFs...")
        if source == "upload" and uploaded_files:
            docs, tmpdir = load_pdfs_from_uploaded_files(uploaded_files)
            st.session_state.uploaded_tmpdirs = st.session_state.get("uploaded_tmpdirs", [])
            st.session_state.uploaded_tmpdirs.append(tmpdir)
        else:
            docs = load_pdfs_from_folder(PDF_FOLDER)
        
        if not docs:
            st.error("No documents found!")
            return None
        
        st.sidebar.info(f"Loaded {len(docs)} pages")
        
        # Step 2: Split
        progress.progress(0.3, "Splitting documents...")
        splitter = custom_recursive_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)
        st.sidebar.info(f"Created {len(chunks)} chunks")
        
        # Step 3: Create index
        embeddings = get_embeddings()
        vectorstore = create_faiss_from_chunks(chunks, embeddings, progress)
        
        progress.empty()
        st.sidebar.success(f"✅ Indexed {len(chunks)} chunks successfully!")
        return vectorstore
        
    except Exception as e:
        progress.empty()
        st.error(f"Reindex failed: {e}")
        return None
