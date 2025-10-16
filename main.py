# streamlit_app_optimized.py
import os
import time
import glob
import hashlib
import json
import streamlit as st
from dotenv import load_dotenv
import shutil
import tempfile
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio

load_dotenv()

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

# ===== CONFIG =====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except Exception:
        OPENAI_API_KEY = None

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found.")
    st.stop()

PDF_FOLDER = "docs"
PERSIST_DIR = "faiss_store"


# Optimized settings
CHAT_MODEL_NAME = "gpt-3.5-turbo"
MAX_TOKENS = 512  
EMBED_MODEL = "text-embedding-3-small"

# Optimized chunking (smaller chunks = faster retrieval)
CHUNK_SIZE = 400  # Reduced from 600
CHUNK_OVERLAP = 50  # Reduced from 70

# Retrieval settings
DEFAULT_TOP_K = 3  # Only get what we need
SIMILARITY_THRESHOLD = 0.7  # Simplified threshold

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

def custom_recursive_splitter():
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )


@st.cache_resource
def get_embeddings():
    """Cached embeddings instance"""
    return OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

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
        splitter = custom_recursive_splitter()
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

# ===== STREAMLIT APP =====
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("🤖 RAG PDF Chatbot")

# ===== SIDEBAR - SIMPLIFIED REINDEX =====
st.sidebar.markdown("## 📚 Index Management")

# Option selector
index_source = st.sidebar.radio(
    "Index from:",
    ["Local folder (docs/)", "Upload PDFs"],
    key="index_source"
)

uploaded_files = None
if index_source == "Upload PDFs":
    uploaded_files = st.sidebar.file_uploader(
        "Select PDF files",
        type="pdf",
        accept_multiple_files=True
    )

# ONE CLICK REINDEX
if st.sidebar.button("🔄 Reindex Now", type="primary", use_container_width=True):
    source = "upload" if index_source == "Upload PDFs" else "local"
    if source == "upload" and not uploaded_files:
        st.sidebar.error("Please upload PDFs first!")
    else:
        vectorstore = perform_reindex(source, uploaded_files)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.answer_cache = {}  # Clear cache
            st.rerun()

st.sidebar.markdown("---")

# Initialize session state
if "vectorstore" not in st.session_state:
    embeddings = get_embeddings()
    try:
        st.session_state.vectorstore = try_load_faiss(embeddings)
        st.sidebar.success("✅ Index loaded")
    except:
        st.session_state.vectorstore = None
        st.sidebar.warning("⚠️ No index found - please reindex")

if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}

if "history" not in st.session_state:
    st.session_state.history = []

# Settings
top_k = DEFAULT_TOP_K
use_streaming = True

# Check if vectorstore exists
if st.session_state.vectorstore is None:
    st.warning("⚠️ No index loaded. Please reindex using the sidebar.")
    st.stop()

vectorstore = st.session_state.vectorstore

# ===== CHAT INTERFACE =====
chat_container = st.container()

# Display chat history
for msg in st.session_state.history:
    chat_container.chat_message("user").write(msg["user"])
    chat_container.chat_message("assistant").write(msg["assistant"])

# Chat input
user_query = st.chat_input("Ask about your documents...")

if user_query:
    # Display user message
    chat_container.chat_message("user").write(user_query)
    
    # Check cache first
    cache_key = make_cache_key(user_query)
    
    if cache_key in st.session_state.answer_cache:
        # Use cached answer
        answer = st.session_state.answer_cache[cache_key]["answer"]
        chat_container.chat_message("assistant").write(answer)
        st.session_state.history.append({"user": user_query, "assistant": answer})
        st.info("💾 Using cached response")
    else:
        # Process new query
        with st.spinner("🔍 Searching..."):
            # Retrieve relevant chunks
            t_start = time.time()
            docs_and_scores = get_relevant_chunks(vectorstore, user_query, k=top_k)
            retrieval_time = time.time() - t_start
            
            # Build context
            context = build_context(docs_and_scores)
            
            # Generate answer
            prompt = f"""Use the following context to answer the question. Include source citations.

Context:
{context}

Question: {user_query}

Answer:"""
            
            llm = get_llm(streaming=use_streaming)
            
            if use_streaming:
                # Stream the response
                response_container = chat_container.chat_message("assistant").empty()
                callback = StreamingCallback(response_container)
                
                t_llm = time.time()
                answer = llm.predict(prompt, callbacks=[callback])
                llm_time = time.time() - t_llm
            else:
                t_llm = time.time()
                answer = llm.predict(prompt)
                llm_time = time.time() - t_llm
                chat_container.chat_message("assistant").write(answer)
            
            # Cache the answer
            st.session_state.answer_cache[cache_key] = {
                "answer": answer,
                "context": context,
                "timestamp": time.time()
            }
            
            # Save to history
            st.session_state.history.append({"user": user_query, "assistant": answer})
            
            # Show timing in expander
            with st.expander("⏱️ Performance Metrics"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Retrieval", f"{retrieval_time:.2f}s")
                col2.metric("LLM", f"{llm_time:.2f}s")
                col3.metric("Total", f"{retrieval_time + llm_time:.2f}s")
                
                st.markdown("**Retrieved chunks:**")
                for idx, (doc, score) in enumerate(docs_and_scores, 1):
                    meta = doc.metadata
                    st.write(f"{idx}. {meta.get('source')} (p.{meta.get('page')}) - Score: {score:.3f}")

# Clear history button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    st.rerun()