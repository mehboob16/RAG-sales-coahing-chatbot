import os
import argparse
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# ===== CONFIG =====
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
PDF_FOLDER = "docs"        # change this to your local PDF file path
PERSIST_DIR = "faiss_store"
CHAT_MODEL_NAME = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

# ===== UTILITIES =====
import glob

def load_pdfs_from_folder(folder_path):
    """Load all PDFs from a folder and add filename metadata."""
    pdf_paths = glob.glob(os.path.join(folder_path, "*.pdf"))
    all_docs = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = os.path.basename(pdf)
        all_docs.extend(docs)
    return all_docs

def custom_recursive_splitter():
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],  # paragraph → line → space
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

def create_or_load_faiss(chunks, embeddings, reindex=False):
    """Reuses FAISS if exists unless reindex=True."""
    if not reindex and os.path.exists(PERSIST_DIR) and any(f.endswith(".index") for f in os.listdir(PERSIST_DIR)):
        print("✅ Using existing FAISS index.")
        return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)

    print("⚙️ Creating new FAISS index...")
    faiss_db = FAISS.from_documents(chunks, embeddings)
    faiss_db.save_local(PERSIST_DIR)
    return faiss_db

from langchain.prompts import PromptTemplate

def build_chain(vectorstore):
    llm = ChatOpenAI(model_name=CHAT_MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful assistant. Use the following context to answer the user's question.
If you mention facts, include the document name (from metadata 'source') and the page number if available.

{context}

Question: {question}
Helpful answer:"""
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True
    )

def summarize_chunk_usage(llm, question, source_docs):
    """Ask GPT which chunk contributed most."""
    summaries = []
    for i, doc in enumerate(source_docs, 1):
        content = doc.page_content.strip()
        meta = doc.metadata
        start_line = content.splitlines()[0][:80]
        end_line = content.splitlines()[-1][:80]
        summaries.append(
            f"Chunk {i} — File: {meta.get('source','Unknown file')} (Page {meta.get('page','?')}):\n"
            f"Start: {start_line}\nEnd: {end_line}\n"
        )

    summary_prompt = f"""
The user asked: {question}

Here are the chunks retrieved:
{chr(10).join(summaries)}

Based on the answer you gave, which chunk most strongly supported your response? Reply like:
"Most influential: Chunk X (Page Y)".
"""
    resp = llm.invoke(summary_prompt)
    return summaries, resp.content.strip()

# ===== CLI ARGS =====
parser = argparse.ArgumentParser()
parser.add_argument("--reindex", action="store_true", help="Delete old FAISS and rebuild embeddings.")
args, _ = parser.parse_known_args()

# ===== LOAD PDF + EMBEDDINGS =====
st.set_page_config(page_title="Local RAG PDF Chatbot", layout="centered")
st.title("RAG Chatbot for Local PDFs")


if OPENAI_API_KEY.startswith("YOUR_"):
    st.error("Please set your OpenAI API key in the code or as environment variable OPENAI_API_KEY.")
    st.stop()

if not os.path.exists(PDF_FOLDER):
    st.error(f"Folder not found at {PDF_FOLDER}. Update the path in code.")
    st.stop()

docs = load_pdfs_from_folder(PDF_FOLDER)

splitter = custom_recursive_splitter()
chunks = splitter.split_documents(docs)
embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)
vectorstore = create_or_load_faiss(chunks, embeddings, reindex=args.reindex)
qa_chain = build_chain(vectorstore)

# ===== CHAT UI =====
if "history" not in st.session_state:
    st.session_state.history = []

chat_container = st.container()

user_query = st.chat_input("Ask something about the document...")
if user_query:
    chat_container.chat_message("user").write(user_query)
    chat_history = [(msg["user"], msg["assistant"]) for msg in st.session_state.history]

    result = qa_chain({"question": user_query, "chat_history": chat_history})
    answer = result["answer"]
    src_docs = result["source_documents"]

    # Summarize chunks used
    llm = ChatOpenAI(model_name=CHAT_MODEL_NAME, temperature=0, openai_api_key=OPENAI_API_KEY)
    summaries, strongest = summarize_chunk_usage(llm, user_query, src_docs)

    # Display response
    chat_container.chat_message("assistant").write(answer)
    st.session_state.history.append({"user": user_query, "assistant": answer})

    with st.expander("📄 Chunks used to construct this answer"):
        st.write("**Most influential:**", strongest)
        st.markdown("**Chunk details:**")
        for s in summaries:
            st.code(s, language="markdown")

# Show history
for msg in st.session_state.history:
    with chat_container.chat_message("user"):
        st.write(msg["user"])
    with chat_container.chat_message("assistant"):
        st.write(msg["assistant"])
