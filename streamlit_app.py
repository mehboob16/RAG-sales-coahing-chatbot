import streamlit as st
from config import DEFAULT_TOP_K
from chat_logic import perform_reindex, get_relevant_chunks, build_context, get_llm, make_cache_key
from faiss_utils import try_load_faiss
from utils import get_embeddings, StreamingCallback
import time

def main():
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

if __name__ == "__main__":
    main()
