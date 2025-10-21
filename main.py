import streamlit as st
from config import DEFAULT_TOP_K, NUM_CLUSTERS, OPENAI_API_KEY
from cluster_logic import perform_reindex, load_clustered_index  # Changed import
from chat_logic import get_llm, make_cache_key, build_context
from utils import get_embeddings, StreamingCallback
import time
import numpy as np

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
            clustered_index = perform_reindex(source, uploaded_files)
            if clustered_index:
                st.session_state.clustered_index = clustered_index
                st.session_state.answer_cache = {}  # Clear cache
                st.rerun()

    st.sidebar.markdown("---")

    # Initialize session state
    if "clustered_index" not in st.session_state:
        embeddings = get_embeddings()
        try:
            st.session_state.clustered_index = load_clustered_index(embeddings)
            st.sidebar.success("✅ Clustered Index loaded")
            
            # Add cluster statistics
            with st.sidebar.expander("📊 Cluster Statistics"):
                total_docs = sum(len(idx.index_to_docstore_id) 
                               for idx in st.session_state.clustered_index["cluster_indices"].values())
                st.write(f"Total Documents: {total_docs}")
                st.write(f"Number of Clusters: {st.session_state.clustered_index['num_clusters']}")
                
                # Display documents per cluster
                st.write("\n**Documents per Cluster:**")
                for cluster_id, idx in st.session_state.clustered_index["cluster_indices"].items():
                    doc_count = len(idx.index_to_docstore_id)
                    st.write(f"- Cluster {cluster_id}: {doc_count} docs")
                    
        except Exception as e:
            st.session_state.clustered_index = None
            st.sidebar.warning("⚠️ No clustered index found - please reindex")

    if "answer_cache" not in st.session_state:
        st.session_state.answer_cache = {}

    if "history" not in st.session_state:
        st.session_state.history = []

    # Settings
    use_streaming = True

    # Check if clustered_index exists
    if st.session_state.get("clustered_index") is None:
        st.warning("⚠️ No index loaded. Please reindex using the sidebar.")
        st.stop()

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
                t_start = time.time()
                # Get query embedding
                embeddings = get_embeddings()
                query_embedding = embeddings.embed_query(user_query)
                
                # Find nearest cluster
                clusters_to_search = st.session_state.clustered_index.get("clusters_to_search", 1)
                _, cluster_ids = st.session_state.clustered_index["centroid_index"].search(
                    np.array([query_embedding], dtype=np.float32), 
                    clusters_to_search
                )
                target_clusters = cluster_ids[0]
                
                # Get relevant documents from the nearest cluster
                top_k = st.session_state.clustered_index.get("top_k", DEFAULT_TOP_K)
                all_docs_and_scores = []
                for cluster_id in target_clusters:
                    if cluster_id in st.session_state.clustered_index["cluster_indices"]:
                        cluster_index = st.session_state.clustered_index["cluster_indices"][cluster_id]
                        all_docs_and_scores.extend(cluster_index.similarity_search_with_score(user_query, k=top_k))
                
                # Sort all results by score (distance) and take the best `top_k`
                docs_and_scores = sorted(all_docs_and_scores, key=lambda x: x[1])[:top_k]
                retrieval_time = time.time() - t_start
                
                # Show cluster information
                st.info(f"📊 Searched in Cluster(s): {', '.join(map(str, target_clusters))}")
                
                # Build context
                context = build_context(docs_and_scores)
                
                # Generate answer
                prompt = f"""Use the following context to answer the question. Include source citations.

Context:
{context}

Question: {user_query}

Answer:"""
                
                # t_start_llm = time.time() # Start timing before LLM call
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
                
                total_time = time.time() - t_start
                
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
                    # Retrieval is now too fast to measure meaningfully here, focusing on total time
                    col1.metric("Retrieval Time", f"{retrieval_time:.2f}s")
                    col2.metric("LLM Time", f"{llm_time:.2f}s")
                    col3.metric("Total Time", f"{total_time:.2f}s")
                    
                    st.markdown("**Retrieved chunks:**")
                    for idx, (doc, score) in enumerate(docs_and_scores, 1):
                        meta = doc.metadata
                        st.write(f"{idx}. {meta.get('source')} (p.{meta.get('page')}) - Score: {score:.3f}")

    # Clear history button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.history = []
        st.rerun()

    # Add cluster visualization
    if st.session_state.clustered_index is not None:
        with st.sidebar.expander("🔍 Cluster Analysis", expanded=False):
            st.write("**Active Clusters:**")
            
            # Create columns for cluster stats
            cols = st.columns(2)
            for i, (cluster_id, index) in enumerate(st.session_state.clustered_index["cluster_indices"].items()):
                with cols[i % 2]:
                    doc_count = len(index.index_to_docstore_id)
                    st.metric(
                        f"Cluster {cluster_id}",
                        f"{doc_count} docs",
                        delta=None
                    )

if __name__ == "__main__":
    main()
