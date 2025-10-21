import streamlit as st
import numpy as np
import faiss
import pickle
import os
import joblib  # Add this import
from sklearn.cluster import MiniBatchKMeans
from langchain_community.vectorstores import FAISS as LangchainFAISS

from config import (
    PERSIST_DIR, PDF_FOLDER, CHUNK_SIZE, CHUNK_OVERLAP, PERSIST_FILE, get_dynamic_settings
)
from utils import (
    get_embeddings, load_pdfs_from_folder, load_pdfs_from_uploaded_files,
    custom_recursive_splitter
)

def create_clustered_index(chunks, embeddings_model, num_clusters, clusters_to_search, top_k, progress_bar=None):
    """
    Creates a clustered FAISS index.
    1. Embeds all chunks.
    2. Runs KMeans to find centroids.
    3. Creates a FAISS index for centroids.
    4. Creates a separate FAISS index for each cluster's chunks.
    """
    st.sidebar.info(f"Embedding {len(chunks)} chunks...")
    if progress_bar: progress_bar.progress(0.5, "Creating embeddings for all chunks...")
    
    chunk_embeddings = embeddings_model.embed_documents([c.page_content for c in chunks])
    chunk_embeddings_np = np.array(chunk_embeddings, dtype=np.float32)

    # --- Clustering ---
    if progress_bar: progress_bar.progress(0.6, f"Running KMeans clustering ({num_clusters} clusters)...")
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=200, n_init='auto')
    kmeans.fit(chunk_embeddings_np)

    # --- Create Centroid Index ---
    if progress_bar: progress_bar.progress(0.7, "Creating centroid index...")
    centroid_index = faiss.IndexFlatL2(chunk_embeddings_np.shape[1])
    centroid_index.add(kmeans.cluster_centers_)

    # --- Create Per-Cluster Indices ---
    if progress_bar: progress_bar.progress(0.8, "Creating indices for each cluster...")
    
    # Group chunks by cluster
    clustered_chunks = [[] for _ in range(num_clusters)]
    for i, label in enumerate(kmeans.labels_):
        clustered_chunks[label].append(chunks[i])

    # Create a FAISS index for each cluster
    cluster_indices = {}
    cluster_paths = {}
    for i in range(num_clusters):
        if clustered_chunks[i]:
            cluster_path = os.path.join(PERSIST_DIR, f"cluster_{i}")
            index = LangchainFAISS.from_documents(clustered_chunks[i], embeddings_model)
            cluster_indices[i] = index # Keep the in-memory index for the return value
            index.save_local(cluster_path)
            cluster_paths[i] = cluster_path

    # --- Save components separately ---
    if progress_bar: progress_bar.progress(0.9, "Saving clustered index...")
    
    os.makedirs(PERSIST_DIR, exist_ok=True)
    

    # Save KMeans model separately using joblib
    kmeans_path = os.path.join(PERSIST_DIR, "kmeans.joblib")
    joblib.dump(kmeans, kmeans_path)
    

    # Save FAISS centroid index
    centroid_path = os.path.join(PERSIST_DIR, "centroid.index")
    faiss.write_index(centroid_index, centroid_path)
    
    # Package the rest without the problematic objects
    clustered_index = {
        "num_clusters": num_clusters,
        "cluster_paths": cluster_paths,
        "clusters_to_search": clusters_to_search,
        "top_k": top_k
    }
    
    with open(os.path.join(PERSIST_DIR, PERSIST_FILE), "wb") as f:
        pickle.dump(clustered_index, f)

    if progress_bar: progress_bar.progress(1.0, "Done!")
    return {
        "centroid_index": centroid_index,
        "cluster_indices": cluster_indices,
        "num_clusters": num_clusters,
        "kmeans": kmeans
    }

def load_clustered_index(embeddings_model):
    """Loads the clustered index from disk."""
    try:
        # Load KMeans model
        kmeans_path = os.path.join(PERSIST_DIR, "kmeans.joblib")
        kmeans = joblib.load(kmeans_path)
        
        # Load FAISS centroid index
        centroid_path = os.path.join(PERSIST_DIR, "centroid.index")
        centroid_index = faiss.read_index(centroid_path)
        
        # Load the rest of the components
        with open(os.path.join(PERSIST_DIR, PERSIST_FILE), "rb") as f:
            clustered_index = pickle.load(f)
            

        # Load individual cluster indices
        cluster_indices = {}
        for i, path in clustered_index["cluster_paths"].items():
            cluster_indices[i] = LangchainFAISS.load_local(
                path, embeddings_model, allow_dangerous_deserialization=True
            )

        # Combine all components
        return {
            "centroid_index": centroid_index,
            "cluster_indices": cluster_indices,
            "num_clusters": clustered_index["num_clusters"],
            "kmeans": kmeans,
            "clusters_to_search": clustered_index.get("clusters_to_search", 1), # Default to 1 for old indexes
            "top_k": clustered_index.get("top_k", 3) # Default to 3 for old indexes
        }
    except Exception as e:
        raise Exception(f"Failed to load index: {str(e)}")

def perform_reindex(source="local", uploaded_files=None):
    """Single function to handle all reindexing with clustering."""
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
        
        # Get dynamic settings based on number of chunks
        num_clusters, clusters_to_search, top_k = get_dynamic_settings(len(chunks))
        st.sidebar.info(f"Using {num_clusters} clusters, searching {clusters_to_search}, retrieving {top_k} chunks.")
        
        # Step 3: Create and save clustered index
        embeddings = get_embeddings()
        vectorstore = create_clustered_index(chunks, embeddings, num_clusters, clusters_to_search, top_k, progress)
        
        progress.empty()
        st.sidebar.success(f"✅ Indexed {len(chunks)} chunks into {num_clusters} clusters!")
        return vectorstore
        
    except Exception as e:
        progress.empty()
        st.error(f"Reindex failed: {e}")
        return None

def add_file_to_clusters(uploaded_files, clustered_index):
    """Adds new documents to an existing clustered index."""
    if not uploaded_files:
        st.sidebar.warning("Please upload a file to add.")
        return clustered_index

    with st.spinner(f"Adding {len(uploaded_files)} file(s) to index..."):
        embeddings = get_embeddings()
        docs, tmpdir = load_pdfs_from_uploaded_files(uploaded_files)
        splitter = custom_recursive_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(docs)

        chunk_embeddings = embeddings.embed_documents([c.page_content for c in chunks])
        chunk_embeddings_np = np.array(chunk_embeddings, dtype=np.float32)

        # Find the nearest cluster for each new chunk
        _, labels = clustered_index["centroid_index"].search(chunk_embeddings_np, 1)

        # Add chunks to their respective cluster indices
        for i, chunk in enumerate(chunks):
            cluster_id = labels[i][0]
            if cluster_id in clustered_index["cluster_indices"]:
                clustered_index["cluster_indices"][cluster_id].add_documents([chunk])

        # Save the updated index back to disk
        with open(f"{PERSIST_DIR}/{PERSIST_FILE}", "wb") as f:
            pickle.dump(clustered_index, f)

        st.sidebar.success(f"Added {len(chunks)} new chunks to the index.")
    return clustered_index
