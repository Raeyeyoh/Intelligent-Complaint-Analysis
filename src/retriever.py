

import numpy as np
import pandas as pd
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from typing import Optional
from embedder import EMBEDDING_MODEL, load_embedding_model


def load_retriever(
    vector_store_dir: str = '../vector_store',
    model_name: str = EMBEDDING_MODEL,
) -> tuple:
    
    index_path    = os.path.join(vector_store_dir, 'faiss_index.bin')
    metadata_path = os.path.join(vector_store_dir, 'chunk_metadata.pkl')

    for path in [index_path, metadata_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Vector store file not found: {path}\n"
                f"Run chunking-embedding.ipynb first to build the index."
            )

    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        chunks_df = pickle.load(f)

    model = load_embedding_model(model_name)

    print(f"Retriever ready — {index.ntotal:,} vectors loaded")
    return index, chunks_df, model


def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    chunks_df: pd.DataFrame,
    model: SentenceTransformer,
    k: int = 5,
    product_filter: Optional[str] = None,
) -> list[dict]:
  
    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}.")

    query_embedding = model.encode(
        [query.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    fetch_k = k * 10 if product_filter else k
    fetch_k = min(fetch_k, index.ntotal)

    distances, indices = index.search(query_embedding, fetch_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        chunk = chunks_df.iloc[idx]

        if product_filter and chunk.get('product_category') != product_filter:
            continue

        results.append({
            'chunk_text':       chunk['chunk_text'],
            'product_category': chunk.get('product_category', 'Unknown'),
            'complaint_id':     chunk.get('complaint_id', 'Unknown'),
            'issue':            chunk.get('issue', 'Unknown'),
            'score':            float(dist),
            'rank':             len(results) + 1,
        })

        if len(results) == k:
            break

    if len(results) == 0:
        print(f"Warning: No results found for query: '{query}'")

    return results


def format_context(retrieved_chunks: list[dict]) -> str:
    
    if not retrieved_chunks:
        return "No relevant complaint excerpts found."

    parts = []
    for chunk in retrieved_chunks:
        parts.append(
            f"[Source {chunk['rank']} | "
            f"Product: {chunk['product_category']} | "
            f"Issue: {chunk['issue']} | "
            f"Complaint ID: {chunk['complaint_id']}]\n"
            f"{chunk['chunk_text']}"
        )

    return "\n\n---\n\n".join(parts)