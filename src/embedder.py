

import numpy as np
import pandas as pd
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Optional


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:

    try:
        model = SentenceTransformer(model_name)
        print(f"Loaded embedding model: {model_name}")
        print(
            f"Embedding dimensions: {model.get_sentence_embedding_dimension()}")
        return model
    except Exception as e:
        raise OSError(
            f"Failed to load model '{model_name}'.\n"
            f"Check your internet connection or model name.\n"
            f"Original error: {e}"
        )


def generate_embeddings(
    chunks_df: pd.DataFrame,
    model: SentenceTransformer,
    text_col: str = 'chunk_text',
    batch_size: int = 64
) -> np.ndarray:

    if text_col not in chunks_df.columns:
        raise KeyError(f"Column '{text_col}' not found in chunks DataFrame.")

    if len(chunks_df) == 0:
        raise ValueError("chunks_df is empty — nothing to embed.")

    texts = chunks_df[text_col].tolist()
    print(f"Embedding {len(texts):,} chunks in batches of {batch_size}...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True   # L2 normalise for cosine similarity via IndexFlatIP
    )

    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:

    if embeddings.shape[0] == 0:
        raise ValueError("Embeddings array is empty — cannot build index.")

    if embeddings.dtype != np.float32:
        raise ValueError(
            f"Embeddings must be float32, got {embeddings.dtype}.\n"
            f"Call .astype(np.float32) before passing to build_faiss_index()."
        )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"FAISS index built — {index.ntotal:,} vectors, {dim} dimensions")
    return index


def save_vector_store(
    index: faiss.IndexFlatIP,
    chunks_df: pd.DataFrame,
    out_dir: str = '../vector_store'
) -> None:

    if index.ntotal != len(chunks_df):
        raise ValueError(
            f"Index has {index.ntotal:,} vectors but chunks_df has "
            f"{len(chunks_df):,} rows. They must match exactly."
        )

    os.makedirs(out_dir, exist_ok=True)

    index_path = os.path.join(out_dir, 'faiss_index.bin')
    metadata_path = os.path.join(out_dir, 'chunk_metadata.pkl')

    faiss.write_index(index, index_path)
    print(f"FAISS index saved  : {index_path}")

    with open(metadata_path, 'wb') as f:
        pickle.dump(chunks_df, f)
    print(f"Metadata saved     : {metadata_path}")


def load_vector_store(
    out_dir: str = '../vector_store'
) -> tuple:

    index_path = os.path.join(out_dir, 'faiss_index.bin')
    metadata_path = os.path.join(out_dir, 'chunk_metadata.pkl')

    for path in [index_path, metadata_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Vector store file not found: {path}\n"
                f"Run the embedding notebook first to build the index."
            )

    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        chunks_df = pickle.load(f)

    print(f"Loaded FAISS index : {index.ntotal:,} vectors")
    print(f"Loaded metadata    : {len(chunks_df):,} chunks")
    return index, chunks_df
