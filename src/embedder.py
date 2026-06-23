"""
embedder.py
-----------
Handles embedding generation and FAISS vector store creation.

Design choice: all-MiniLM-L6-v2 is chosen because:
- It is fast and lightweight (80MB, 384 dimensions)
- It is optimised for semantic similarity tasks — exactly what
  retrieval in a RAG pipeline requires
- It matches the pre-built vector store embedding model used in
  Tasks 3-4, ensuring our sample index is directly comparable
- It runs on CPU without significant performance degradation

Design choice: FAISS IndexFlatIP (inner product) is used with
normalised vectors, which is equivalent to cosine similarity.
Cosine similarity is preferred over L2 distance for text embeddings
because it is invariant to vector magnitude — two complaints of
very different lengths should still be comparable by topic.
"""

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
    """
    Load the sentence-transformers embedding model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded SentenceTransformer model.

    Raises:
        OSError: If the model cannot be downloaded or loaded.
    """
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
    """
    Generate embeddings for all text chunks in batches.

    Design choice: Batched encoding is used rather than encoding all
    chunks at once to avoid memory errors on large datasets. batch_size=64
    is a good default for CPU; increase to 128-256 on GPU.

    Args:
        chunks_df:  DataFrame of chunks from chunk_complaints().
        model:      Loaded SentenceTransformer model.
        text_col:   Column containing chunk text.
        batch_size: Number of chunks to encode per batch.

    Returns:
        numpy array of shape (n_chunks, embedding_dim), float32, L2-normalised.

    Raises:
        KeyError:   If text_col is not in chunks_df.
        ValueError: If chunks_df is empty.
    """
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
    """
    Build a FAISS IndexFlatIP index from normalised embeddings.

    Design choice: IndexFlatIP (inner product) on L2-normalised vectors
    is mathematically equivalent to cosine similarity search. It is
    exact (no approximation) which is acceptable at the 10K-15K scale
    of the sample. For the full 464K complaint dataset, an approximate
    index (IndexIVFFlat or IndexHNSW) would be more appropriate.

    Args:
        embeddings: float32 numpy array of shape (n, dim), L2-normalised.

    Returns:
        Populated faiss.IndexFlatIP index.

    Raises:
        ValueError: If embeddings array is empty or wrong dtype.
    """
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
    """
    Persist the FAISS index and chunk metadata to disk.

    Saves two files:
    - faiss_index.bin   : The FAISS index (binary format)
    - chunk_metadata.pkl: The chunks DataFrame with all metadata

    Design choice: Metadata is stored separately from the FAISS index
    because FAISS only stores vectors — it has no native support for
    arbitrary metadata. The chunk position in the DataFrame corresponds
    directly to its FAISS index ID, so retrieval by ID maps directly
    to the correct metadata row.

    Args:
        index:     Populated FAISS index.
        chunks_df: DataFrame of chunks with metadata.
        out_dir:   Directory to save files to.

    Raises:
        ValueError: If index and chunks_df have mismatched counts.
    """
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
    """
    Load a persisted FAISS index and chunk metadata from disk.

    Args:
        out_dir: Directory where the vector store was saved.

    Returns:
        Tuple of (faiss_index, chunks_df).

    Raises:
        FileNotFoundError: If either file is missing.
    """
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
