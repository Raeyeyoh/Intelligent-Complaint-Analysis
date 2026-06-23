"""
chunker.py
----------
Handles stratified sampling and text chunking of cleaned complaint narratives.

Design choice: RecursiveCharacterTextSplitter is used over a simple
split because it tries to break on natural boundaries (paragraphs,
sentences, words) before falling back to characters — preserving
semantic coherence within each chunk.

Design choice: chunk_size=500, chunk_overlap=50 matches the pre-built
vector store specification, ensuring our sample index is comparable
to the full dataset index used in Tasks 3-4.
"""

import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Optional


def stratified_sample(
    df: pd.DataFrame,
    n: int = 12000,
    category_col: str = 'product_category',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Draw a stratified sample from the cleaned complaints dataset,
    preserving the proportional representation of each product category.

    Design choice: Stratified sampling is critical here because the
    product categories are heavily imbalanced. A random sample would
    under-represent minority categories like Money Transfer, producing
    a vector store that performs poorly on those queries.

    Args:
        df:            Cleaned complaints DataFrame.
        n:             Total sample size (10,000–15,000 recommended).
        category_col:  Column to stratify on.
        random_state:  Seed for reproducibility.

    Returns:
        Stratified sample DataFrame, reset index.

    Raises:
        KeyError:  If category_col is not in the DataFrame.
        ValueError: If n exceeds the dataset size.
    """
    if category_col not in df.columns:
        raise KeyError(
            f"Column '{category_col}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if n > len(df):
        raise ValueError(
            f"Requested sample size ({n:,}) exceeds dataset size ({len(df):,})."
        )

    category_counts = df[category_col].value_counts()
    fractions = (category_counts / len(df))

    sample_parts = []
    for category, frac in fractions.items():
        category_n = max(1, round(frac * n))
        subset = df[df[category_col] == category]
        category_sample = subset.sample(
            n=min(category_n, len(subset)),
            random_state=random_state
        )
        sample_parts.append(category_sample)

    sample = pd.concat(sample_parts).sample(frac=1, random_state=random_state)
    sample = sample.reset_index(drop=True)

    print(f"Stratified sample: {len(sample):,} rows")
    print("\nSample distribution:")
    print(sample[category_col].value_counts())
    print("\nOriginal distribution (%):")
    print((category_counts / len(df) * 100).round(1))
    print("\nSample distribution (%):")
    print((sample[category_col].value_counts() / len(sample) * 100).round(1))

    return sample


def build_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> RecursiveCharacterTextSplitter:
    """
    Build a LangChain RecursiveCharacterTextSplitter.

    Design choice: chunk_size=500 characters keeps chunks short enough
    to carry a focused semantic meaning, while chunk_overlap=50 ensures
    context is not lost at chunk boundaries — a sentence split across
    two chunks will still be retrievable from either side.

    Args:
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks in characters.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )
    print(
        f"Splitter built — chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    return splitter


def chunk_complaints(
    df: pd.DataFrame,
    splitter: RecursiveCharacterTextSplitter,
    text_col: str = 'cleaned_narrative'
) -> pd.DataFrame:
    """
    Split each complaint narrative into chunks and attach metadata.

    Each output row represents one chunk with:
    - chunk_text: the text content of the chunk
    - chunk_index: position of this chunk within the complaint
    - total_chunks: total chunks produced from this complaint
    - All original metadata columns (complaint_id, product_category, etc.)

    Design choice: Metadata is attached per chunk (not per complaint)
    so that retrieved chunks can be traced directly back to their source
    complaint and product without a secondary lookup.

    Args:
        df:       Stratified sample DataFrame with cleaned narratives.
        splitter: Configured RecursiveCharacterTextSplitter.
        text_col: Column containing the narrative text to chunk.

    Returns:
        DataFrame of chunks with metadata — one row per chunk.

    Raises:
        KeyError: If text_col is not in the DataFrame.
    """
    if text_col not in df.columns:
        raise KeyError(
            f"Column '{text_col}' not found.\n"
            f"Run apply_cleaning() first to generate this column."
        )

    metadata_cols = [c for c in df.columns if c != text_col]
    rows = []

    for _, record in df.iterrows():
        text = record[text_col]

        if not isinstance(text, str) or len(text.strip()) == 0:
            continue

        chunks = splitter.split_text(text)
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            row = {col: record[col] for col in metadata_cols}
            row['chunk_text'] = chunk
            row['chunk_index'] = i
            row['total_chunks'] = total
            rows.append(row)

    chunks_df = pd.DataFrame(rows)
    print(f"Produced {len(chunks_df):,} chunks from {len(df):,} complaints")
    print(f"Avg chunks per complaint: {len(chunks_df)/len(df):.1f}")
    return chunks_df
