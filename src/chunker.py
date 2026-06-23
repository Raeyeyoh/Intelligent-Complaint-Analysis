

import pandas as pd
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Optional


def stratified_sample(
    df: pd.DataFrame,
    n: int = 12000,
    category_col: str = 'product_category',
    random_state: int = 42
) -> pd.DataFrame:

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
