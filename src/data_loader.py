
import pandas as pd
import os


def load_complaints(path: str) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Download the CFPB dataset and place it in data/raw/"
        )

    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows and {df.shape[1]} columns from {path}")
    return df


def inspect_nulls(df: pd.DataFrame) -> pd.DataFrame:

    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100).round(2)

    summary = pd.DataFrame({
        'null_count': null_counts,
        'null_%':     null_pct
    }).query('null_count > 0').sort_values('null_count', ascending=False)

    return summary


def narrative_availability(df: pd.DataFrame,
                           narrative_col: str = 'Consumer complaint narrative') -> dict:

    if narrative_col not in df.columns:
        raise KeyError(
            f"Column '{narrative_col}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    has_narrative = int(df[narrative_col].notna().sum())
    no_narrative = int(df[narrative_col].isna().sum())
    narrative_pct = round(has_narrative / len(df) * 100, 2)

    print(f"With narrative   : {has_narrative:,}  ({narrative_pct:.1f}%)")
    print(f"Without narrative: {no_narrative:,}  ({100 - narrative_pct:.1f}%)")

    return {
        'has_narrative': has_narrative,
        'no_narrative':  no_narrative,
        'narrative_pct': narrative_pct,
    }


def word_count_stats(df: pd.DataFrame,
                     narrative_col: str = 'Consumer complaint narrative') -> pd.DataFrame:

    if narrative_col not in df.columns:
        raise KeyError(f"Column '{narrative_col}' not found in DataFrame.")

    df_text = df[df[narrative_col].notna()].copy()
    df_text['word_count'] = df_text[narrative_col].str.split().str.len()

    stats = df_text['word_count'].describe().round(1)
    short = int((df_text['word_count'] < 10).sum())
    long = int((df_text['word_count'] > 500).sum())

    print("=== Narrative Word Count Stats ===")
    print(stats)
    print(f"\nVery short (<10 words) : {short:,}")
    print(f"Very long  (>500 words): {long:,}")

    return df_text[['word_count']]
