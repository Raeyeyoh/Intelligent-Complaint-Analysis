

import pandas as pd
import re
import os
from typing import Optional


PRODUCT_MAP = {
    'Credit card':                               'Credit Card',
    'Credit Card':                               'Credit Card',
    'credit card':                               'Credit Card',
    'Prepaid card':                              'Credit Card',

    'Personal loan':                             'Personal Loan',
    'Personal Loan':                             'Personal Loan',
    'Consumer Loan':                             'Personal Loan',
    'Payday loan':                               'Personal Loan',
    'Payday loan, title loan, or personal loan': 'Personal Loan',
    'Student loan':                              'Personal Loan',

    'Savings account':                           'Savings Account',
    'Bank account or service':                   'Savings Account',
    'Checking or savings account':               'Savings Account',

    'Money transfer':                            'Money Transfer',
    'Money transfer, virtual currency, or money service': 'Money Transfer',
    'Money transfers':                           'Money Transfer',
    'Virtual currency':                          'Money Transfer',
}

BOILERPLATE_PATTERNS = [
    r"i am writing to file a complaint",
    r"i am writing to complain",
    r"to whom it may concern",
    r"dear cfpb",
    r"dear consumer financial protection bureau",
    r"i am filing this complaint",
    r"i would like to file a complaint",
    r"this is a complaint",
    r"i am submitting this complaint",
]

RENAME_MAP = {
    'Complaint ID':  'complaint_id',
    'Date received': 'date_received',
    'Product':       'product',
    'Issue':         'issue',
    'Sub-issue':     'sub_issue',
    'Company':       'company',
    'State':         'state',
}

KEEP_COLS = [
    'Complaint ID',
    'Date received',
    'Product',
    'product_category',
    'Issue',
    'Sub-issue',
    'Company',
    'State',
    'cleaned_narrative',
]


def map_products(df: pd.DataFrame,
                 product_col: str = 'Product',
                 product_map: Optional[dict] = None) -> pd.DataFrame:

    if product_col not in df.columns:
        raise KeyError(
            f"Column '{product_col}' not found.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    mapping = product_map or PRODUCT_MAP
    df = df.copy()
    df['product_category'] = df[product_col].map(mapping)

    print("Mapped product counts (including NaN = unmapped):")
    print(df['product_category'].value_counts(dropna=False))
    return df


def filter_complaints(df: pd.DataFrame,
                      narrative_col: str = 'Consumer complaint narrative',
                      min_narrative_len: int = 20) -> pd.DataFrame:

    for col in ['product_category', narrative_col]:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame.")

    df_out = df[df['product_category'].notna()].copy()
    print(f"After product filter   : {len(df_out):,} rows")

    df_out = df_out[df_out[narrative_col].notna()].copy()
    print(f"After narrative filter : {len(df_out):,} rows")

    if len(df_out) == 0:
        raise ValueError(
            "All rows were removed after filtering. "
            "Check that PRODUCT_MAP matches your dataset's product names."
        )

    print("\nProduct distribution after filtering:")
    print(df_out['product_category'].value_counts())
    return df_out


def clean_narrative(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.lower()

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = re.sub(r'x{2,}', '', text)

    text = re.sub(r"[^a-z0-9\s.,!?'\-]", ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def apply_cleaning(df: pd.DataFrame,
                   narrative_col: str = 'Consumer complaint narrative',
                   min_len: int = 20) -> pd.DataFrame:

    if narrative_col not in df.columns:
        raise KeyError(f"Column '{narrative_col}' not found in DataFrame.")

    df = df.copy()
    df['cleaned_narrative'] = df[narrative_col].apply(clean_narrative)

    before = len(df)
    df = df[df['cleaned_narrative'].str.len() > min_len].copy()
    dropped = before - len(df)

    print(
        f"After cleaning: {len(df):,} rows remain ({dropped} dropped — too short after cleaning)")
    return df


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing columns required for standardisation: {missing}\n"
            f"Ensure map_products(), filter_complaints(), and apply_cleaning() "
            f"have all been run first."
        )

    df_out = df[KEEP_COLS].rename(columns=RENAME_MAP).reset_index(drop=True)
    print(f"Final shape: {df_out.shape}")
    return df_out


def save_dataset(df: pd.DataFrame,
                 out_path: str = '../data/processed/filtered_complaints.csv') -> None:

    if len(df) == 0:
        raise ValueError("Cannot save an empty DataFrame.")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df):,} rows to {out_path}")
