"""
unify_dataset.py
Cleans, standardises, and unifies all scraped + LIAR datasets.
Outputs: unified_train.tsv, unified_val.tsv, unified_test.tsv
Split: 80/10/10 stratified
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
PROC_DIR  = os.path.join(BASE_DIR, "datasets", "processed")
RAW_DIR   = os.path.join(BASE_DIR, "datasets", "raw")

SCRAPED_FILES = [
    (os.path.join(PROC_DIR, "gossipcop_fake_scraped.tsv"),  0),
    (os.path.join(PROC_DIR, "gossipcop_real_scraped.tsv"),  1),
    (os.path.join(PROC_DIR, "politifact_fake_scraped.tsv"), 0),
    (os.path.join(PROC_DIR, "politifact_real_scraped.tsv"), 1),
]

LIAR_FAKE = os.path.join(RAW_DIR, "liar_fake", "Fake.tsv")
LIAR_REAL = os.path.join(RAW_DIR, "liar_real", "True.tsv")

MIN_LEN      = 50
RANDOM_SEED  = 42

# ── Cleaning ──────────────────────────────────────────────────────────────────

def is_mostly_english(text: str) -> bool:
    """Reject text where more than 30% of chars are non-ASCII."""
    if not text:
        return False
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return (non_ascii / len(text)) < 0.30

def is_whitespace_or_punct(text: str) -> bool:
    """True if text has no real word characters."""
    return not bool(re.search(r'[a-zA-Z]{3,}', text))

def clean_text(text: str) -> str:
    text = str(text).strip()
    # collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text)
    # remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()

def is_valid(text: str) -> bool:
    if len(text) < MIN_LEN:
        return False
    if is_whitespace_or_punct(text):
        return False
    if not is_mostly_english(text):
        return False
    return True

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_scraped(filepath: str, label: int) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t")
    df = df[["text", "label"]].copy()
    return df

def load_liar(filepath: str, label: int) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t")
    # join title + text
    title = df["title"].fillna("").astype(str).str.strip()
    text  = df["text"].fillna("").astype(str).str.strip()
    # combine: "TITLE. TEXT" — skip if both empty
    combined = (title + ". " + text).str.strip(". ").str.strip()
    out = pd.DataFrame({"text": combined, "label": label})
    return out

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    chunks = []

    # Load scraped sources
    for filepath, label in SCRAPED_FILES:
        df = load_scraped(filepath, label)
        print(f"Loaded {len(df):>5} rows from {os.path.basename(filepath)}")
        chunks.append(df)

    # Load LIAR
    liar_fake = load_liar(LIAR_FAKE, label=0)
    liar_real = load_liar(LIAR_REAL, label=1)
    print(f"Loaded {len(liar_fake):>5} rows from liar_fake/Fake.tsv")
    print(f"Loaded {len(liar_real):>5} rows from liar_real/True.tsv")
    chunks += [liar_fake, liar_real]

    # Concat
    unified = pd.concat(chunks, ignore_index=True)
    print(f"\nTotal before cleaning: {len(unified)}")

    # Clean
    unified["text"] = unified["text"].apply(clean_text)
    before = len(unified)
    unified = unified[unified["text"].apply(is_valid)].reset_index(drop=True)
    print(f"Dropped (garbage/short/non-english): {before - len(unified)}")

    # Deduplicate
    before = len(unified)
    unified = unified.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Dropped (duplicates): {before - len(unified)}")

    print(f"Total after cleaning: {len(unified)}")
    print(f"  Fake (0): {(unified['label'] == 0).sum()}")
    print(f"  Real (1): {(unified['label'] == 1).sum()}")

    # Shuffle
    unified = unified.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Split 80/10/10 stratified
    train_df, temp_df = train_test_split(
        unified, test_size=0.20, random_state=RANDOM_SEED, stratify=unified["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["label"]
    )

    # Save
    train_path = os.path.join(PROC_DIR, "unified_train.tsv")
    val_path   = os.path.join(PROC_DIR, "unified_val.tsv")
    test_path  = os.path.join(PROC_DIR, "unified_test.tsv")

    train_df.to_csv(train_path, sep="\t", index=False)
    val_df.to_csv(val_path,     sep="\t", index=False)
    test_df.to_csv(test_path,   sep="\t", index=False)

    print(f"\nSaved:")
    print(f"  Train : {len(train_df)} rows -> {train_path}")
    print(f"  Val   : {len(val_df)} rows -> {val_path}")
    print(f"  Test  : {len(test_df)} rows -> {test_path}")

if __name__ == "__main__":
    main()
