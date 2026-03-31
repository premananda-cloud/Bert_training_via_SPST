"""
build_dataset.py
Builds unified_train.tsv and unified_test.tsv from:
  - GossipCop (fake + real) — scrape news_url, fallback to title — capped at 3000 each
  - PolitiFact (fake + real) — scrape news_url, fallback to title — capped at 500 each
  - LIAR (fake + real)      — use text column directly

Output columns: text, label (0=fake, 1=real)
Output location: datasets/processed/
"""

import os
import time
import pandas as pd
from newspaper import Article
from sklearn.model_selection import train_test_split

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "datasets", "raw")
OUT_DIR  = os.path.join(BASE_DIR, "datasets", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

# Config
TEST_SIZE    = 0.20
RANDOM_SEED  = 42
SCRAPE_DELAY = 0.5

CAPS = {
    "gossipcop": 3000,
    "politifact": 500,
}

def scrape_url(url: str) -> str | None:
    try:
        article = Article(url, fetch_images=False, request_timeout=10)
        article.download()
        article.parse()
        text = article.text.strip()
        return text if len(text) > 50 else None
    except Exception:
        return None

def fetch_text(row) -> str:
    scraped = scrape_url(str(row["news_url"]))
    if scraped:
        return scraped
    title = str(row.get("title", "")).strip()
    return title if title else ""

def load_scraped(filepath: str, label: int, cap: int) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t")
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df = df.head(cap)
    records = []
    total = len(df)
    print(f"  Scraping {total} rows from {os.path.basename(filepath)} (cap={cap}) ...")
    for i, row in df.iterrows():
        text = fetch_text(row)
        if text:
            records.append({"text": text, "label": label})
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{total} done")
        time.sleep(SCRAPE_DELAY)
    result = pd.DataFrame(records)
    print(f"  -> Collected {len(result)} valid rows")
    return result

def load_liar(filepath: str, label: int) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t")
    df = df[["text"]].copy()
    df = df[df["text"].notna() & (df["text"].str.strip() != "")]
    df["label"] = label
    print(f"  -> Loaded {len(df)} rows from {os.path.basename(filepath)}")
    return df[["text", "label"]]

def main():
    chunks = []

    print("\n[GossipCop]")
    chunks.append(load_scraped(os.path.join(RAW_DIR, "gossipcop_fake.tsv"), label=0, cap=CAPS["gossipcop"]))
    chunks.append(load_scraped(os.path.join(RAW_DIR, "gossipcop_real.tsv"), label=1, cap=CAPS["gossipcop"]))

    print("\n[PolitiFact]")
    chunks.append(load_scraped(os.path.join(RAW_DIR, "politifact_fake.tsv"), label=0, cap=CAPS["politifact"]))
    chunks.append(load_scraped(os.path.join(RAW_DIR, "politifact_real.tsv"), label=1, cap=CAPS["politifact"]))

    print("\n[LIAR]")
    chunks.append(load_liar(os.path.join(RAW_DIR, "liar_fake", "Fake.tsv"), label=0))
    chunks.append(load_liar(os.path.join(RAW_DIR, "liar_real", "True.tsv"), label=1))

    unified = pd.concat(chunks, ignore_index=True)
    unified = unified[unified["text"].str.strip() != ""].reset_index(drop=True)
    unified = unified.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"\nTotal samples: {len(unified)}")
    print(f"  Fake (0): {(unified['label'] == 0).sum()}")
    print(f"  Real (1): {(unified['label'] == 1).sum()}")

    train_df, test_df = train_test_split(
        unified,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=unified["label"]
    )

    train_path = os.path.join(OUT_DIR, "unified_train.tsv")
    test_path  = os.path.join(OUT_DIR, "unified_test.tsv")
    train_df.to_csv(train_path, sep="\t", index=False)
    test_df.to_csv(test_path,  sep="\t", index=False)

    print(f"\nSaved:")
    print(f"  Train : {len(train_df)} rows -> {train_path}")
    print(f"  Test  : {len(test_df)} rows -> {test_path}")

if __name__ == "__main__":
    main()
