"""
scrape_gossipcop_fake.py
Scrapes GossipCop fake articles, falls back to title on failure.
Output: datasets/processed/gossipcop_fake_scraped.tsv  (text, label)
"""

import os
import time
import signal
import pandas as pd
from newspaper import Article

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
INPUT     = os.path.join(BASE_DIR, "datasets", "raw", "gossipcop_fake.tsv")
OUT_DIR   = os.path.join(BASE_DIR, "datasets", "processed")
OUTPUT    = os.path.join(OUT_DIR, "gossipcop_fake_scraped.tsv")
os.makedirs(OUT_DIR, exist_ok=True)

CAP          = 3000
LABEL        = 0
SCRAPE_DELAY = 0.3
TIMEOUT      = 15   # hard per-URL timeout in seconds


class TimeoutError(Exception):
    pass

def _handler(signum, frame):
    raise TimeoutError()

def scrape_url(url: str) -> str | None:
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(TIMEOUT)
    try:
        article = Article(url, fetch_images=False, request_timeout=TIMEOUT)
        article.download()
        article.parse()
        text = article.text.strip()
        signal.alarm(0)
        return text if len(text) > 50 else None
    except Exception:
        signal.alarm(0)
        return None


def main():
    df = pd.read_csv(INPUT, sep="\t")
    df = df.head(CAP)
    total = len(df)
    records = []

    print(f"Processing {total} rows from gossipcop_fake.tsv ...")
    for i, row in df.iterrows():
        url   = str(row.get("news_url", ""))
        title = str(row.get("title", "")).strip()
        text  = scrape_url(url)
        used  = "scraped" if text else "title"
        if not text:
            text = title
        if text:
            records.append({"text": text, "label": LABEL})
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{total} done  (last: {used})")
        time.sleep(SCRAPE_DELAY)

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT, sep="\t", index=False)
    print(f"\nSaved {len(out_df)} rows -> {OUTPUT}")

if __name__ == "__main__":
    main()
