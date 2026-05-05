# BERT + SPST Fake News Detection

> **97.52% accuracy · 97.71% F1 · 0.9981 AUC**  
> Resource-efficient BERT fine-tuning via Sequential Parameter Segment Training (SPST)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qdwF6kxTwKsSB89Nh0ymQhrRtcjVdPoI#scrollTo=aSB5Mr0fh0pR)

---

## Overview

This project fine-tunes `bert-base-uncased` for binary fake news classification (fake=0, real=1) using **SPST — Sequential Parameter Segment Training**: a progressive layer-unfreezing strategy that reduces peak GPU memory by only training one layer group at a time, then unlocking the next.

Training on a free Colab T4 GPU went from a ~75% accuracy baseline (standard fine-tuning) to **97.52%** with SPST.

---

## Results

| Metric    | Score   |
|-----------|---------|
| Accuracy  | 97.52%  |
| Precision | 97.45%  |
| Recall    | 97.98%  |
| F1 Score  | 97.71%  |
| AUC-ROC   | 0.9981  |
| Test Loss | 0.0586  |

Full training history, plots (loss curves, F1 curves, confusion matrix, ROC curve), model weights, and `results.json` are available on Google Drive:  
📁 [Results & Model Weights](https://drive.google.com/drive/folders/1zAD6Q5RxQ-YGfiV455_DpG3_Wd2sgLkW?usp=sharing)

---

## Method: SPST (Progressive Unfreezing)

The BERT model is divided into 4 layer groups, trained sequentially:

| Segment           | Layers Trained              | Epochs | Learning Rate |
|-------------------|-----------------------------|--------|---------------|
| `classifier_only` | Classifier + Pooler         | 2      | 3e-4          |
| `top_layers`      | + Encoder layers 10–11      | 2      | 1e-4          |
| `mid_layers`      | + Encoder layers 7–9        | 2      | 5e-5          |
| `full_model`      | All layers                  | 2      | 2e-5          |

Additional optimisations: FP16 mixed precision, gradient checkpointing, gradient accumulation (effective batch size 64).

---

## Dataset

Three sources unified into a single corpus:

| Dataset     | Fake | Real | Text Source                        |
|-------------|------|------|------------------------------------|
| GossipCop   | 3000 | 3000 | Scraped article / title fallback   |
| PolitiFact  | 500  | 500  | Scraped article / title fallback   |
| LIAR        | ~6300| ~6300| Concatenated title + statement     |

Split: **80% train / 10% validation / 10% test** (stratified).

📦 [Download Dataset ZIP](https://drive.google.com/file/d/1BfIvNcVJQIx8-KAqWFavy-htKcjTbMTb/view?usp=sharing)

---

## Repository Structure

```
├── datasets/
│   ├── raw/                        # Original source TSVs
│   │   ├── gossipcop_fake.tsv
│   │   ├── gossipcop_real.tsv
│   │   ├── politifact_fake.tsv
│   │   ├── politifact_real.tsv
│   │   ├── liar_fake/Fake.tsv
│   │   └── liar_real/True.tsv
│   └── processed/                  # Cleaned, unified splits
│       ├── unified_train.tsv
│       ├── unified_val.tsv
│       └── unified_test.tsv
├── scrape_gossipcop_fake.py        # Per-source scrapers
├── scrape_gossipcop_real.py
├── scrape_politifact_fake.py
├── scrape_politifact_real.py
├── unify_dataset.py                # Clean + unify + split
├── train/
│   └── train_bert.py               # Training script (local)
├── train_bert_colab.ipynb          # Colab notebook (recommended)
├── config.py
└── requirements.txt
```

---

## Reproduce

### 1. Clone & install

```bash
git clone https://github.com/premananda-cloud/Bert_training_via_SPST
cd Bert_training_via_SPST
pip install -r requirements.txt
```

### 2. Get the dataset

Download the [dataset ZIP](https://drive.google.com/file/d/1BfIvNcVJQIx8-KAqWFavy-htKcjTbMTb/view?usp=sharing) and unzip it into the project root so `datasets/processed/unified_train.tsv` exists.  

Or re-scrape from scratch:

```bash
# Run each scraper independently (they're slow — run in parallel terminals)
python scrape_gossipcop_fake.py
python scrape_gossipcop_real.py
python scrape_politifact_fake.py
python scrape_politifact_real.py

# Then clean, unify, and split
python unify_dataset.py
```

### 3. Train on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qdwF6kxTwKsSB89Nh0ymQhrRtcjVdPoI#scrollTo=aSB5Mr0fh0pR)

1. Open the notebook in Colab
2. Set runtime to **GPU (T4)**: Runtime → Change runtime type
3. Run all cells — the notebook mounts Drive, downloads the dataset, trains, saves weights + plots to `My Drive/spst/`, and downloads a results ZIP

---

## Requirements

```
torch
transformers==4.40.0
scikit-learn
pandas
numpy
newspaper3k
lxml_html_clean
matplotlib
seaborn
tqdm
```

---

## Links

| Resource | Link |
|----------|------|
| Colab Notebook | https://colab.research.google.com/drive/1iKxeyO9S4c6nVdNyFzsxoybEWTLk2MB2?usp=drive_link |
| Dataset ZIP | https://drive.google.com/file/d/1BfIvNcVJQIx8-KAqWFavy-htKcjTbMTb/view?usp=sharing |
| Model Weights & Results | https://drive.google.com/drive/folders/1zAD6Q5RxQ-YGfiV455_DpG3_Wd2sgLkW?usp=sharing |
