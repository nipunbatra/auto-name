"""
Data preparation for next-character name generation.
Downloads Indian names dataset, builds char vocabulary, creates train/val split.

Usage:
    python prepare.py

This file is READ-ONLY during experiments. Do not modify.
"""

import os
import requests
import pandas as pd
import torch
import torch.nn.functional as F
import json

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_URL = "https://raw.githubusercontent.com/balasahebgulave/Dataset-Indian-Names/master/Indian_Names.csv"
BLOCK_SIZE_MAX = 8       # max context length any experiment can use
TIME_BUDGET = 120        # training time budget in seconds (2 minutes)
VAL_FRACTION = 0.1       # 10% of data for validation
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Data download and preparation
# ---------------------------------------------------------------------------

def download_and_prepare():
    """Download names, clean, split into train/val, save."""
    os.makedirs(DATA_DIR, exist_ok=True)

    csv_path = os.path.join(DATA_DIR, "names.csv")
    if not os.path.exists(csv_path):
        print("Downloading Indian Names dataset...")
        response = requests.get(DATA_URL)
        response.raise_for_status()
        with open(csv_path, "w") as f:
            f.write(response.text)
        print(f"  Saved to {csv_path}")

    # Load and clean
    words = pd.read_csv(csv_path)["Name"]
    words = words.str.lower().str.strip().str.replace(" ", "", regex=False)
    words = words[words.str.len().between(3, 9)]
    words = words[words.apply(lambda w: w.isalpha())]
    words = words.drop_duplicates()
    words = words.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True).tolist()

    print(f"Total unique names: {len(words)}")
    print(f"Examples: {words[:10]}")

    # Build character vocabulary
    chars = sorted(set("".join(words)))
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0  # special token for start/end
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size} characters")

    # Train/val split
    n_val = int(len(words) * VAL_FRACTION)
    val_words = words[:n_val]
    train_words = words[n_val:]
    print(f"Train: {len(train_words)} names, Val: {len(val_words)} names")

    # Save everything
    meta = {
        "stoi": stoi,
        "itos": {int(k): v for k, v in itos.items()},
        "vocab_size": vocab_size,
        "n_train": len(train_words),
        "n_val": len(val_words),
    }
    with open(os.path.join(DATA_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(DATA_DIR, "train.txt"), "w") as f:
        f.write("\n".join(train_words))

    with open(os.path.join(DATA_DIR, "val.txt"), "w") as f:
        f.write("\n".join(val_words))

    print(f"Saved to {DATA_DIR}/")
    return train_words, val_words, stoi, itos, vocab_size


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_meta():
    """Load vocabulary metadata."""
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    # Convert itos keys back to int
    meta["itos"] = {int(k): v for k, v in meta["itos"].items()}
    return meta


def load_words(split):
    """Load train or val word list."""
    path = os.path.join(DATA_DIR, f"{split}.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def build_dataset(words, block_size, stoi, device="cpu"):
    """Build (context, target) pairs from words."""
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context[:])
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X, device=device)
    Y = torch.tensor(Y, device=device)
    return X, Y


@torch.no_grad()
def evaluate(model, X_val, Y_val, batch_size=4096):
    """Compute average cross-entropy loss on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(X_val), batch_size):
        logits = model(X_val[i:i + batch_size])
        loss = F.cross_entropy(logits, Y_val[i:i + batch_size])
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / n_batches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    download_and_prepare()
    print("\nDone! Ready to train with: python train.py")
