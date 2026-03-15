"""
Streamlit app for next-character name generation.
Shows all experiment results and lets you generate names with each variant.

Usage:
    streamlit run app.py
"""

import os
import csv
import json
import torch
import torch.nn as nn
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_TSV = os.path.join(BASE_DIR, "results.tsv")
PLOT_PATH = os.path.join(BASE_DIR, "experiments_plot.png")


# --- Model definition (must match train.py) ---

class CharModel(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim, hidden_dim, n_hidden, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.emb = nn.Embedding(vocab_size, emb_dim)
        input_dim = block_size * emb_dim

        if hidden_dim > 0 and n_hidden > 0:
            layers = []
            in_d = input_dim
            for _ in range(n_hidden):
                layers.append(nn.Linear(in_d, hidden_dim))
                layers.append(nn.ReLU())
                in_d = hidden_dim
            layers.append(nn.Linear(hidden_dim, vocab_size))
            self.net = nn.Sequential(*layers)
        else:
            self.net = nn.Linear(input_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = x.view(x.shape[0], -1)
        return self.net(x)


def load_meta():
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    meta["itos"] = {int(k): v for k, v in meta["itos"].items()}
    return meta


def generate_names(model, block_size, itos, n=10, temperature=1.0):
    model.eval()
    names = []
    with torch.no_grad():
        for _ in range(n):
            context = [0] * block_size
            name = ""
            for _ in range(15):
                x = torch.tensor([context])
                logits = model(x)
                if temperature != 1.0:
                    logits = logits / temperature
                ix = torch.distributions.Categorical(logits=logits).sample().item()
                if ix == 0:
                    break
                name += itos[ix]
                context = context[1:] + [ix]
            names.append(name)
    return names


# --- App ---

st.set_page_config(page_title="Auto-Name: Next-Char Generation", layout="wide")
st.title("Auto-Name: Next-Character Name Generation")
st.markdown("Autoresearch experiment results for character-level name generation on Indian names.")

# Show experiment results
if os.path.exists(RESULTS_TSV):
    st.header("Experiment Results")
    df = pd.read_csv(RESULTS_TSV, sep="\t")

    # Highlight best
    best_idx = df["val_loss"].idxmin()
    st.dataframe(
        df.style.highlight_min(subset=["val_loss"], color="#3fb950"),
        use_container_width=True,
    )

    best = df.loc[best_idx]
    st.success(f"Best: Experiment #{int(best['run'])} — **{best['description']}** — val_loss = {best['val_loss']:.4f}")

# Show plot
if os.path.exists(PLOT_PATH):
    st.header("Experiment Plot")
    st.image(PLOT_PATH)

# Name generation section
st.header("Generate Names")
st.markdown("Enter model checkpoint details to generate names, or use the defaults from the best experiment.")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
    n_names = st.slider("Number of names", 1, 50, 20)

with col2:
    block_size = st.number_input("Block size (context)", 1, 8, 3)
    emb_dim = st.number_input("Embedding dim", 2, 64, 16)
    hidden_dim = st.number_input("Hidden dim (0=linear)", 0, 512, 128)
    n_hidden = st.number_input("Num hidden layers", 0, 3, 1)

if st.button("Generate Names (random model)"):
    meta = load_meta()
    model = CharModel(meta["vocab_size"], block_size, emb_dim, hidden_dim, n_hidden)
    names = generate_names(model, block_size, meta["itos"], n_names, temperature)
    st.write("**Generated names (untrained model — random):**")
    cols = st.columns(4)
    for i, name in enumerate(names):
        cols[i % 4].write(f"- {name}")

# Show dataset info
st.header("Dataset")
meta = load_meta()
st.write(f"- **Vocabulary size:** {meta['vocab_size']} characters")
st.write(f"- **Training names:** {meta['n_train']:,}")
st.write(f"- **Validation names:** {meta['n_val']:,}")

# Show sample names
train_path = os.path.join(DATA_DIR, "train.txt")
if os.path.exists(train_path):
    with open(train_path) as f:
        sample_names = [line.strip() for line in f][:20]
    st.write("**Sample training names:**")
    st.write(", ".join(sample_names))
