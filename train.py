"""
Next-character name generation training script.
This is the file you modify during experiments.

Usage:
    python train.py
"""

import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import TIME_BUDGET, load_meta, load_words, build_dataset, evaluate

# ---------------------------------------------------------------------------
# Hyperparameters (edit these)
# ---------------------------------------------------------------------------

BLOCK_SIZE = 3
EMB_DIM = 16
HIDDEN_DIM = 128
N_HIDDEN = 1
BATCH_SIZE = 2048
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.1
DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CharModel(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim, hidden_dim, n_hidden, dropout):
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
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_d = hidden_dim
            layers.append(nn.Linear(hidden_dim, vocab_size))
            self.net = nn.Sequential(*layers)
        else:
            self.net = nn.Linear(input_dim, vocab_size)

    def forward(self, x):
        x = self.emb(x)                    # (B, block_size, emb_dim)
        x = x.view(x.shape[0], -1)         # (B, block_size * emb_dim)
        return self.net(x)                  # (B, vocab_size)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    print(f"Device: {device}")

    # Load data
    meta = load_meta()
    stoi = meta["stoi"]
    itos = meta["itos"]
    vocab_size = meta["vocab_size"]

    train_words = load_words("train")
    val_words = load_words("val")

    X_train, Y_train = build_dataset(train_words, BLOCK_SIZE, stoi, device)
    X_val, Y_val = build_dataset(val_words, BLOCK_SIZE, stoi, device)

    print(f"Train: {X_train.shape[0]:,} examples, Val: {X_val.shape[0]:,} examples")

    # Build model
    model = CharModel(vocab_size, BLOCK_SIZE, EMB_DIM, HIDDEN_DIM, N_HIDDEN, DROPOUT).to(device)
    n_params = count_params(model)
    print(f"Model: {n_params:,} parameters ({n_params/1e3:.1f}K)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Training loop
    model.train()
    step = 0
    epoch = 0
    smooth_loss = 0
    t_start = time.time()

    while True:
        epoch += 1
        # Shuffle training data each epoch
        perm = torch.randperm(X_train.shape[0], device=device)
        X_shuf = X_train[perm]
        Y_shuf = Y_train[perm]

        for i in range(0, len(X_shuf), BATCH_SIZE):
            t0 = time.time()

            logits = model(X_shuf[i:i + BATCH_SIZE])
            loss = F.cross_entropy(logits, Y_shuf[i:i + BATCH_SIZE])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            loss_val = loss.item()

            if math.isnan(loss_val) or loss_val > 50:
                print(f"\nLoss exploded ({loss_val:.2f}), aborting.")
                exit(1)

            beta = 0.95
            smooth_loss = beta * smooth_loss + (1 - beta) * loss_val if step > 1 else loss_val
            debiased = smooth_loss / (1 - beta ** step)

            elapsed = time.time() - t_start
            remaining = max(0, TIME_BUDGET - elapsed)

            if step % 20 == 0 or step == 1:
                print(f"\r  step {step:05d} | loss: {debiased:.4f} | epoch {epoch} | {remaining:.0f}s left   ", end="", flush=True)

            if elapsed >= TIME_BUDGET:
                break

        if time.time() - t_start >= TIME_BUDGET:
            break

    print()
    total_time = time.time() - t_start
    print(f"\nTraining: {step} steps, {epoch} epochs, {total_time:.1f}s")

    # Validation
    val_loss = evaluate(model, X_val, Y_val)
    print(f"Val loss: {val_loss:.4f}")

    # Generate sample names
    print("\nGenerated names:")
    model.eval()
    generated_names = []
    with torch.no_grad():
        for _ in range(10):
            context = [0] * BLOCK_SIZE
            name = ""
            for _ in range(15):
                x = torch.tensor([context], device=device)
                logits = model(x)
                ix = torch.distributions.Categorical(logits=logits).sample().item()
                if ix == 0:
                    break
                name += itos[ix]
                context = context[1:] + [ix]
            generated_names.append(name)
            print(f"  {name}")

    # Print 5 samples in machine-readable format for experiment harness
    print("\nSAMPLES:" + "|".join(generated_names[:5]))
    print(f"\nModel: {n_params/1e3:.1f}K params")


if __name__ == "__main__":
    train()
