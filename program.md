# auto-name

Autoresearch for next-character name generation. Based on the L07 notebook from principles-ai-teaching.

## Setup

1. Run `python prepare.py` to download the Indian names dataset, build char vocab, and create train/val split.
2. Run `python run_experiments.py` to kick off all experiments.
3. Once done, run `streamlit run app.py` for the interactive demo.

## Experiments

We test 5 model variants (plus a baseline), all trained for 2 minutes each:

| Variant | block_size | emb_dim | hidden | description |
|---------|-----------|---------|--------|-------------|
| Baseline | 3 | 16 | 128x1 | Default MLP |
| V1 | 1 | 2 | none | Bigram (embedding → linear) |
| V2 | 3 | 2 | none | Longer context, no hidden layer |
| V3 | 3 | 32 | 128x1 | Bigger embeddings |
| V4 | 5 | 32 | 256x2 | Deep + longer context + dropout |
| V5 | 8 | 16 | 512x1 | Wide + full context |

## What you CAN modify
- `train.py` — model architecture, hyperparameters, training loop

## What you CANNOT modify
- `prepare.py` — data loading, vocab, train/val split, evaluation function
