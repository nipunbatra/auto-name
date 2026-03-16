"""
Automated experiment harness for next-char name generation.
Runs a batch of experiments, records results, and generates a plot.

Usage:
    python run_experiments.py
"""

import os
import re
import subprocess
import time
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_TSV = os.path.join(BASE_DIR, "results.tsv")
PLOT_PATH = os.path.join(BASE_DIR, "experiments_plot.png")
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train.py")

BASELINE_LOSS = None  # set after first run

# --- Best config (updated as experiments improve) ---
BEST_CONFIG = {
    "BLOCK_SIZE": 3,
    "EMB_DIM": 16,
    "HIDDEN_DIM": 128,
    "N_HIDDEN": 1,
    "BATCH_SIZE": 2048,
    "LEARNING_RATE": 1e-2,
    "WEIGHT_DECAY": 0.1,
    "DROPOUT": 0.2,
}

# --- Experiments to run ---
# Each experiment: (description, {overrides})
# R4: 5min budget. Best so far: drop=0.2 wd=0.1 (val=2.0260 @2min)
# With 5min we get 2.5x more steps — should help a lot
EXPERIMENTS = [
    # Re-establish baseline at 5min
    ("R4 5min baseline drop=0.2 wd=0.1", {}),
    # LR: lower LR + more steps might converge better
    ("R4 lr=5e-3", {"LEARNING_RATE": 5e-3}),
    ("R4 lr=3e-3", {"LEARNING_RATE": 3e-3}),
    # Stronger regularization for longer training
    ("R4 drop=0.3", {"DROPOUT": 0.3}),
    ("R4 drop=0.25 wd=0.15", {"DROPOUT": 0.25, "WEIGHT_DECAY": 0.15}),
    # Block size — more context helps with more training
    ("R4 block=4", {"BLOCK_SIZE": 4}),
    ("R4 block=5 drop=0.25", {"BLOCK_SIZE": 5, "DROPOUT": 0.25}),
    # 2 hidden layers (now we have enough steps to train deeper)
    ("R4 2xhidden=128 drop=0.25", {"N_HIDDEN": 2, "DROPOUT": 0.25}),
    # Smaller batch = more updates in 5min
    ("R4 bs=512 lr=5e-3", {"BATCH_SIZE": 512, "LEARNING_RATE": 5e-3}),
    ("R4 bs=1024 lr=7e-3", {"BATCH_SIZE": 1024, "LEARNING_RATE": 7e-3}),
]


def get_next_run_number():
    """Read results.tsv and return the next experiment number."""
    if not os.path.exists(RESULTS_TSV):
        return 1
    with open(RESULTS_TSV) as f:
        lines = f.readlines()
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("run"):
            return int(line.split("\t")[0]) + 1
    return 1


def apply_config(overrides):
    """Patch train.py constants with the given config."""
    config = {**BEST_CONFIG, **overrides}
    with open(TRAIN_SCRIPT) as f:
        src = f.read()

    for key, val in config.items():
        if isinstance(val, bool):
            val_str = str(val)
        elif isinstance(val, float):
            if val < 0.01:
                val_str = re.sub(r'e([+-])0*(\d+)', r'e\1\2', f"{val:e}")
                mantissa, exp = val_str.split("e")
                mantissa = mantissa.rstrip("0").rstrip(".")
                val_str = f"{mantissa}e{exp}"
            else:
                val_str = str(val)
        else:
            val_str = str(val)

        pattern = rf'^({key}\s*=\s*).*$'
        replacement = rf'\g<1>{val_str}'
        src, count = re.subn(pattern, replacement, src, count=1, flags=re.MULTILINE)
        if count == 0:
            print(f"  WARNING: Could not find {key} in train.py")

    with open(TRAIN_SCRIPT, "w") as f:
        f.write(src)


def run_training(timeout=400):
    """Run train.py and return (val_loss, params_K, output)."""
    try:
        result = subprocess.run(
            ["python", TRAIN_SCRIPT],
            capture_output=True, text=True, timeout=timeout,
            cwd=BASE_DIR,
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT"

    val_match = re.search(r'Val loss:\s*([\d.]+)', output)
    val_loss = float(val_match.group(1)) if val_match else None

    params_match = re.search(r'Model:\s*([\d.]+)K', output)
    params_K = float(params_match.group(1)) if params_match else None

    # Parse samples
    samples_match = re.search(r'SAMPLES:(.*)', output)
    samples = samples_match.group(1).strip() if samples_match else ""

    return val_loss, params_K, samples, output


def append_result(run_num, val_loss, params_K, status, time_s, description, baseline_loss, samples=""):
    """Append a row to results.tsv."""
    improv = (baseline_loss - val_loss) / baseline_loss * 100 if baseline_loss else 0
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{run_num:03d}\t{val_loss:.4f}\t{params_K}\t{status}\t{time_s}\t{improv:.1f}\t{description}\t{samples}\n")


def generate_plot(baseline_loss):
    """Regenerate experiments_plot.png from results.tsv."""
    runs, losses, statuses, descs, params = [], [], [], [], []
    with open(RESULTS_TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            runs.append(int(row["run"]))
            losses.append(float(row["val_loss"]))
            statuses.append(row["status"])
            descs.append(row["description"])
            params.append(float(row["params_K"]))

    fig, (ax_main, ax_bar) = plt.subplots(2, 1, figsize=(12, 8),
                                           gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#0d1117")

    for ax in (ax_main, ax_bar):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#8b949e", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.12, color="#8b949e")

    # Main plot: val_loss per experiment
    keep_x = [r for r, s in zip(runs, statuses) if s == "keep"]
    keep_y = [l for l, s in zip(losses, statuses) if s == "keep"]
    disc_x = [r for r, s in zip(runs, statuses) if s == "discard"]
    disc_y = [l for l, s in zip(losses, statuses) if s == "discard"]

    ax_main.plot(runs, losses, c="#30363d", alpha=0.4, linewidth=1, zorder=1)
    ax_main.scatter(disc_x, disc_y, c="#f85149", s=50, zorder=3, alpha=0.5, marker="x", linewidths=1.5)
    ax_main.scatter(keep_x, keep_y, c="#3fb950", s=80, zorder=4, edgecolors="white", linewidth=0.8)

    # Best envelope
    best_so_far = []
    current_best = float("inf")
    for l in losses:
        current_best = min(current_best, l)
        best_so_far.append(current_best)
    ax_main.plot(runs, best_so_far, c="#3fb950", linewidth=2.5, alpha=0.8,
                 linestyle="--", label="best so far", zorder=2)

    # Annotate milestones
    prev_best = float("inf")
    for r, l, s, d in zip(runs, losses, statuses, descs):
        if s == "keep" and l < prev_best:
            improv_pct = (baseline_loss - l) / baseline_loss * 100
            ax_main.annotate(
                f"#{r} ({improv_pct:.0f}%)", (r, l), fontsize=8, fontweight="bold",
                textcoords="offset points", xytext=(6, 10), color="#3fb950",
                arrowprops=dict(arrowstyle="-", color="#3fb950", alpha=0.3, lw=0.6),
            )
            prev_best = l

    ax_main.set_xlabel("Experiment #", fontsize=11, color="#c9d1d9")
    ax_main.set_ylabel("Validation Loss", fontsize=11, color="#c9d1d9")
    ax_main.set_title("Next-Char Name Generation Experiments", fontsize=14, fontweight="bold", color="#c9d1d9")
    ax_main.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d",
                   labelcolor="#c9d1d9", fontsize=9)

    # Bar chart: improvement %
    if baseline_loss:
        improvs = [(baseline_loss - l) / baseline_loss * 100 for l in losses]
        colors = ["#3fb950" if s == "keep" else "#f8514944" for s in statuses]
        ax_bar.bar(runs, improvs, color=colors, width=0.8, zorder=3)
        ax_bar.axhline(y=0, color="#8b949e", linewidth=0.5)
        ax_bar.set_xlabel("Experiment #", fontsize=11, color="#c9d1d9")
        ax_bar.set_ylabel("Improvement %", fontsize=11, color="#c9d1d9")
        ax_bar.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))

    # Summary
    best_loss = min(losses)
    best_idx = losses.index(best_loss)
    best_improv = (baseline_loss - best_loss) / baseline_loss * 100 if baseline_loss else 0
    summary = f"Experiments: {len(runs)}  |  Best: {best_loss:.4f} (#{runs[best_idx]}, {best_improv:.1f}%)"
    fig.text(0.5, 0.005, summary, ha="center", fontsize=11, color="#58a6ff",
             family="monospace", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", fc="#161b22", ec="#58a6ff", alpha=0.9, lw=1))

    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved to {PLOT_PATH}")


def main():
    global BASELINE_LOSS

    print("=" * 60)
    print("AUTORESEARCH: Next-Char Name Generation")
    print("=" * 60)

    # Initialize results.tsv if needed
    if not os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV, "w") as f:
            f.write("run\tval_loss\tparams_K\tstatus\ttime_s\timprov_%\tdescription\tsamples\n")

    # Get current best from existing results
    best_val_loss = float("inf")
    if os.path.exists(RESULTS_TSV):
        with open(RESULTS_TSV) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row["status"] == "keep":
                    best_val_loss = min(best_val_loss, float(row["val_loss"]))
                    if BASELINE_LOSS is None:
                        BASELINE_LOSS = float(row["val_loss"])

    print(f"Experiments to run: {len(EXPERIMENTS)}")
    print()

    for desc, overrides in EXPERIMENTS:
        run_num = get_next_run_number()
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {run_num:03d}: {desc}")
        print(f"  Overrides: {overrides}")
        print(f"{'='*60}")

        # Apply config
        apply_config(overrides)
        print(f"  Config applied to train.py")

        # Train
        t0 = time.time()
        val_loss, params_K, samples, output = run_training(timeout=400)
        elapsed = int(time.time() - t0)

        if val_loss is None:
            print(f"  FAILED. Output tail:")
            print(output[-500:] if output else "(no output)")
            apply_config({})
            continue

        # Set baseline from first run
        if BASELINE_LOSS is None:
            BASELINE_LOSS = val_loss

        # Determine status
        if val_loss < best_val_loss:
            status = "keep"
            best_val_loss = val_loss
            tag = "NEW BEST!"
        else:
            status = "discard"
            tag = ""
            apply_config({})  # revert to best config

        improv = (BASELINE_LOSS - val_loss) / BASELINE_LOSS * 100
        desc_full = f"{desc} {tag}".strip()
        print(f"  Val loss: {val_loss:.4f} | {improv:.1f}% | Params: {params_K}K | {status} | {elapsed}s")
        if samples:
            print(f"  Samples: {samples}")

        append_result(run_num, val_loss, params_K, status, elapsed, desc_full, BASELINE_LOSS, samples)
        generate_plot(BASELINE_LOSS)

    # Final summary
    best_improv = (BASELINE_LOSS - best_val_loss) / BASELINE_LOSS * 100 if BASELINE_LOSS else 0
    print(f"\n{'='*60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Best val_loss: {best_val_loss:.4f} ({best_improv:.1f}% improvement)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
