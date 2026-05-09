#!/usr/bin/env python3
"""
UMAP visualization of StyleDistance embeddings from the RAID dataset.

Three classes:
  - human          → gray
  - known LLM      → blue   (chatgpt, gpt3, gpt4, cohere, llama-chat, mpt, mpt-chat)
  - unknown LLM    → red    (everything else that is not human)

Usage:
    python analysis/plot_umap.py [--cache_dir analysis/cache] [--output analysis/figures/umap_embeddings.pdf]
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants matching the training configuration (final_model/metadata.json)
# ---------------------------------------------------------------------------

MODEL_ID = "StyleDistance/styledistance"
DATASET_ID = "liamdugan/raid"
DATASET_SPLITS = ["train", "extra"]
MAX_LENGTH = 512
EMBEDDING_LAYER = -2  # second-to-last hidden layer
BATCH_SIZE = 32

KNOWN_MODELS = frozenset(
    ["chatgpt", "gpt3", "gpt4", "cohere", "llama-chat", "mpt", "mpt-chat"]
)
HUMAN_LABELS = frozenset(["human", "real", "original", "gold", "none_human"])

SAMPLES_PER_CLASS = 800
SEED = 42

# UMAP hyperparameters
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"

# Plot colours
COLOR_HUMAN = "#888888"  # gray
COLOR_KNOWN = "#2166ac"  # blue
COLOR_UNKNOWN = "#d6604d"  # red


# ---------------------------------------------------------------------------
# Helpers that mirror the training code exactly
# ---------------------------------------------------------------------------


def norm(x: str) -> str:
    return str(x).strip().lower()


def is_human(label: str) -> bool:
    return norm(label) in HUMAN_LABELS


def raid_row_to_sample(row: dict) -> dict:
    text = str(row.get("generation", row.get("text", row.get("content", ""))))
    model = norm(row.get("model", row.get("source", row.get("label", ""))))
    if is_human(model):
        model = "human"
    return {
        "text": text,
        "model": model,
        "attack": norm(row.get("attack", "unknown")),
    }


def classify_sample(model: str) -> str | None:
    """Return 'human', 'known', 'unknown', or None to skip."""
    if model == "human":
        return "human"
    if model in KNOWN_MODELS:
        return "known"
    # Any non-human model that is not in the known set is "unknown"
    return "unknown"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_texts(n_per_class: int, seed: int) -> dict[str, list[str]]:
    """Stream RAID and collect n_per_class text samples for each of the three classes."""
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[str]] = {"human": [], "known": [], "unknown": []}

    from itertools import chain

    streams = []
    for split in DATASET_SPLITS:
        print(f"  Opening {DATASET_ID} / {split} (streaming)…", flush=True)
        ds = load_dataset(DATASET_ID, split=split, streaming=True)
        ds = ds.shuffle(buffer_size=50_000, seed=int(rng.integers(1 << 31)))
        streams.append(ds)

    seen = 0
    for raw in tqdm(chain(*streams), desc="Collecting samples"):
        s = raid_row_to_sample(raw)
        if not s["text"].strip():
            continue
        # Skip adversarial attacks (matching training --include_adversarial=False default for
        # the cascade model; we just want clean text for the visualization)
        if s["attack"] not in {"none", "unknown", ""}:
            continue

        cls = classify_sample(s["model"])
        if cls is None:
            continue
        if len(buckets[cls]) < n_per_class:
            buckets[cls].append(s["text"])

        seen += 1
        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    for cls, texts in buckets.items():
        print(f"  Collected {len(texts)} {cls} samples (target {n_per_class})")
    return buckets


# ---------------------------------------------------------------------------
# Embedding extraction — identical to train_open_set_chtc.py::extract_embeddings
# ---------------------------------------------------------------------------


def extract_embeddings(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    if not texts:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)

    out = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embedding"):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[EMBEDDING_LAYER]
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        out.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(os.path.dirname(__file__), "cache"),
        help="Directory to cache computed embeddings.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(__file__), ".", "figures", "umap_embeddings.pdf"
        ),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--n_per_class",
        type=int,
        default=SAMPLES_PER_CLASS,
        help="Number of samples per class.",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Ignore cached embeddings and recompute from scratch.",
    )
    args = parser.parse_args()

    np.random.seed(SEED)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    cache_path = os.path.join(
        args.cache_dir, f"embeddings_n{args.n_per_class}_seed{SEED}.npz"
    )

    # ------------------------------------------------------------------
    # 1. Load or compute embeddings
    # ------------------------------------------------------------------
    if not args.force_recompute and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=False)
        X_human = data["X_human"]
        X_known = data["X_known"]
        X_unknown = data["X_unknown"]
    else:
        print("Collecting text samples from RAID…")
        buckets = collect_texts(args.n_per_class, SEED)

        print(f"Loading encoder: {MODEL_ID}")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        print(f"  device={device}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        encoder = AutoModel.from_pretrained(MODEL_ID).to(device)

        print("Extracting embeddings (human)…")
        X_human = extract_embeddings(buckets["human"], encoder, tokenizer, device)
        print("Extracting embeddings (known)…")
        X_known = extract_embeddings(buckets["known"], encoder, tokenizer, device)
        print("Extracting embeddings (unknown)…")
        X_unknown = extract_embeddings(buckets["unknown"], encoder, tokenizer, device)

        print(f"Saving embeddings to {cache_path}")
        np.savez_compressed(
            cache_path,
            X_human=X_human,
            X_known=X_known,
            X_unknown=X_unknown,
        )

    print(
        f"Embedding shapes — human: {X_human.shape}, known: {X_known.shape}, unknown: {X_unknown.shape}"
    )

    # ------------------------------------------------------------------
    # 2. UMAP projection
    # ------------------------------------------------------------------
    try:
        import umap
    except ImportError:
        sys.exit("umap-learn is not installed. Run:\n  pip install umap-learn\n")

    X_all = np.vstack([X_human, X_known, X_unknown])
    n_human = len(X_human)
    n_known = len(X_known)

    print(
        f"Running UMAP (n_neighbors={UMAP_N_NEIGHBORS}, min_dist={UMAP_MIN_DIST}, "
        f"metric={UMAP_METRIC}, random_state={SEED})…"
    )
    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=SEED,
        n_jobs=1,  # deterministic
    )
    Z = reducer.fit_transform(X_all)

    Z_human = Z[:n_human]
    Z_known = Z[n_human : n_human + n_known]
    Z_unknown = Z[n_human + n_known :]

    # ------------------------------------------------------------------
    # 3. Plot
    # ------------------------------------------------------------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is not installed. Run:\n  pip install matplotlib\n")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    scatter_kwargs = dict(s=12, alpha=0.55, linewidths=0, rasterized=True)

    ax.scatter(
        Z_human[:, 0],
        Z_human[:, 1],
        color=COLOR_HUMAN,
        label="Human",
        zorder=1,
        **scatter_kwargs,
    )
    ax.scatter(
        Z_known[:, 0],
        Z_known[:, 1],
        color=COLOR_KNOWN,
        label="Known LLM",
        zorder=2,
        **scatter_kwargs,
    )
    # Unknowns drawn last and slightly larger so they stay visible on top
    ax.scatter(
        Z_unknown[:, 0],
        Z_unknown[:, 1],
        color=COLOR_UNKNOWN,
        label="Unknown LLM",
        s=18,
        alpha=0.65,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    legend = ax.legend(
        frameon=True,
        framealpha=0.9,
        edgecolor="#cccccc",
        fontsize=9,
        markerscale=1.6,
        handletextpad=0.4,
        loc="best",
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(args.output, format="pdf", bbox_inches="tight", dpi=150)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
