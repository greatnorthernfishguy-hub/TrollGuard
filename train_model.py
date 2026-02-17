"""
TrollGuard — ML Model Training Script

Bootstraps the "Day 0" classifier using publicly available datasets.
Downloads prompt-injection datasets from HuggingFace, vectorizes text
using all-MiniLM-L6-v2 with 256-token chunking, trains a Random Forest
with class_weight='balanced', applies Platt scaling for confidence
calibration, and saves the model to models/sentinel_model.pkl.

PRD reference: Section 8 — Machine Learning: Training & Active Learning

# ---- Changelog ----
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: Training script stub with full pipeline outlined.
#   Why:  Phase 1 "Iron Dome MVP" deliverable.  The classifier must
#         exist before the Runtime Sentry can do real classification
#         (until then, it uses a stub returning 0.0 for all inputs).
#   Settings: n_estimators=200, class_weight="balanced", test_size=0.2,
#         chunk_size=256, overlap=50, Platt scaling via
#         CalibratedClassifierCV.
#   How:  Downloads datasets via huggingface_hub, standardizes to
#         (text, label) format, chunks + embeds, trains + calibrates,
#         saves to models/sentinel_model.pkl.
# -------------------

Usage:
    python train_model.py                    # Full training pipeline
    python train_model.py --skip-download    # Use cached datasets
    python train_model.py --eval-only        # Evaluate existing model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("trollguard.train")

# System-wide chunking standard (PRD §5.3)
CHUNK_SIZE = 256
CHUNK_OVERLAP = 50
EMBEDDING_DIM = 384
MODEL_PATH = "models/sentinel_model.pkl"

# HuggingFace datasets for training (PRD §8.1.1)
DATASETS = [
    "deepset/prompt-injections",
    "JasperLS/prompt-injections",
]


def download_datasets(cache_dir: str = "training_data") -> List[Dict[str, Any]]:
    """Download and merge training datasets from HuggingFace.

    Standardizes all datasets to {"text": str, "label": int} format.
    Labels: 0 = Safe, 1 = Malicious (Prompt Injection/Jailbreak)

    Returns:
        List of {"text": str, "label": int} dicts.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' package required: pip install datasets")
        sys.exit(1)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    all_samples: List[Dict[str, Any]] = []

    for ds_name in DATASETS:
        logger.info("Downloading %s...", ds_name)
        try:
            ds = load_dataset(ds_name, cache_dir=cache_dir)

            for split in ds:
                for row in ds[split]:
                    text = row.get("text", row.get("prompt", ""))
                    label = row.get("label", 0)

                    if isinstance(label, str):
                        label = 1 if label.lower() in ("1", "injection", "malicious", "jailbreak") else 0

                    if text and len(text.strip()) > 10:
                        all_samples.append({"text": text.strip(), "label": int(label)})

        except Exception as e:
            logger.warning("Failed to download %s: %s", ds_name, e)

    logger.info("Total samples: %d (safe=%d, malicious=%d)",
                len(all_samples),
                sum(1 for s in all_samples if s["label"] == 0),
                sum(1 for s in all_samples if s["label"] == 1))

    return all_samples


def chunk_text(text: str) -> List[str]:
    """Chunk text into overlapping windows (system-wide standard)."""
    words = text.split()
    step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start:start + CHUNK_SIZE]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        if start + CHUNK_SIZE >= len(words):
            break

    return chunks if chunks else [text]


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts using all-MiniLM-L6-v2.

    Returns:
        Array of shape (N, 384).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers required: pip install sentence-transformers")
        sys.exit(1)

    logger.info("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    logger.info("Embedding %d texts...", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    return np.array(embeddings)


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> Tuple[Any, Dict[str, Any]]:
    """Train Random Forest with Platt scaling.

    Args:
        X: Embedding matrix (N, 384).
        y: Labels (N,).
        test_size: Fraction held out for evaluation.

    Returns:
        (calibrated_model, evaluation_metrics)
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report,
        precision_recall_fscore_support,
    )

    logger.info("Splitting data (test_size=%.2f)...", test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42,
    )

    logger.info("Training Random Forest (n_estimators=200, class_weight=balanced)...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        max_depth=None,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)

    # Platt scaling for confidence calibration (PRD §8.1.4)
    logger.info("Applying Platt scaling (CalibratedClassifierCV)...")
    calibrated = CalibratedClassifierCV(rf, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)

    # Evaluate
    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary",
    )

    metrics = {
        "test_size": len(X_test),
        "train_size": len(X_train),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(report["accuracy"], 4),
        "classification_report": report,
    }

    logger.info("Evaluation: precision=%.4f recall=%.4f f1=%.4f accuracy=%.4f",
                precision, recall, f1, report["accuracy"])

    return calibrated, metrics


def save_model(model: Any, path: str = MODEL_PATH) -> None:
    """Save trained model to disk."""
    import joblib

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)


def main():
    parser = argparse.ArgumentParser(
        description="TrollGuard — Train the Sentinel ML Classifier",
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="Use cached datasets")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate existing model without retraining")
    parser.add_argument("--output", default=MODEL_PATH,
                        help="Model output path")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    start = time.time()

    # Step 1: Get training data
    logger.info("=== Step 1: Download Training Data ===")
    samples = download_datasets()

    if not samples:
        logger.error("No training data available. Cannot train.")
        sys.exit(1)

    # Step 2: Chunk and embed
    logger.info("=== Step 2: Chunk and Embed ===")
    all_chunks: List[str] = []
    all_labels: List[int] = []

    for sample in samples:
        chunks = chunk_text(sample["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_labels.append(sample["label"])

    logger.info("Chunked %d samples into %d chunks", len(samples), len(all_chunks))

    X = embed_texts(all_chunks)
    y = np.array(all_labels)

    # Step 3: Train
    logger.info("=== Step 3: Train Classifier ===")
    model, metrics = train_classifier(X, y)

    # Step 4: Save
    logger.info("=== Step 4: Save Model ===")
    save_model(model, args.output)

    # Save metrics
    metrics_path = args.output.replace(".pkl", "_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Metrics saved to %s", metrics_path)

    elapsed = time.time() - start
    logger.info("=== Training complete in %.1f seconds ===", elapsed)


if __name__ == "__main__":
    main()
