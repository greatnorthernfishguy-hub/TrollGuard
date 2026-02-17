"""
TrollGuard — ML Classifier (Layer 2: The Mind Reader)

Random Forest classifier for semantic threat detection.  Trained on
prompt-injection datasets, calibrated via Platt scaling so output
scores map to meaningful probabilities.

Integrates with NG-Lite: classification outcomes (especially near-
threshold cases) are recorded so the Hebbian substrate learns which
score ranges correspond to true vs false positives for different
content types.

PRD reference: Section 5 — Layer 2: Sentinel ML Pipeline

# ---- Changelog ----
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: SentinelClassifier class wrapping scikit-learn Random Forest
#         with model loading, prediction, and score calibration.
#   Why:  Core component of the 4-layer pipeline.  Layer 2 catches
#         threats that bypass static signatures — novel prompt injections,
#         obfuscated logic, imperative commands hidden in docstrings.
#   Settings: n_estimators=200, class_weight="balanced" — chosen per
#         PRD §8.1.3 to handle skewed real-world data without hard
#         undersampling.  Model path defaults to "models/sentinel_model.pkl".
#   How:  Follows scikit-learn's predict_proba interface so VectorSentry
#         can use it directly.  Platt scaling (CalibratedClassifierCV)
#         applied during training, not at inference time.
# -------------------
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("trollguard.ml_classifier")


class SentinelClassifier:
    """Random Forest classifier for prompt injection detection.

    Wraps a scikit-learn model with load/save, predict, and stats.
    The model is trained separately by train_model.py and loaded at
    runtime.

    Usage:
        clf = SentinelClassifier()
        clf.load("models/sentinel_model.pkl")

        # Classify embeddings
        scores = clf.predict_proba(embeddings)  # shape (N, 2)
        malicious_probs = scores[:, 1]

        # Classify single text (requires embedder)
        from sentinel_core.vector_sentry import VectorSentry
        sentry = VectorSentry(classifier=clf)
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model = None
        self._model_path = model_path or "models/sentinel_model.pkl"
        self._prediction_count = 0

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def load(self, path: str) -> bool:
        """Load a trained model from disk.

        Args:
            path: Path to the pickled model file.

        Returns:
            True if loaded successfully.
        """
        try:
            import joblib
            self._model = joblib.load(path)
            self._model_path = path
            logger.info("Loaded sentinel model from %s", path)
            return True
        except Exception as e:
            logger.warning("Failed to load model from %s: %s", path, e)
            return False

    def save(self, path: Optional[str] = None) -> bool:
        """Save the current model to disk.

        Args:
            path: Where to save. Defaults to self._model_path.

        Returns:
            True if saved successfully.
        """
        if self._model is None:
            logger.warning("No model to save")
            return False

        save_path = path or self._model_path
        try:
            import joblib
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._model, save_path)
            logger.info("Saved sentinel model to %s", save_path)
            return True
        except Exception as e:
            logger.error("Failed to save model: %s", e)
            return False

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict class probabilities for embedding vectors.

        Args:
            embeddings: Array of shape (N, embedding_dim).

        Returns:
            Array of shape (N, 2) with columns [P(safe), P(malicious)].
            If no model is loaded, returns 0.5/0.5 for all inputs.
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        self._prediction_count += embeddings.shape[0]

        if self._model is None:
            # No model loaded — return neutral probabilities
            return np.full((embeddings.shape[0], 2), 0.5)

        return self._model.predict_proba(embeddings)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict class labels (0=safe, 1=malicious).

        Args:
            embeddings: Array of shape (N, embedding_dim).

        Returns:
            Array of shape (N,) with class labels.
        """
        probas = self.predict_proba(embeddings)
        return np.argmax(probas, axis=1)

    @property
    def is_loaded(self) -> bool:
        """Whether a trained model is available."""
        return self._model is not None

    def get_stats(self) -> Dict[str, Any]:
        """Classifier telemetry."""
        return {
            "model_loaded": self.is_loaded,
            "model_path": self._model_path,
            "total_predictions": self._prediction_count,
        }
