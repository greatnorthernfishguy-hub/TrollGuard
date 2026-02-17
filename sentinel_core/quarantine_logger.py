"""
TrollGuard — Quarantine Logger

Incident logging and training data capture.  Every UNSAFE verdict or
BLOCKED runtime chunk is recorded with full context: raw text, vector
embeddings, pipeline telemetry, and human review status.

The quarantine log serves dual purposes:
  1. Operational: human reviewers can audit blocked content and mark
     false positives/negatives for classifier retraining.
  2. Training: vector_embedding fields are saved as lists so the
     active learning loop can retrain without re-vectorizing.

PRD reference: Section 9.2 — Quarantine Log

# ---- Changelog ----
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: QuarantineLogger class with log_incident(), get_pending(),
#         mark_reviewed(), and export_training_data() methods.
#   Why:  Required for the active learning loop (Phase 4 "The Gym")
#         but also needed from Phase 1 onward to capture operational
#         data for future training.  Every scan should log from day 1.
#   Settings: quarantine_path defaults to "quarantine.json" in the
#         project root, matching the PRD §3 project structure.
#   How:  Append-only JSON lines format for crash safety.  Each entry
#         is a self-contained record that can be parsed independently.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("trollguard.quarantine_logger")


class QuarantineLogger:
    """Append-only incident logger for quarantined content.

    Usage:
        ql = QuarantineLogger("quarantine.json")

        # Log an incident
        ql.log_incident(
            source="web_search",
            trigger_engine="vector_sentry",
            layer=4,
            vector_score=0.82,
            raw_text="suspicious text...",
            vector_embedding=embedding.tolist(),
            pipeline_telemetry={...},
        )

        # Review incidents
        pending = ql.get_pending()
        ql.mark_reviewed(incident_id, "false_positive")
    """

    def __init__(self, quarantine_path: str = "quarantine.json"):
        self._path = Path(quarantine_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._incident_count = 0

    def log_incident(
        self,
        source: str,
        trigger_engine: str,
        layer: int,
        vector_score: float,
        raw_text: str = "",
        vector_embedding: Optional[List[float]] = None,
        pipeline_telemetry: Optional[Dict[str, Any]] = None,
        canary_result: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log a quarantine incident.

        Args:
            source: Where the content came from (e.g., "web_search").
            trigger_engine: Which engine flagged it (e.g., "vector_sentry").
            layer: Which pipeline layer caught it (1-4).
            vector_score: The ML classifier score [0.0, 1.0].
            raw_text: The flagged content (for human review).
            vector_embedding: The embedding vector as a list of floats.
            pipeline_telemetry: Full pipeline state at time of flag.
            canary_result: Canary protocol results (if from Swarm Audit).
            metadata: Additional context.

        Returns:
            Unique incident ID for later reference.
        """
        self._incident_count += 1
        incident_id = f"tg_{int(time.time())}_{self._incident_count}"

        record = {
            "incident_id": incident_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "trigger_engine": trigger_engine,
            "layer": layer,
            "vector_score": round(vector_score, 6),
            "raw_text": raw_text[:5000],  # Cap for storage
            "vector_embedding": vector_embedding,
            "pipeline_telemetry": pipeline_telemetry or {},
            "canary_result": canary_result,
            "metadata": metadata or {},
            "human_review": None,
        }

        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
            logger.info("Quarantined: %s (%s, layer=%d, score=%.3f)",
                        incident_id, trigger_engine, layer, vector_score)
        except OSError as e:
            logger.error("Failed to write quarantine record: %s", e)

        return incident_id

    def get_pending(self) -> List[Dict[str, Any]]:
        """Get all incidents pending human review."""
        return [
            r for r in self._read_all()
            if r.get("human_review") is None
        ]

    def mark_reviewed(
        self,
        incident_id: str,
        review: str,
        reviewer: str = "human",
    ) -> bool:
        """Mark an incident as reviewed.

        Args:
            incident_id: The ID returned by log_incident().
            review: "false_positive", "true_positive", "false_negative",
                   or "confirmed_threat".
            reviewer: Who performed the review.

        Returns:
            True if the incident was found and updated.
        """
        records = self._read_all()
        found = False

        for r in records:
            if r.get("incident_id") == incident_id:
                r["human_review"] = {
                    "verdict": review,
                    "reviewer": reviewer,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                }
                found = True
                break

        if found:
            self._write_all(records)
            logger.info("Reviewed %s: %s by %s", incident_id, review, reviewer)

        return found

    def export_training_data(self) -> List[Dict[str, Any]]:
        """Export reviewed incidents as training data.

        Returns records that have been human-reviewed, with labels
        derived from the review verdict:
          false_positive → label=0 (safe)
          true_positive / confirmed_threat → label=1 (malicious)
          false_negative → label=1 (malicious, was missed)
        """
        label_map = {
            "false_positive": 0,
            "true_positive": 1,
            "confirmed_threat": 1,
            "false_negative": 1,
        }

        training = []
        for r in self._read_all():
            review = r.get("human_review")
            if review is None:
                continue

            verdict = review.get("verdict", "")
            label = label_map.get(verdict)
            if label is None:
                continue

            training.append({
                "text": r.get("raw_text", ""),
                "label": label,
                "vector_embedding": r.get("vector_embedding"),
                "source": r.get("source", ""),
                "original_score": r.get("vector_score", 0.0),
            })

        return training

    def get_stats(self) -> Dict[str, Any]:
        """Logger statistics."""
        records = self._read_all()
        reviewed = [r for r in records if r.get("human_review") is not None]
        return {
            "total_incidents": len(records),
            "pending_review": len(records) - len(reviewed),
            "reviewed": len(reviewed),
            "false_positives": sum(
                1 for r in reviewed
                if r["human_review"].get("verdict") == "false_positive"
            ),
            "path": str(self._path),
        }

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _read_all(self) -> List[Dict[str, Any]]:
        """Read all records from the quarantine file."""
        if not self._path.exists():
            return []

        records = []
        try:
            with open(self._path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to read quarantine file: %s", e)

        return records

    def _write_all(self, records: List[Dict[str, Any]]) -> None:
        """Rewrite all records (used after mark_reviewed)."""
        try:
            with open(self._path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, default=str) + "\n")
        except OSError as e:
            logger.error("Failed to write quarantine file: %s", e)
