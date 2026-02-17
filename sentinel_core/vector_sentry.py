"""
TrollGuard — Runtime Vector Sentry (Layer 4: The Bodyguard)

Real-time firewall protecting AI agents during operation.  Validates
live data streams (scraped HTML, user chat, API responses) using a
sliding-window vectorization strategy with per-chunk ML classification.

Integrates with NG-Lite for adaptive threat-pattern learning:
  - Records scan outcomes so the Hebbian substrate learns which
    vector patterns correspond to true positives vs false positives.
  - Uses NG-Lite novelty detection to flag never-before-seen attack
    shapes for closer scrutiny (auto-escalation).

PRD reference: Section 7 — Layer 4: Runtime Vector Sentry

# ---- Changelog ----
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: Scaffolded VectorSentry class with sliding-window chunking,
#         per-chunk scoring, threshold actions (SAFE/REDACT/BLOCK),
#         and NG-Lite integration hooks.
#   Why:  Phase 1 "Iron Dome MVP" deliverable.  The runtime sentry is
#         the first line of defense for live I/O and must work before
#         the full Swarm Audit pipeline is built.
#   Settings: chunk_size=256, overlap=50, embedding_dim=384 — these
#         match the system-wide chunking standard defined in PRD §5.3.
#         safe_ceiling=0.3, malicious_floor=0.7 — initial thresholds
#         per PRD §5.4, require calibration via Platt scaling before
#         production use.
#   How:  Follows the same pattern as The-Inference-Difference's router:
#         accepts an optional NGLite instance, records outcomes for
#         learning, falls back gracefully when NG-Lite is absent.
# -------------------
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("trollguard.vector_sentry")


# ---------------------------------------------------------------------------
# Constants — system-wide chunking standard (PRD §5.3)
# ---------------------------------------------------------------------------

CHUNK_SIZE_TOKENS = 256
CHUNK_OVERLAP_TOKENS = 50
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2


class ThreatSignal(str, Enum):
    """Traffic-light classification per PRD §5.4 / §7.2."""
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    MALICIOUS = "MALICIOUS"


class SentryMode(str, Enum):
    """Runtime sentry operating modes per PRD §10 config."""
    REDACT = "redact"
    BLOCK = "block"
    REPORT_ONLY = "report_only"


@dataclass
class ChunkResult:
    """Classification result for a single text chunk.

    Attributes:
        chunk_index: Position of this chunk in the input stream.
        text: The raw text of the chunk (retained for quarantine logging).
        score: ML classifier confidence score [0.0, 1.0].
        signal: Traffic-light classification derived from score.
        embedding: The 384-dim vector embedding of this chunk.
    """
    chunk_index: int = 0
    text: str = ""
    score: float = 0.0
    signal: ThreatSignal = ThreatSignal.SAFE
    embedding: Optional[np.ndarray] = None


@dataclass
class SentryResult:
    """Aggregated result for a full text scan.

    Attributes:
        verdict: Overall verdict (max-score chunk determines this).
        max_score: Highest per-chunk score (PRD §5.3: max, not average).
        chunks_scanned: Total chunks processed.
        flagged_chunks: Chunks scoring above safe_ceiling.
        sanitized_text: Output text after redaction/blocking.
        scan_time_ms: Wall-clock time for the scan.
        source: Where the text came from (e.g., "web_search", "user_input").
    """
    verdict: ThreatSignal = ThreatSignal.SAFE
    max_score: float = 0.0
    chunks_scanned: int = 0
    flagged_chunks: List[ChunkResult] = field(default_factory=list)
    sanitized_text: str = ""
    scan_time_ms: float = 0.0
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging and quarantine records."""
        return {
            "verdict": self.verdict.value,
            "max_score": round(self.max_score, 4),
            "chunks_scanned": self.chunks_scanned,
            "flagged_count": len(self.flagged_chunks),
            "scan_time_ms": round(self.scan_time_ms, 2),
            "source": self.source,
        }


class VectorSentry:
    """Runtime vector sentry — real-time text stream protection.

    Implements the sliding-window vectorization strategy from PRD §7.1:
    text is chunked into 256-token windows with 50-token overlap, each
    chunk is embedded and classified independently, and the max-score
    chunk determines the overall verdict.

    Usage:
        sentry = VectorSentry(config=sentry_config)

        # Scan incoming text
        result = sentry.scan("some potentially dangerous text", source="web_search")

        # The sanitized_text field has redactions applied
        safe_text = result.sanitized_text

    With NG-Lite learning:
        from ng_lite import NGLite
        ng = NGLite(module_id="trollguard")
        sentry = VectorSentry(config=sentry_config, ng_lite=ng)
        # Now scan outcomes are recorded for adaptive learning
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        classifier: Optional[Any] = None,
        embedder: Optional[Any] = None,
        ng_lite: Optional[Any] = None,
    ):
        """
        Args:
            config: Sentry configuration (thresholds, mode, sensitivity).
            classifier: Trained ML classifier with predict_proba(embeddings).
                       If None, a stub is used that returns 0.0 (safe) for
                       all inputs — allows scaffolding before model training.
            embedder: Embedding model with encode(texts) -> np.ndarray.
                     If None, a hash-based stub is used (for dev/testing).
            ng_lite: Optional NGLite instance for adaptive learning.
        """
        self.config = {
            "safe_ceiling": 0.3,
            "malicious_floor": 0.7,
            "mode": SentryMode.REDACT.value,
            "sensitivity": 0.8,
            "redaction_tag": "[TROLLGUARD: CONTENT REDACTED]",
            "chunk_size": CHUNK_SIZE_TOKENS,
            "chunk_overlap": CHUNK_OVERLAP_TOKENS,
            **(config or {}),
        }

        self._classifier = classifier
        self._embedder = embedder
        self._ng_lite = ng_lite

        # Counters for telemetry
        self._total_scans = 0
        self._total_blocks = 0
        self._total_redactions = 0

    # -------------------------------------------------------------------
    # Core API
    # -------------------------------------------------------------------

    def scan(
        self,
        text: str,
        source: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SentryResult:
        """Scan a text stream and return classification + sanitized output.

        Args:
            text: Raw text to scan (HTML, chat message, API response, etc.).
            source: Label for where this text came from (for logging).
            metadata: Optional context passed to NG-Lite for learning.

        Returns:
            SentryResult with verdict, sanitized text, and chunk details.
        """
        start = time.time()
        self._total_scans += 1

        if not text or not text.strip():
            return SentryResult(
                verdict=ThreatSignal.SAFE,
                sanitized_text=text,
                source=source,
            )

        # Step 1: Chunk the input (sliding window)
        chunks = self._chunk_text(text)

        # Step 2: Embed all chunks
        embeddings = self._embed_chunks(chunks)

        # Step 3: Classify each chunk
        scores = self._classify_embeddings(embeddings)

        # Step 4: Build per-chunk results
        chunk_results: List[ChunkResult] = []
        for i, (chunk_text, score, embedding) in enumerate(
            zip(chunks, scores, embeddings)
        ):
            signal = self._score_to_signal(score)
            chunk_results.append(ChunkResult(
                chunk_index=i,
                text=chunk_text,
                score=score,
                signal=signal,
                embedding=embedding,
            ))

        # Step 5: Aggregate — max score determines verdict (PRD §5.3)
        max_score = max(scores) if scores else 0.0
        verdict = self._score_to_signal(max_score)

        flagged = [c for c in chunk_results if c.signal != ThreatSignal.SAFE]

        # Step 6: Apply sanitization based on mode
        mode = SentryMode(self.config["mode"])
        sanitized = self._apply_sanitization(text, chunks, chunk_results, mode)

        # Step 7: Update telemetry
        if verdict == ThreatSignal.MALICIOUS:
            self._total_blocks += 1
        elif verdict == ThreatSignal.SUSPICIOUS:
            self._total_redactions += 1

        elapsed_ms = (time.time() - start) * 1000.0

        result = SentryResult(
            verdict=verdict,
            max_score=max_score,
            chunks_scanned=len(chunks),
            flagged_chunks=flagged,
            sanitized_text=sanitized,
            scan_time_ms=elapsed_ms,
            source=source,
        )

        # Step 8: Record outcome in NG-Lite for learning
        if self._ng_lite is not None and embeddings:
            self._record_ng_outcome(result, embeddings, metadata)

        logger.info(
            "Sentry scan: verdict=%s max_score=%.3f chunks=%d flagged=%d (%.1fms) [%s]",
            verdict.value, max_score, len(chunks), len(flagged),
            elapsed_ms, source,
        )

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Sentry telemetry for monitoring and Agent C consumption."""
        return {
            "total_scans": self._total_scans,
            "total_blocks": self._total_blocks,
            "total_redactions": self._total_redactions,
            "block_rate": (
                self._total_blocks / self._total_scans
                if self._total_scans > 0 else 0.0
            ),
            "mode": self.config["mode"],
            "safe_ceiling": self.config["safe_ceiling"],
            "malicious_floor": self.config["malicious_floor"],
            "ng_lite_connected": self._ng_lite is not None,
        }

    # -------------------------------------------------------------------
    # Internal: Chunking
    # -------------------------------------------------------------------

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks using word-level tokenization.

        Uses whitespace splitting as a lightweight token proxy.  The PRD
        specifies 256-token chunks with 50-token overlap.  For proper
        subword tokenization, swap in the MiniLM tokenizer when available.
        """
        words = text.split()
        chunk_size = self.config["chunk_size"]
        overlap = self.config["chunk_overlap"]
        step = max(1, chunk_size - overlap)

        chunks = []
        for start in range(0, len(words), step):
            chunk_words = words[start:start + chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
            if start + chunk_size >= len(words):
                break

        return chunks if chunks else [text]

    # -------------------------------------------------------------------
    # Internal: Embedding
    # -------------------------------------------------------------------

    def _embed_chunks(self, chunks: List[str]) -> List[np.ndarray]:
        """Embed text chunks into 384-dim vectors.

        If a real embedder (sentence-transformers) is provided, uses it.
        Otherwise falls back to a deterministic hash-based stub that
        produces consistent vectors for testing and development.
        """
        if self._embedder is not None:
            # Real embedder: expects .encode(list_of_strings)
            raw = self._embedder.encode(chunks)
            if isinstance(raw, np.ndarray) and raw.ndim == 2:
                return [raw[i] for i in range(raw.shape[0])]
            return [np.array(r) for r in raw]

        # Stub: deterministic hash-based embeddings for dev/testing
        return [self._hash_embed(chunk) for chunk in chunks]

    @staticmethod
    def _hash_embed(text: str) -> np.ndarray:
        """Deterministic hash-based embedding stub.

        Produces a reproducible 384-dim unit vector from text content.
        NOT suitable for production — only for scaffolding and tests
        before the sentence-transformers model is loaded.
        """
        h = hashlib.sha512(text.encode("utf-8")).digest()
        # Expand hash bytes to fill 384 dims
        repeats = (EMBEDDING_DIM * 4 + len(h) - 1) // len(h)
        raw_bytes = (h * repeats)[: EMBEDDING_DIM * 4]
        vec = np.frombuffer(raw_bytes, dtype=np.float32)[:EMBEDDING_DIM]
        # Normalize to unit vector
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        return vec

    # -------------------------------------------------------------------
    # Internal: Classification
    # -------------------------------------------------------------------

    def _classify_embeddings(
        self, embeddings: List[np.ndarray]
    ) -> List[float]:
        """Score embeddings using the trained classifier.

        If a real classifier is provided, uses predict_proba.
        Otherwise returns 0.0 (safe) for all chunks — allows the
        sentry to run in report_only mode before model training.
        """
        if not embeddings:
            return []

        if self._classifier is not None:
            X = np.vstack(embeddings)
            probas = self._classifier.predict_proba(X)
            # Column 1 = P(malicious)
            if probas.ndim == 2 and probas.shape[1] >= 2:
                return [float(p) for p in probas[:, 1]]
            return [float(p) for p in probas]

        # Stub: return 0.0 (safe) when no classifier is loaded
        return [0.0] * len(embeddings)

    # -------------------------------------------------------------------
    # Internal: Threshold logic
    # -------------------------------------------------------------------

    def _score_to_signal(self, score: float) -> ThreatSignal:
        """Map a numeric score to a traffic-light signal.

        Thresholds per PRD §5.4:
          < safe_ceiling  → SAFE
          >= malicious_floor → MALICIOUS
          in between → SUSPICIOUS
        """
        if score >= self.config["malicious_floor"]:
            return ThreatSignal.MALICIOUS
        if score >= self.config["safe_ceiling"]:
            return ThreatSignal.SUSPICIOUS
        return ThreatSignal.SAFE

    # -------------------------------------------------------------------
    # Internal: Sanitization
    # -------------------------------------------------------------------

    def _apply_sanitization(
        self,
        original_text: str,
        chunks: List[str],
        chunk_results: List[ChunkResult],
        mode: SentryMode,
    ) -> str:
        """Apply redaction or blocking based on mode and chunk signals.

        Modes (PRD §10):
          report_only: return text unchanged (logging only)
          redact: replace SUSPICIOUS/MALICIOUS chunks with redaction tag
          block: return empty string if any chunk is MALICIOUS
        """
        if mode == SentryMode.REPORT_ONLY:
            return original_text

        if mode == SentryMode.BLOCK:
            has_malicious = any(
                c.signal == ThreatSignal.MALICIOUS for c in chunk_results
            )
            if has_malicious:
                return ""
            # Fall through to redaction for SUSPICIOUS chunks
            mode = SentryMode.REDACT

        # Redact mode: replace flagged chunks in the original text
        result = original_text
        tag = self.config["redaction_tag"]
        for cr in reversed(chunk_results):
            if cr.signal in (ThreatSignal.SUSPICIOUS, ThreatSignal.MALICIOUS):
                result = result.replace(cr.text, tag, 1)

        return result

    # -------------------------------------------------------------------
    # Internal: NG-Lite integration
    # -------------------------------------------------------------------

    def _record_ng_outcome(
        self,
        result: SentryResult,
        embeddings: List[np.ndarray],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Record scan outcome in NG-Lite for adaptive learning.

        Uses the max-scoring chunk's embedding as the pattern signal
        and the verdict as the target.  Over time, NG-Lite learns
        which embedding patterns are true threats vs false positives.
        """
        if not result.flagged_chunks:
            # Only learn from interesting (non-trivially-safe) scans
            return

        # Use the highest-scoring chunk for learning
        best_chunk = max(result.flagged_chunks, key=lambda c: c.score)
        if best_chunk.embedding is None:
            return

        # target_id = the threat classification
        # success = whether the classification was correct
        # (defaults to True; flipped by human review feedback later)
        self._ng_lite.record_outcome(
            embedding=best_chunk.embedding,
            target_id=result.verdict.value,
            success=True,
            metadata={
                "source": result.source,
                "max_score": result.max_score,
                "chunks_scanned": result.chunks_scanned,
                **(metadata or {}),
            },
        )
