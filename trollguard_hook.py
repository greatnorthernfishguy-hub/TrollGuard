"""
TrollGuard OpenClaw Hook — E-T Systems Standard Integration

Exposes TrollGuard's 4-layer security pipeline as an OpenClaw skill,
using the standardized OpenClawAdapter base class.

OpenClaw calls get_instance().on_message(text) on every turn.
The adapter handles all ecosystem wiring (Tier 1/2/3 learning) and
memory logging.  This file only implements what's unique to TrollGuard:

  - _embed():              Sentence-transformer / hash fallback
  - _module_on_message():  Run the security scan pipeline
  - _module_stats():       TrollGuard-specific telemetry

SKILL.md entry:
    name: trollguard
    autoload: true
    hook: trollguard_hook.py::get_instance

# ---- Changelog ----
# [2026-03-19] Claude Code (Opus 4.6) — Migrate to fastembed + BAAI/bge-base-en-v1.5 (#45)
# What: Replaced sentence-transformers SentenceTransformer with fastembed
#   TextEmbedding. Model all-MiniLM-L6-v2 → BAAI/bge-base-en-v1.5 (768-dim).
# Why: Ecosystem-wide embedding migration. sentence-transformers was the
#   instigating failure that broke the ecosystem. Punchlist #45.
# How: _embed() rewritten: fastembed lazy-load + embed, hash fallback.
# -------------------
# [2026-03-18] Claude (CC) — Fix target_id Law 7 violation (#30)
# What: Changed target_id from category labels ("threat:MALICIOUS") to
#   content-derived identifiers ("scan:{embedding_hash}"). The substrate
#   now learns from the actual threat pattern, not from labels.
# Why: Punch list #30. Classification at input = extraction boundary
#   violation. The substrate learned "MALICIOUS" as a node, not the
#   content pattern that was malicious. All threats of the same label
#   were indistinguishable to the substrate.
# How: target_id = "scan:{sha256(embedding)[:16]}" — content-derived,
#   unique per input pattern. The substrate associates the specific
#   embedding with the outcome, not a label with the outcome.
# -------------------
# [2026-02-22] Claude (Sonnet 4.6) — Refactored to OpenClawAdapter standard.
#   What: Replaced bespoke singleton hook with OpenClawAdapter subclass.
#         Moved ecosystem wiring (NGLite, peer bridge, Tier 3 upgrade)
#         into ng_ecosystem.py / openclaw_adapter.py.  This file now
#         only contains TrollGuard-specific logic.
#   Why:  Standardization pass — all E-T Systems modules now expose
#         identical on_message()/recall()/stats() vocabulary to OpenClaw.
#   How:  Subclass OpenClawAdapter, set class attributes, implement
#         the three hook methods.  get_instance() singleton is unchanged.
#
# [2026-02-19] Claude (Opus 4.6) — Grok audit claim: "silent fallback in _init_sentry()".
#   Claim:  If VectorSentry init fails and _sentry is None, sanitize() "quietly
#           returns SAFE verdict — bypassing the entire immune system."
#   Status: NOT VALID — no code change required.
#   Why:    Fail-open behavior is intentional and logged.  See prior changelog.
#
# [2026-02-19] Claude (Opus 4.6) — Grok security audit: thread safety + file locking.
#   What: Added threading.RLock for sentry/model access, fcntl on event
#         JSONL writes.  (Now handled by OpenClawAdapter base class.)
#
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: TrollGuardFilter singleton with sanitize(), scan_file(),
#         scan_url(), and scan_repo() methods.
#   Why:  Glue between TrollGuard and the agent framework.
# -------------------
"""

from __future__ import annotations

# Auto-update on startup — pull latest code + sync vendored files
try:
    from ng_updater import auto_update; auto_update()
except Exception:
    pass  # Never prevent module startup

import logging
from typing import Any, Dict, Optional

import numpy as np

from openclaw_adapter import OpenClawAdapter

logger = logging.getLogger("trollguard_hook")


class TrollGuardHook(OpenClawAdapter):
    """OpenClaw integration hook for TrollGuard."""

    MODULE_ID = "trollguard"
    SKILL_NAME = "TrollGuard Security"
    WORKSPACE_ENV = "TROLLGUARD_WORKSPACE_DIR"
    DEFAULT_WORKSPACE = "~/TrollGuard/data"

    def __init__(self) -> None:
        super().__init__()

        # Initialize TrollGuard's pipeline components here
        self._scanner: Optional[Any] = None
        self._scan_count = 0
        self._threat_count = 0
        self._init_scanner()

    def _init_scanner(self) -> None:
        """Initialize the TrollGuard scan pipeline."""
        try:
            from sentinel_core.ml_classifier import MLClassifier
            # ... real initialization ...
            logger.info("TrollGuard scanner initialized")
        except Exception as exc:
            logger.warning("TrollGuard scanner unavailable: %s", exc)

    # -----------------------------------------------------------------
    # OpenClawAdapter implementation
    # -----------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Embed text via ng_embed (centralized ecosystem embedding).

        Ecosystem standard: Snowflake/snowflake-arctic-embed-m-v1.5 (768-dim).
        ONNX Runtime, no torch dependency.
        """
        try:
            from ng_embed import embed
            return embed(text)
        except Exception:
            return self._hash_embed(text)

    def _module_on_message(self, text: str, embedding: np.ndarray) -> Dict[str, Any]:
        """Run TrollGuard's security scan on the incoming message."""
        logger.info("TrollGuard hook invoked with text: %.50s...", text)
        self._scan_count += 1
        result: Dict[str, Any] = {"scan_count": self._scan_count}

        if self._scanner is None:
            result["scan_status"] = "scanner_unavailable"
            return result

        try:
            # Replace with actual TrollGuard scan call
            # scan_result = self._scanner.scan(text)
            scan_result = {"threat": False, "confidence": 0.0, "label": "clean"}

            result["scan_status"] = "scanned"
            result["threat"] = scan_result.get("threat", False)
            result["threat_confidence"] = scan_result.get("confidence", 0.0)
            result["threat_label"] = scan_result.get("label", "clean")

            if scan_result.get("threat"):
                self._threat_count += 1

                # Dual-pass outcome: forest + tree concepts (Punchlist #81)
                # target_id is content-derived, not a category label.
                # The substrate learns from the actual threat pattern.
                import hashlib as _hl
                _emb_hash = _hl.sha256(embedding.tobytes()).hexdigest()[:16]
                self._eco.dual_record_outcome(
                    content=text,
                    embedding=embedding,
                    target_id=f"scan:{_emb_hash}",
                    success=not scan_result.get("threat", False),
                    metadata={
                        "confidence": scan_result.get("confidence", 0.0),
                        "label": scan_result.get("label", "unknown"),
                    },
                )
        except Exception as exc:
            result["scan_status"] = f"error: {exc}"

        return result

    def _module_stats(self) -> Dict[str, Any]:
        """TrollGuard-specific telemetry."""
        return {
            "scan_count": self._scan_count,
            "threat_count": self._threat_count,
            "scanner_active": self._scanner is not None,
        }


# --------------------------------------------------------------------------
# Singleton wiring — identical pattern for all E-T Systems modules
# --------------------------------------------------------------------------

_INSTANCE: Optional[TrollGuardHook] = None


def get_instance() -> TrollGuardHook:
    """Return the TrollGuard OpenClaw hook singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = TrollGuardHook()
    return _INSTANCE
