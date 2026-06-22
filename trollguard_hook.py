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
# [2026-06-22] Claude Code (Opus 4.8) — #328 Step 3 (C): TrollGuard → depositor (not autonomic writer)
#   What: _update_autonomic_state() renamed → _deposit_perimeter_threat(): on a text-level threat it
#         DEPOSITS perimeter:threat:<hash> (severity in metadata) to the Commons instead of
#         ng_autonomic.write_state(). Removed the PARASYMPATHETIC write entirely. Caller passes the
#         embedding so the deposit carries the raw threat experience (LAW 7).
#   Why: #328 single-authority — Immunis is the SOLE arousal authority; it buckets TrollGuard's raw
#         perimeter-threat experience and decides SYMPATHETIC. TrollGuard writing the autonomic file
#         (incl. PARASYMPATHETIC) made it a writer/relaxer — the multi-writer clobber #328 removes.
#         (Josh-approved #328 conversion of autonomic write logic; design SYL-ACCEPTED.)
#   How: commons.deposit(embedding, "perimeter:threat:<hash>", metadata={threat_level, confidence,...}).
#         Single raw deposit (Immunis classifies at its bucket; NOT dual-pass). _autonomic_last_state
#         reused as a deposit-debounce. Fail-soft. Requires Immunis Step-3(A) listener (committed).
# [2026-05-25] Claude Code (Sonnet 4.6) — NEW-12+8: Autonomic write on threat + document River stub
#   What: (1) Added _autonomic_last_state tracking + _update_autonomic_state() — writes SYMPATHETIC
#         when threat detected, PARASYMPATHETIC when clear. Mirrors Immunis pattern (same struct).
#         (2) Documented that _on_river_events() is intentionally not overridden in CLAUDE.md §8.
#   Why:  TrollGuard is one of three authorized autonomic writers (§3 CLAUDE.md) but never called
#         ng_autonomic.write_state(). SYMPATHETIC was never actually being set by TrollGuard despite
#         being its core perimeter responsibility. Immunis already implements this pattern correctly.
#   How:  _update_autonomic_state(is_threat, confidence) called after is_threat assignment.
#         Confidence → threat_level: ≥0.90 critical, ≥0.70 high, else medium. Clear on clean
#         scan if we last wrote SYMPATHETIC (single clean message resets — matches text threat
#         domain where threats are per-message, not persistent like host-level Immunis threats).
# [2026-04-23] Claude Code (Sonnet 4.6) — Wire real TrollGuardPipeline, fix scan stub (#208)
#   What: _init_scanner() now instantiates TrollGuardPipeline (was only importing
#         SentinelClassifier without ever instantiating). Pre-imports
#         sentinel_core.vector_sentry so lazy import in scan_text() works after
#         _bootstrap_modules restores sys.path. _module_on_message() now calls
#         self._scanner.scan_text(text, source="openclaw") instead of a
#         hardcoded stub dict. scan_count and threat_count now increment.
#   Why:  _scan_count was permanently 0. Two failures: no scanner object and
#         stub that bypassed all real scanning logic.
#   How:  sys.path injection safe during __init__ (inside bootstrap try block).
#         scan_text() = Layer 4 VectorSentry, the layer designed for live I/O.
# [2026-04-19] CC (punchlist #5) -- Add pulse loop + River drain (first ever)
#   What: _pulse_loop() daemon thread drains River tracts between conversations.
#   Why:  TrollGuard never had a pulse loop despite fanout removal requiring one.
#   How:  Standard pattern (Immunis/Darwin). _pulse_cycle() calls _drain_river().
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
import threading
from typing import Any, Dict, Optional

import numpy as np

from openclaw_adapter import OpenClawAdapter

logger = logging.getLogger("trollguard_hook")


class TrollGuardHook(OpenClawAdapter):
    """OpenClaw integration hook for TrollGuard."""

    MODULE_ID = "trollguard"
    SKIP_ECOSYSTEM = True
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

        # Autonomic state tracking — write SYMPATHETIC on threat, clear on clean
        self._autonomic_last_state: str = "PARASYMPATHETIC"

        # Pulse loop — drains River tracts between conversations (#5)
        self._in_conversation = False
        self._pulse_thread = threading.Thread(
            target=self._pulse_loop, name="trollguard-pulse", daemon=True
        )
        self._pulse_thread.start()

    def _init_scanner(self) -> None:
        """Initialize the TrollGuard scan pipeline."""
        try:
            import sys as _sys, os as _os
            _tg_dir = _os.path.expanduser("~/TrollGuard")
            if _tg_dir not in _sys.path:
                _sys.path.insert(0, _tg_dir)
            # Pre-import sentinel_core so the lazy import inside scan_text()
            # survives _bootstrap_modules' sys.path cleanup (finally block).
            import sentinel_core.vector_sentry  # noqa: F401
            import main as _tg_main
            _config = _tg_main.load_config(_os.path.join(_tg_dir, "config.yaml"))
            self._scanner = _tg_main.TrollGuardPipeline(_config)
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

    def on_conversation_started(self) -> None:
        self._in_conversation = True

    def on_conversation_ended(self) -> None:
        self._in_conversation = False

    def _pulse_loop(self) -> None:
        import time as _t
        while True:
            try:
                self._pulse_cycle()
            except Exception as _exc:
                logger.debug("TrollGuard pulse error: %s", _exc)
            _t.sleep(10.0 if self._in_conversation else 60.0)

    def _pulse_cycle(self) -> None:
        """Drain River tracts and process substrate signals."""
        self._drain_river()

    def _module_on_message(self, text: str, embedding: np.ndarray) -> Dict[str, Any]:
        """Run TrollGuard's security scan on the incoming message."""
        logger.info("TrollGuard hook invoked with text: %.50s...", text)
        self._scan_count += 1
        result: Dict[str, Any] = {"scan_count": self._scan_count}

        if self._scanner is None:
            result["scan_status"] = "scanner_unavailable"
            return result

        try:
            raw = self._scanner.scan_text(text, source="openclaw")
            verdict = raw.get("verdict", "SAFE")
            scan_result = {
                "threat": verdict == "MALICIOUS",
                "confidence": raw.get("max_score", 0.0),
                "label": verdict.lower(),
            }

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
                if self._eco:
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

        # Domain-specific substrate outcome (#18)
        # Adapter reads these and records to ecosystem with domain context
        is_threat = result.get("threat", False)
        # #328 Step 3 (C): deposit perimeter-threat experience to the Commons (Immunis decides arousal).
        self._deposit_perimeter_threat(is_threat, result.get("threat_confidence", 0.0), embedding)
        import hashlib as _hl2
        _emb_hash2 = _hl2.sha256(embedding.tobytes()).hexdigest()[:16]
        result["_substrate_target_id"] = f"scan:{_emb_hash2}"
        result["_substrate_success"] = not is_threat  # clean = success, threat = failure

        return result

    def _deposit_perimeter_threat(self, is_threat: bool, confidence: float, embedding) -> None:
        """#328 Step 3 (C): deposit raw perimeter-threat experience to the Commons (NOT write arousal).

        TrollGuard is a DEPOSITOR, not an arousal writer — Immunis (the SOLE arousal authority) buckets
        this and decides SYMPATHETIC. On a text-level threat, deposit perimeter:threat:<hash> with the
        severity level in metadata; Immunis dedups by content-hash. NO PARASYMPATHETIC write — Immunis
        owns relaxation (single-authority; the old peer PARASYMPATHETIC write was the multi-writer
        clobber #328 removes). _autonomic_last_state is reused as a deposit-debounce so a standing
        threat isn't re-deposited every message. (Josh-approved #328 conversion of the autonomic write.)
        """
        try:
            if is_threat:
                if confidence >= 0.90:
                    level = "critical"
                elif confidence >= 0.70:
                    level = "high"
                else:
                    level = "medium"
                if self._autonomic_last_state != "SYMPATHETIC" and embedding is not None:
                    import hashlib as _hl, time as _t
                    from commons import get_commons
                    _c = get_commons()
                    if _c is not None:
                        h = _hl.sha256(embedding.tobytes()).hexdigest()[:16]
                        _c.deposit(
                            embedding, f"perimeter:threat:{h}",
                            metadata={"kind": "threat", "threat_level": level,
                                      "confidence": round(float(confidence), 3),
                                      "source": "trollguard", "ts": _t.time()},
                        )
                    self._autonomic_last_state = "SYMPATHETIC"
            else:
                # clean scan — reset the deposit-debounce; relaxation is Immunis's call (no write here)
                self._autonomic_last_state = "PARASYMPATHETIC"
        except Exception as exc:
            import logging as _log
            _log.getLogger("trollguard").debug("perimeter-threat deposit failed: %s", exc)

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
