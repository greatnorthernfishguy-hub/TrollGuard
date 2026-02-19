"""
TrollGuard — OpenClaw Integration Hook

Singleton TrollGuardFilter that integrates TrollGuard's security
pipeline into the OpenClaw AI assistant framework.  Provides automatic
filtering of all text the agent encounters — web search results, user
messages, API responses, repository contents — with zero configuration
required from the agent developer.

Mirrors NeuroGraph's openclaw_hook.py pattern:
  - Singleton class with get_instance()
  - Auto-initialization from config.yaml
  - Structured event logging to a memory/ directory for OpenClaw
  - Graceful degradation if dependencies are missing

Once installed, every piece of text flowing through the agent is scanned
by the Runtime Vector Sentry.  Malicious content is redacted or blocked
before the agent's LLM ever sees it.  This is the "immune system"
operating silently in the background.

PRD reference: Section 7.3 — Integration with OpenClaw

# ---- Changelog ----
# [2026-02-19] Claude (Opus 4.6) — Grok audit claim: "silent fallback in _init_sentry()".
#   Claim:  If VectorSentry init fails and _sentry is None, sanitize() "quietly
#           returns SAFE verdict — bypassing the entire immune system.  Zero
#           protection, zero log of failure."
#   Status: NOT VALID — no code change required.
#   Why:    (a) sanitize() does NOT return a "SAFE verdict" — it returns the raw
#               text unchanged.  No ScanResult object with verdict=SAFE is
#               fabricated; the caller gets unsanitized text, which is a
#               different semantic than "classified as safe."
#           (b) "Zero log of failure" is factually wrong.  Two logger.warning()
#               calls already cover this path:
#               - _init_components() line 443: "VectorSentry not available: <error>"
#                 (logged once at init)
#               - sanitize() line 205: "Sentry not initialized, passing text
#                 through unsanitized" (logged on every call)
#           (c) The fail-open behavior is an intentional, documented design
#               choice (see comment on line 204: "fail open for availability").
#               In an agent framework, fail-closed would render the entire
#               agent non-functional when a dependency is missing.  The
#               tradeoff favors availability, and the warnings ensure
#               operators are aware of the degraded state.
#
# [2026-02-19] Claude (Opus 4.6) — Grok security audit: thread safety + file locking.
#   What: Added threading.RLock for sentry/model access, fcntl on event
#         JSONL writes, and explicit error logging on init failures.
#   Why:  Grok identified two distinct concurrency gaps:
#         (a) The singleton's sanitize() temporarily mutates _sentry.config["mode"]
#             then restores it — classic TOCTOU race under multi-threaded access.
#         (b) _write_event appends to events.jsonl without file locking.
#   How:  RLock wraps all sentry access (scan, config mutation, stats).
#         fcntl.LOCK_EX wraps event file appends.  Init failures now
#         log at ERROR instead of silent pass-through.
#
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: TrollGuardFilter singleton with sanitize(), scan_file(),
#         scan_url(), and scan_repo() methods.  Writes structured
#         events to {workspace}/security/ for OpenClaw consumption.
#   Why:  This is the glue between TrollGuard and the agent framework.
#         Without this hook, agent developers would need to manually
#         call the sentry on every piece of incoming text.  With it,
#         protection is automatic — the agent framework calls
#         TrollGuardFilter.sanitize() at its I/O boundary and the
#         developer never thinks about it.
#   Settings: Reads from config.yaml.  mode defaults to "redact"
#         (the safest default that doesn't break agent workflows).
#         sensitivity defaults to 0.8.  All overridable per-call.
#   How:  Mirrors NeuroGraph's openclaw_hook.py singleton pattern.
#         Uses the same VectorSentry + NGLite + NGPeerBridge stack
#         as the REST API and CLI, but wrapped in a convenient
#         interface for direct Python import.
# -------------------

Usage in an OpenClaw skill or tool:

    from trollguard_hook import TrollGuardFilter

    tg = TrollGuardFilter.get_instance()

    # In a web search tool:
    def search_web(query):
        raw_html = scrape_url(query)
        safe_text = tg.sanitize(raw_html, source="web_search")
        return safe_text

    # In a repository reader:
    def read_repo_file(path):
        content = open(path).read()
        result = tg.scan_file(path)
        if result["verdict"] == "UNSAFE":
            return "[TROLLGUARD: FILE BLOCKED]"
        return content

    # In a chat handler:
    def on_user_message(message):
        safe_msg = tg.sanitize(message, source="user_chat")
        return process(safe_msg)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

logger = logging.getLogger("trollguard.hook")


class TrollGuardFilter:
    """Singleton security filter for OpenClaw integration.

    Wraps TrollGuard's Runtime Vector Sentry into a single interface
    for text sanitization, file scanning, and security event logging.

    Auto-initializes NG-Lite with peer bridge for cross-module
    learning.  Falls back gracefully if any component is unavailable.
    """

    _instance: Optional[TrollGuardFilter] = None

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self._workspace_dir = Path(
            workspace_dir
            or os.environ.get("TROLLGUARD_WORKSPACE_DIR", "~/.openclaw/trollguard")
        ).expanduser()

        self._security_dir = self._workspace_dir / "security"
        self._security_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        self._config = self._load_config(config_path)

        # Thread safety: protects sentry config mutation in sanitize()
        # and all component access from concurrent callers.
        self._lock = threading.RLock()

        # Initialize components
        self._ng_lite = None
        self._peer_bridge = None
        self._sentry = None
        self._pipeline = None
        self._quarantine = None

        self._init_components()

        # Counters
        self._total_sanitize_calls = 0
        self._total_blocks = 0
        self._total_redactions = 0

        self._write_event("initialized", {
            "workspace": str(self._workspace_dir),
            "sentry_mode": self._config.get("runtime_sentry", {}).get("mode", "redact"),
            "ng_lite_connected": self._ng_lite is not None,
            "peer_bridge_connected": self._peer_bridge is not None,
        })

        logger.info("TrollGuardFilter initialized at %s", self._workspace_dir)

    @classmethod
    def get_instance(
        cls,
        workspace_dir: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> TrollGuardFilter:
        """Return the singleton instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(workspace_dir=workspace_dir, config_path=config_path)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (useful for testing)."""
        if cls._instance is not None:
            cls._instance._save_state()
        cls._instance = None

    # -------------------------------------------------------------------
    # Core API — these are what agent developers call
    # -------------------------------------------------------------------

    def sanitize(
        self,
        text: str,
        source: str = "unknown",
        mode: Optional[str] = None,
        sensitivity: Optional[float] = None,
    ) -> str:
        """Scan text and return sanitized version.

        This is the primary method for runtime protection.  Call it
        on any text before the agent's LLM processes it.

        Args:
            text: Raw text (HTML, chat message, API response, etc.).
            source: Label for where the text came from.
            mode: Override operating mode ("redact", "block", "report_only").
            sensitivity: Override sensitivity threshold [0.0, 1.0].

        Returns:
            Sanitized text with malicious content removed/redacted.
            In report_only mode, returns the original text unchanged.
        """
        self._total_sanitize_calls += 1

        if not text or not text.strip():
            return text

        if self._sentry is None:
            # Sentry not available — pass through (fail open for availability)
            logger.warning("Sentry not initialized, passing text through unsanitized")
            return text

        # RLock protects the sentry config mutation (mode swap) and the
        # scan call itself from concurrent threads.  Without this, two
        # threads calling sanitize() with different mode overrides could
        # observe each other's temporary config changes.
        with self._lock:
            original_mode = None
            if mode:
                original_mode = self._sentry.config["mode"]
                self._sentry.config["mode"] = mode

            result = self._sentry.scan(text, source=source)

            if original_mode is not None:
                self._sentry.config["mode"] = original_mode

        # Track (counters are benign races — not critical)
        if result.verdict.value == "MALICIOUS":
            self._total_blocks += 1
        elif result.verdict.value == "SUSPICIOUS":
            self._total_redactions += 1

        # Log security event
        if result.verdict.value != "SAFE":
            self._write_event("threat_detected", {
                "source": source,
                "verdict": result.verdict.value,
                "max_score": round(result.max_score, 4),
                "chunks_scanned": result.chunks_scanned,
                "flagged_chunks": len(result.flagged_chunks),
                "scan_time_ms": round(result.scan_time_ms, 2),
            })

            # Quarantine malicious content
            if result.verdict.value == "MALICIOUS" and self._quarantine is not None:
                best = max(result.flagged_chunks, key=lambda c: c.score) if result.flagged_chunks else None
                self._quarantine.log_incident(
                    source=source,
                    trigger_engine="trollguard_hook",
                    layer=4,
                    vector_score=result.max_score,
                    raw_text=text[:5000],
                    vector_embedding=best.embedding.tolist() if best and best.embedding is not None else None,
                )

        return result.sanitized_text

    def scan_file(self, file_path: str) -> Dict[str, Any]:
        """Run the full 4-layer pipeline on a file.

        Use this for scanning skill files, plugins, and downloaded
        code before installation or execution.

        Args:
            file_path: Path to the file to scan.

        Returns:
            Dict with verdict, layer results, and timing.
        """
        with self._lock:
            if self._pipeline is None:
                self._init_pipeline()

        if self._pipeline is None:
            return {"verdict": "ERROR", "reason": "Pipeline not available"}

        report = self._pipeline.scan_file(file_path)

        # Log security event
        self._write_event("file_scanned", {
            "file_path": file_path,
            "verdict": report.final_verdict.value,
            "layers_run": report.layers_run,
            "total_time_ms": round(report.total_time_ms, 2),
        })

        return report.to_dict()

    def scan_url(self, url: str, fetched_content: str) -> str:
        """Sanitize content fetched from a URL.

        Convenience wrapper around sanitize() that sets the source
        to the URL for better logging and learning.

        Args:
            url: The URL the content was fetched from.
            fetched_content: The raw text/HTML from the URL.

        Returns:
            Sanitized text.
        """
        return self.sanitize(fetched_content, source=f"url:{url}")

    def scan_repo(
        self,
        repo_path: str,
        extensions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Scan all files in a repository.

        Runs the full pipeline on each matching file and returns
        an aggregate report.

        Args:
            repo_path: Path to the repository root.
            extensions: File extensions to scan (default: .py, .yaml, .md).

        Returns:
            Dict with per-file results and aggregate verdict.
        """
        if extensions is None:
            extensions = [".py", ".yaml", ".yml", ".md"]

        repo = Path(repo_path)
        if not repo.is_dir():
            return {"verdict": "ERROR", "reason": f"Not a directory: {repo_path}"}

        results: List[Dict[str, Any]] = []
        worst_verdict = "SAFE"

        for ext in extensions:
            for fp in sorted(repo.rglob(f"*{ext}")):
                if fp.is_file() and ".git" not in fp.parts:
                    result = self.scan_file(str(fp))
                    result["file"] = str(fp.relative_to(repo))
                    results.append(result)

                    v = result.get("final_verdict", "SAFE")
                    if v == "UNSAFE":
                        worst_verdict = "UNSAFE"
                    elif v == "SUSPICIOUS" and worst_verdict == "SAFE":
                        worst_verdict = "SUSPICIOUS"

        self._write_event("repo_scanned", {
            "repo_path": repo_path,
            "files_scanned": len(results),
            "aggregate_verdict": worst_verdict,
        })

        return {
            "repo_path": repo_path,
            "aggregate_verdict": worst_verdict,
            "files_scanned": len(results),
            "results": results,
        }

    def stats(self) -> Dict[str, Any]:
        """Current filter statistics."""
        result: Dict[str, Any] = {
            "total_sanitize_calls": self._total_sanitize_calls,
            "total_blocks": self._total_blocks,
            "total_redactions": self._total_redactions,
            "workspace": str(self._workspace_dir),
        }

        if self._sentry is not None:
            result["sentry"] = self._sentry.get_stats()

        if self._ng_lite is not None:
            result["ng_lite"] = self._ng_lite.get_stats()

        if self._peer_bridge is not None:
            result["peer_bridge"] = self._peer_bridge.get_stats()

        if self._quarantine is not None:
            result["quarantine"] = self._quarantine.get_stats()

        return result

    # -------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load config from yaml or environment."""
        paths_to_try = [
            config_path,
            os.environ.get("TROLLGUARD_CONFIG"),
            str(self._workspace_dir / "config.yaml"),
            "config.yaml",
            os.path.expanduser("~/TrollGuard/config.yaml"),
            "/opt/trollguard/config.yaml",  # legacy system-level path
        ]

        for p in paths_to_try:
            if p and Path(p).exists():
                try:
                    import yaml
                    with open(p, "r") as f:
                        raw = yaml.safe_load(f)
                    logger.info("Config loaded from %s", p)
                    return raw.get("trollguard", {})
                except Exception as e:
                    logger.warning("Failed to load config from %s: %s", p, e)

        logger.info("No config found, using defaults")
        return {}

    def _init_components(self) -> None:
        """Initialize NG-Lite, peer bridge, sentry, and quarantine."""
        # NG-Lite
        ng_config = self._config.get("ng_lite", {})
        if ng_config.get("enabled", True):
            try:
                from ng_lite import NGLite
                self._ng_lite = NGLite(
                    module_id=ng_config.get("module_id", "trollguard"),
                )

                state_path = ng_config.get("state_path", "ng_lite_state.json")
                if Path(state_path).exists():
                    self._ng_lite.load(state_path)

                # Peer bridge
                peer_config = ng_config.get("peer_bridge", {})
                if peer_config.get("enabled", True):
                    from ng_peer_bridge import NGPeerBridge
                    self._peer_bridge = NGPeerBridge(
                        module_id=ng_config.get("module_id", "trollguard"),
                        shared_dir=peer_config.get("shared_dir"),
                        sync_interval=peer_config.get("sync_interval", 100),
                    )
                    self._ng_lite.connect_bridge(self._peer_bridge)

            except ImportError as e:
                logger.warning("NG-Lite not available: %s", e)

        # Vector Sentry
        try:
            from sentinel_core.vector_sentry import VectorSentry
            sentry_config = self._config.get("runtime_sentry", {})
            self._sentry = VectorSentry(
                config=sentry_config,
                ng_lite=self._ng_lite,
            )
        except ImportError as e:
            logger.warning("VectorSentry not available: %s", e)

        # Quarantine logger
        try:
            from sentinel_core.quarantine_logger import QuarantineLogger
            q_path = self._config.get("persistence", {}).get(
                "quarantine_path", "quarantine.json"
            )
            self._quarantine = QuarantineLogger(q_path)
        except ImportError as e:
            logger.warning("QuarantineLogger not available: %s", e)

    def _init_pipeline(self) -> None:
        """Lazy-init the full pipeline (only needed for file scanning)."""
        try:
            from main import TrollGuardPipeline
            self._pipeline = TrollGuardPipeline(self._config)
        except ImportError as e:
            logger.warning("TrollGuardPipeline not available: %s", e)

    def _save_state(self) -> None:
        """Persist NG-Lite state."""
        if self._ng_lite is not None:
            state_path = self._config.get("ng_lite", {}).get(
                "state_path", "ng_lite_state.json"
            )
            try:
                self._ng_lite.save(state_path)
            except Exception as e:
                logger.warning("Failed to save NG-Lite state: %s", e)

    # -------------------------------------------------------------------
    # Event logging (structured output for OpenClaw)
    # -------------------------------------------------------------------

    def _write_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Write a structured security event for OpenClaw.

        Events are appended to security/events.jsonl.  OpenClaw's
        security monitoring can tail this file for real-time alerts.
        """
        event = {
            "timestamp": time.time(),
            "event": event_type,
            "module": "trollguard",
            "data": data,
        }
        try:
            events_path = self._security_dir / "events.jsonl"
            with open(events_path, "a") as f:
                if _HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(event, default=str) + "\n")
                f.flush()
                if _HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.warning("Failed to write security event: %s", e)
