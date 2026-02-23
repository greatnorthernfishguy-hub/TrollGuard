"""
TrollGuard — Unified Security Pipeline

Primary entry point for the TrollGuard 4-layer defense pipeline.
Orchestrates all layers in sequence with Fail Fast architecture:
if any layer returns UNSAFE, all subsequent layers are skipped.

  Layer 0: Emergency Stop (blocklist + kill switch)
  Layer 1: Cisco Skill-Scanner (static analysis)
  Layer 2: Sentinel ML Pipeline (vector embeddings + classifier)
  Layer 3: Swarm Audit (multi-agent with semantic air gap)
  Layer 4: Runtime Vector Sentry (sliding window on live I/O)

Integrates with NG-Lite for adaptive learning and NGPeerBridge
for cross-module intelligence sharing with sibling E-T Systems
modules on the same host.

PRD reference: Appendix A — Coding Agent Prompt, TASK 3

# ---- Changelog ----
# [2026-02-19] Claude (Opus 4.6) — Grok security audit: fcntl file locking.
#   What: Added fcntl.LOCK_EX on skills_db.json JSONL appends.
#   Why:  Grok flagged concurrent-append race on skills_db.json when
#         CLI and API write simultaneously.
#   How:  Reuses the same _file_lock context manager from quarantine_logger.
#
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: TrollGuardPipeline class orchestrating all 4 layers,
#         plus CLI entry point using Rich for formatted output.
#   Why:  This is the primary user-facing entry point.  "python main.py
#         scan /path/to/skill.py" runs the full pipeline and outputs
#         a consolidated JSON report.
#   Settings: Reads config.yaml for all settings.  Fail Fast is the
#         default: any UNSAFE at any layer stops the pipeline.
#   How:  Sequential layer execution with early exit.  Each layer
#         returns a LayerResult.  Results are aggregated into a
#         PipelineReport and logged to skills_db.json.
# -------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

logger = logging.getLogger("trollguard")


class Verdict(str, Enum):
    """Pipeline verdict."""
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"
    UNSAFE = "UNSAFE"
    ERROR = "ERROR"


@dataclass
class LayerResult:
    """Result from a single pipeline layer.

    Attributes:
        layer: Layer number (0-4).
        name: Layer name (e.g., "cisco_static", "sentinel_ml").
        verdict: SAFE/SUSPICIOUS/UNSAFE/ERROR.
        score: Numeric confidence score [0.0, 1.0] (if applicable).
        details: Layer-specific details.
        elapsed_ms: Time taken by this layer.
    """
    layer: int = 0
    name: str = ""
    verdict: Verdict = Verdict.SAFE
    score: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class PipelineReport:
    """Full pipeline report for a scanned file.

    Attributes:
        file_path: Path to the scanned file.
        file_hash: SHA-256 of the file content.
        final_verdict: The overall verdict (worst of all layers).
        layers_run: Which layers were executed.
        layer_results: Per-layer results.
        total_time_ms: Total pipeline execution time.
        fail_fast_triggered: Whether the pipeline stopped early.
        ng_lite_learning: Whether NG-Lite recorded this outcome.
    """
    file_path: str = ""
    file_hash: str = ""
    final_verdict: Verdict = Verdict.SAFE
    layers_run: List[int] = field(default_factory=list)
    layer_results: List[LayerResult] = field(default_factory=list)
    total_time_ms: float = 0.0
    fail_fast_triggered: bool = False
    ng_lite_learning: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output and skills_db logging."""
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "final_verdict": self.final_verdict.value,
            "layers_run": self.layers_run,
            "layer_results": [
                {
                    "layer": lr.layer,
                    "name": lr.name,
                    "verdict": lr.verdict.value,
                    "score": round(lr.score, 4),
                    "elapsed_ms": round(lr.elapsed_ms, 2),
                    "details": lr.details,
                }
                for lr in self.layer_results
            ],
            "total_time_ms": round(self.total_time_ms, 2),
            "fail_fast_triggered": self.fail_fast_triggered,
            "ng_lite_learning": self.ng_lite_learning,
        }


class TrollGuardPipeline:
    """Orchestrates the 4-layer defense pipeline.

    Usage:
        pipeline = TrollGuardPipeline(config)
        report = pipeline.scan_file("/path/to/skill.py")
        print(json.dumps(report.to_dict(), indent=2))
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._eco = None
        self._ng_lite = None
        self._peer_bridge = None
        self._init_ecosystem()

    def _init_ecosystem(self) -> None:
        """Initialize NGEcosystem (Tier 1→2→3 automatic progression)."""
        ng_config = self.config.get("ng_lite", {})
        if not ng_config.get("enabled", True):
            return

        try:
            import ng_ecosystem
            eco_config = {}
            peer_config = ng_config.get("peer_bridge", {})
            if peer_config:
                eco_config["peer_bridge"] = {
                    "enabled": peer_config.get("enabled", True),
                    "sync_interval": peer_config.get("sync_interval", 100),
                }
            state_path = ng_config.get("state_path", "ng_lite_state.json")
            self._eco = ng_ecosystem.init(
                module_id=ng_config.get("module_id", "trollguard"),
                state_path=state_path,
                config=eco_config,
            )
            # Expose internals for backward compat with api.py/stats
            self._ng_lite = self._eco._ng
            self._peer_bridge = self._eco._peer_bridge
            logger.info("NGEcosystem initialized at %s", self._eco.tier_name)
        except Exception as e:
            logger.warning("NGEcosystem not available: %s", e)

    def scan_file(self, file_path: str) -> PipelineReport:
        """Run the full pipeline on a file.

        Executes layers 0-3 in sequence.  If fail_fast is enabled
        (default), stops at the first UNSAFE verdict.

        Args:
            file_path: Path to the file to scan.

        Returns:
            PipelineReport with full audit trail.
        """
        start = time.time()
        p = Path(file_path)

        if not p.exists():
            return PipelineReport(
                file_path=file_path,
                final_verdict=Verdict.ERROR,
                layer_results=[LayerResult(
                    layer=0, name="file_check",
                    verdict=Verdict.ERROR,
                    details={"error": f"File not found: {file_path}"},
                )],
            )

        content = p.read_text(errors="replace")
        file_hash = hashlib.sha256(content.encode()).hexdigest()

        report = PipelineReport(
            file_path=str(p.resolve()),
            file_hash=file_hash,
        )

        fail_fast = self.config.get("fail_fast", True)

        # --- Layer 0: Emergency Stop ---
        result = self._layer0_emergency_stop(file_hash)
        report.layer_results.append(result)
        report.layers_run.append(0)
        if result.verdict == Verdict.UNSAFE:
            report.final_verdict = Verdict.UNSAFE
            report.fail_fast_triggered = True
            report.total_time_ms = (time.time() - start) * 1000.0
            self._log_to_skills_db(report)
            return report

        # --- Layer 1: Static Analysis (Cisco) ---
        result = self._layer1_static_analysis(file_path, content)
        report.layer_results.append(result)
        report.layers_run.append(1)
        if fail_fast and result.verdict == Verdict.UNSAFE:
            report.final_verdict = Verdict.UNSAFE
            report.fail_fast_triggered = True
            report.total_time_ms = (time.time() - start) * 1000.0
            self._log_to_skills_db(report)
            return report

        # --- Layer 2: Sentinel ML Pipeline ---
        result = self._layer2_ml_pipeline(content, file_path)
        report.layer_results.append(result)
        report.layers_run.append(2)
        if fail_fast and result.verdict == Verdict.UNSAFE:
            report.final_verdict = Verdict.UNSAFE
            report.fail_fast_triggered = True
            report.total_time_ms = (time.time() - start) * 1000.0
            self._log_to_skills_db(report)
            return report

        # --- Layer 3: Swarm Audit (if SUSPICIOUS) ---
        if result.verdict == Verdict.SUSPICIOUS:
            swarm_result = self._layer3_swarm_audit(content, file_path, result.score)
            report.layer_results.append(swarm_result)
            report.layers_run.append(3)
            if swarm_result.verdict == Verdict.UNSAFE:
                report.final_verdict = Verdict.UNSAFE
                report.fail_fast_triggered = fail_fast
                report.total_time_ms = (time.time() - start) * 1000.0
                self._log_to_skills_db(report)
                return report

        # --- Final verdict: worst of all layers ---
        verdicts = [lr.verdict for lr in report.layer_results]
        if Verdict.UNSAFE in verdicts:
            report.final_verdict = Verdict.UNSAFE
        elif Verdict.SUSPICIOUS in verdicts:
            report.final_verdict = Verdict.SUSPICIOUS
        else:
            report.final_verdict = Verdict.SAFE

        report.total_time_ms = (time.time() - start) * 1000.0

        # NG-Lite learning
        if self._ng_lite is not None:
            report.ng_lite_learning = True

        self._log_to_skills_db(report)

        return report

    def scan_text(self, text: str, source: str = "unknown") -> Dict[str, Any]:
        """Run Layer 4 (Runtime Sentry) on a text stream.

        Used for real-time protection of live I/O.

        Args:
            text: Raw text to scan.
            source: Label for the source (e.g., "web_search").

        Returns:
            Dict with verdict and sanitized text.
        """
        from sentinel_core.vector_sentry import VectorSentry

        sentry_config = self.config.get("runtime_sentry", {})
        sentry = VectorSentry(config=sentry_config, ng_lite=self._ng_lite)
        result = sentry.scan(text, source=source)

        return result.to_dict()

    # -------------------------------------------------------------------
    # Layer implementations
    # -------------------------------------------------------------------

    def _layer0_emergency_stop(self, file_hash: str) -> LayerResult:
        """Layer 0: Check kill switch and global blocklist."""
        start = time.time()

        emergency = self.config.get("emergency", {})

        # Kill switch
        if emergency.get("kill_switch", False):
            return LayerResult(
                layer=0, name="emergency_stop",
                verdict=Verdict.UNSAFE,
                details={"reason": "Emergency Stop Active — kill switch is ON"},
                elapsed_ms=(time.time() - start) * 1000.0,
            )

        # TODO: Remote blocklist check (PRD §12.1)
        # This requires network access and signature verification.
        # Placeholder for Phase 2+ implementation.

        return LayerResult(
            layer=0, name="emergency_stop",
            verdict=Verdict.SAFE,
            details={"kill_switch": False, "blocklist_checked": False},
            elapsed_ms=(time.time() - start) * 1000.0,
        )

    def _layer1_static_analysis(
        self, file_path: str, content: str
    ) -> LayerResult:
        """Layer 1: Cisco skill-scanner static analysis."""
        start = time.time()

        try:
            from cisco_wrapper_mock import scan_file
            result = scan_file(file_path)

            if not result.is_safe:
                return LayerResult(
                    layer=1, name="cisco_static",
                    verdict=Verdict.UNSAFE,
                    details={
                        "threats": result.threats,
                        "engine": "cisco_skill_scanner",
                    },
                    elapsed_ms=(time.time() - start) * 1000.0,
                )

            return LayerResult(
                layer=1, name="cisco_static",
                verdict=Verdict.SAFE,
                details={"engine": "cisco_skill_scanner"},
                elapsed_ms=(time.time() - start) * 1000.0,
            )

        except ImportError:
            # Cisco scanner not available — skip layer
            return LayerResult(
                layer=1, name="cisco_static",
                verdict=Verdict.SAFE,
                details={"engine": "unavailable", "skipped": True},
                elapsed_ms=(time.time() - start) * 1000.0,
            )

    def _layer2_ml_pipeline(
        self, content: str, file_path: str
    ) -> LayerResult:
        """Layer 2: AST extraction + vectorization + ML classification."""
        start = time.time()

        # Extract text segments
        from sentinel_core.ast_extractor import ASTExtractor
        extractor = ASTExtractor()
        suffix = Path(file_path).suffix.lower()
        file_type = "python" if suffix == ".py" else "text"
        extraction = extractor.extract_text(content, file_type=file_type, file_path=file_path)

        # Combine segments for classification
        combined_text = " ".join(
            seg["text"] for seg in extraction.segments
            if seg.get("text")
        )

        if not combined_text.strip():
            return LayerResult(
                layer=2, name="sentinel_ml",
                verdict=Verdict.SAFE, score=0.0,
                details={"segments": 0, "reason": "No extractable text"},
                elapsed_ms=(time.time() - start) * 1000.0,
            )

        # Chunk and classify
        from sentinel_core.vector_sentry import VectorSentry
        sentry = VectorSentry(ng_lite=self._ng_lite)
        sentry_result = sentry.scan(combined_text, source=file_path)

        # Map sentry verdict to pipeline verdict
        thresholds = self.config.get("thresholds", {})
        safe_ceiling = thresholds.get("safe_ceiling", 0.3)
        malicious_floor = thresholds.get("malicious_floor", 0.7)

        score = sentry_result.max_score

        if score >= malicious_floor:
            verdict = Verdict.UNSAFE
        elif score >= safe_ceiling:
            verdict = Verdict.SUSPICIOUS
        else:
            verdict = Verdict.SAFE

        return LayerResult(
            layer=2, name="sentinel_ml",
            verdict=verdict, score=score,
            details={
                "segments_extracted": extraction.total_segments,
                "chunks_scanned": sentry_result.chunks_scanned,
                "flagged_chunks": len(sentry_result.flagged_chunks),
                "extraction_method": extraction.extraction_method,
            },
            elapsed_ms=(time.time() - start) * 1000.0,
        )

    def _layer3_swarm_audit(
        self, content: str, file_path: str, layer2_score: float
    ) -> LayerResult:
        """Layer 3: Multi-agent swarm audit with semantic air gap."""
        start = time.time()

        from sentinel_core.agent_swarm import SwarmAudit, SwarmVerdict

        swarm_config = self.config.get("swarm_audit", {})
        swarm = SwarmAudit(config=swarm_config, ng_lite=self._ng_lite)
        result = swarm.audit(content, file_path=file_path, layer2_score=layer2_score)

        if result.verdict == SwarmVerdict.UNSAFE:
            verdict = Verdict.UNSAFE
        elif result.verdict == SwarmVerdict.SAFE:
            verdict = Verdict.SAFE
        else:
            verdict = Verdict.SUSPICIOUS

        return LayerResult(
            layer=3, name="swarm_audit",
            verdict=verdict,
            details=result.to_dict(),
            elapsed_ms=(time.time() - start) * 1000.0,
        )

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def _log_to_skills_db(self, report: PipelineReport) -> None:
        """Log scan result to skills_db.json."""
        db_path = self.config.get("persistence", {}).get(
            "skills_db_path", "skills_db.json"
        )

        entry = {
            "hash": report.file_hash,
            "verdict": report.final_verdict.value,
            "layers_triggered": report.layers_run,
            "last_scanned": time.time(),
            "report": report.to_dict(),
        }

        try:
            with open(db_path, "a") as f:
                if _HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(entry, default=str) + "\n")
                f.flush()
                if _HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError as e:
            logger.error("Failed to write to skills_db: %s", e)

    def save_ng_state(self) -> None:
        """Persist NG-Lite learning state via NGEcosystem."""
        if self._eco is not None:
            self._eco.save()
        elif self._ng_lite is not None:
            state_path = self.config.get("ng_lite", {}).get(
                "state_path", "ng_lite_state.json"
            )
            self._ng_lite.save(state_path)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load TrollGuard configuration from YAML with Pydantic validation.

    Falls back to raw dict loading if the config_schema module is
    unavailable (e.g., pydantic not installed).
    """
    try:
        from config_schema import load_and_validate
        return load_and_validate(config_path)
    except ImportError:
        logger.debug("config_schema not available, loading config without validation")

    p = Path(config_path)
    if not p.exists():
        logger.warning("Config not found at %s, using defaults", config_path)
        return {}

    with open(p, "r") as f:
        raw = yaml.safe_load(f)

    return raw.get("trollguard", {})


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TrollGuard — The Open-Source Immune System for AI Agents",
    )
    parser.add_argument(
        "command",
        choices=["scan", "scan-text", "status", "train"],
        help="Command to execute",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="File path to scan (for 'scan' command)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of Rich formatted output",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(name)s %(levelname)s: %(message)s",
    )

    config = load_config(args.config)
    pipeline = TrollGuardPipeline(config)

    if args.command == "scan":
        if not args.target:
            parser.error("scan command requires a target file path")

        report = pipeline.scan_file(args.target)

        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            _rich_print_report(report)

        pipeline.save_ng_state()

        # Exit code: 0 for SAFE, 1 for UNSAFE, 2 for SUSPICIOUS
        if report.final_verdict == Verdict.UNSAFE:
            sys.exit(1)
        elif report.final_verdict == Verdict.SUSPICIOUS:
            sys.exit(2)

    elif args.command == "scan-text":
        text = sys.stdin.read()
        result = pipeline.scan_text(text, source="stdin")
        print(json.dumps(result, indent=2))
        pipeline.save_ng_state()

    elif args.command == "status":
        _print_status(pipeline, config)

    elif args.command == "train":
        print("Training not yet implemented. See train_model.py (Phase 1).")
        sys.exit(0)


def _rich_print_report(report: PipelineReport) -> None:
    """Pretty-print a pipeline report using Rich (or fallback)."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel

        console = Console()

        # Verdict color
        color_map = {
            Verdict.SAFE: "green",
            Verdict.SUSPICIOUS: "yellow",
            Verdict.UNSAFE: "red",
            Verdict.ERROR: "red",
        }
        color = color_map.get(report.final_verdict, "white")

        console.print(Panel(
            f"[bold {color}]{report.final_verdict.value}[/]",
            title="TrollGuard Scan Result",
            subtitle=report.file_path,
        ))

        table = Table(title="Layer Results")
        table.add_column("Layer", style="cyan")
        table.add_column("Name", style="white")
        table.add_column("Verdict")
        table.add_column("Score", justify="right")
        table.add_column("Time (ms)", justify="right")

        for lr in report.layer_results:
            v_color = color_map.get(lr.verdict, "white")
            table.add_row(
                str(lr.layer),
                lr.name,
                f"[{v_color}]{lr.verdict.value}[/]",
                f"{lr.score:.4f}" if lr.score else "-",
                f"{lr.elapsed_ms:.1f}",
            )

        console.print(table)
        console.print(f"\nTotal time: {report.total_time_ms:.1f}ms")

        if report.fail_fast_triggered:
            console.print("[yellow]Pipeline stopped early (Fail Fast)[/]")

    except ImportError:
        # Rich not available — plain text output
        print(f"\n=== TrollGuard Scan Result ===")
        print(f"File:    {report.file_path}")
        print(f"Verdict: {report.final_verdict.value}")
        print(f"Time:    {report.total_time_ms:.1f}ms")
        for lr in report.layer_results:
            print(f"  Layer {lr.layer} ({lr.name}): {lr.verdict.value} (score={lr.score:.4f})")


def _print_status(pipeline: TrollGuardPipeline, config: Dict[str, Any]) -> None:
    """Print system status."""
    print("=== TrollGuard Status ===")
    print(f"Config loaded: {bool(config)}")

    if pipeline._eco is not None:
        print(f"Ecosystem: {pipeline._eco.tier_name}")
        eco_stats = pipeline._eco.stats()
        ng_lite = eco_stats.get("ng_lite", {})
        if ng_lite:
            print(f"  Nodes: {ng_lite.get('node_count', '?')}/{ng_lite.get('max_nodes', '?')}")
            print(f"  Synapses: {ng_lite.get('synapse_count', '?')}/{ng_lite.get('max_synapses', '?')}")
            print(f"  Outcomes: {ng_lite.get('total_outcomes', '?')}")
            sr = ng_lite.get('success_rate', 0)
            print(f"  Success rate: {sr:.2%}")
        if eco_stats.get("peer_bridge"):
            print(f"  Peer Bridge: connected")
        if eco_stats.get("ng_memory"):
            print(f"  NeuroGraph: connected (Tier 3)")
    else:
        print(f"NG-Lite: {'connected' if pipeline._ng_lite else 'not loaded'}")
        print(f"Peer Bridge: {'connected' if pipeline._peer_bridge else 'not loaded'}")
        if pipeline._ng_lite:
            stats = pipeline._ng_lite.get_stats()
            print(f"  Nodes: {stats['node_count']}/{stats['max_nodes']}")
            print(f"  Synapses: {stats['synapse_count']}/{stats['max_synapses']}")
            print(f"  Outcomes: {stats['total_outcomes']}")
            print(f"  Success rate: {stats['success_rate']:.2%}")

    # ET Module Manager status
    try:
        from et_modules.manager import ETModuleManager
        manager = ETModuleManager()
        modules = manager.discover()
        print(f"\nET Modules discovered: {len(modules)}")
        for mid, manifest in modules.items():
            print(f"  {mid}: v{manifest.version} at {manifest.install_path}")
    except Exception as e:
        print(f"\nET Module Manager: {e}")


if __name__ == "__main__":
    main()
