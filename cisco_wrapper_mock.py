"""
TrollGuard — Cisco Skill-Scanner Mock Wrapper

Simulates the cisco-ai-defense/skill-scanner interface for development
before the real Cisco repository is cloned.  Includes an optional
threat-simulation mode for testing and development.

TODO: Replace this mock with the real Cisco import:
    from cisco_base.scanner import scan_file
See PRD §4.2 for integration strategy.

# ---- Changelog ----
# [2026-02-19] Claude (Opus 4.6) — Grok security audit: threat simulation.
#   What: Added pattern-based threat simulation that detects dangerous
#         imports, eval/exec, obfuscated strings, network access, and
#         dynamic code generation — toggled via TROLLGUARD_CISCO_SIMULATE
#         env var or enable_threat_simulation().
#   Why:  Grok flagged that the mock always returning SAFE makes Layer 1
#         a dead layer — no threats ever reach the pipeline from static
#         analysis.  Threat simulation gives the pipeline real signals
#         for testing and development without needing the actual Cisco
#         scanner.
#   How:  Regex-based pattern matching on file content.  Patterns are
#         drawn from common code-injection signatures (PRD §4.2).
#         Toggle on/off at runtime or via env var.  When simulation is
#         off, behavior is identical to the original always-SAFE mock.
#
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: Mock scan_file() that returns a ScanResult with is_safe=True.
#   Why:  Allows the full pipeline to run during development without
#         the Cisco scanner.  The mock always returns SAFE so Layer 1
#         passes through to Layer 2 for testing.
#   How:  Returns a namedtuple-like ScanResult.  Replace with the real
#         Cisco import when the cisco_base directory is populated.
# -------------------
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ScanResult:
    """Result from the Cisco skill-scanner (or mock).

    Attributes:
        is_safe: Whether the file passed static analysis.
        threats: List of threat names if any were detected.
        suspicious_lines: Lines flagged as suspicious but not blocking.
    """
    is_safe: bool = True
    threats: List[str] = None
    suspicious_lines: List[int] = None

    def __post_init__(self):
        if self.threats is None:
            self.threats = []
        if self.suspicious_lines is None:
            self.suspicious_lines = []


# ---------------------------------------------------------------------------
# Threat simulation toggle
# ---------------------------------------------------------------------------

_simulate_threats: bool = os.environ.get("TROLLGUARD_CISCO_SIMULATE", "").lower() in ("1", "true", "yes")


def enable_threat_simulation() -> None:
    """Enable pattern-based threat simulation (for testing/dev)."""
    global _simulate_threats
    _simulate_threats = True


def disable_threat_simulation() -> None:
    """Disable threat simulation (revert to always-SAFE mock)."""
    global _simulate_threats
    _simulate_threats = False


def is_simulation_enabled() -> bool:
    """Check whether threat simulation is currently active."""
    return _simulate_threats


# ---------------------------------------------------------------------------
# Threat patterns (subset of what the real Cisco scanner checks)
# ---------------------------------------------------------------------------

# Each pattern: (compiled_regex, threat_name)
_THREAT_PATTERNS: List[tuple] = [
    # Dangerous imports
    (re.compile(r"^\s*(import\s+(?:os|subprocess|shutil|ctypes|sys)|"
                r"from\s+(?:os|subprocess|shutil|ctypes)\s+import)",
                re.MULTILINE),
     "dangerous_import"),

    # eval/exec/compile
    (re.compile(r"\b(eval|exec|compile)\s*\(", re.MULTILINE),
     "dynamic_code_execution"),

    # pickle deserialization (arbitrary code execution)
    (re.compile(r"\bpickle\.(loads?|Unpickler)\s*\(", re.MULTILINE),
     "unsafe_deserialization"),

    # Base64 + decode chains (obfuscation indicator)
    (re.compile(r"base64\.(b64decode|decodebytes)\s*\(", re.MULTILINE),
     "obfuscated_payload"),

    # Network access patterns
    (re.compile(r"\b(socket\.socket|urllib\.request\.urlopen|"
                r"requests\.(get|post|put|delete)|"
                r"http\.client\.HTTP)", re.MULTILINE),
     "network_access"),

    # Shell injection vectors
    (re.compile(r"(os\.system|os\.popen|subprocess\.(call|run|Popen|check_output))\s*\(",
                re.MULTILINE),
     "shell_execution"),

    # __import__ (dynamic import, evasion technique)
    (re.compile(r"__import__\s*\(", re.MULTILINE),
     "dynamic_import"),
]


def _simulate_scan(file_path: str) -> ScanResult:
    """Pattern-based threat simulation on file content."""
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return ScanResult(is_safe=True, threats=[], suspicious_lines=[])

    try:
        content = p.read_text(errors="replace")
    except OSError:
        return ScanResult(is_safe=True, threats=[], suspicious_lines=[])

    threats: List[str] = []
    suspicious_lines: List[int] = []
    lines = content.split("\n")

    for pattern, threat_name in _THREAT_PATTERNS:
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                if threat_name not in threats:
                    threats.append(threat_name)
                if i not in suspicious_lines:
                    suspicious_lines.append(i)

    suspicious_lines.sort()

    return ScanResult(
        is_safe=len(threats) == 0,
        threats=threats,
        suspicious_lines=suspicious_lines,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_file(file_path: str) -> ScanResult:
    """Scan a file using the Cisco scanner mock (or threat simulation).

    When threat simulation is enabled (via TROLLGUARD_CISCO_SIMULATE=1
    or enable_threat_simulation()), runs pattern-based detection.
    Otherwise returns SAFE for all inputs.

    TODO: Replace with real Cisco scanner import:
        from cisco_base.scanner import scan_file as cisco_scan
        return cisco_scan(file_path)

    Args:
        file_path: Path to the file to scan.

    Returns:
        ScanResult with mock or simulated results.
    """
    if _simulate_threats:
        return _simulate_scan(file_path)

    # Default mock: always returns safe
    return ScanResult(is_safe=True, threats=[], suspicious_lines=[])
