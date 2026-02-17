"""
TrollGuard — Cisco Skill-Scanner Mock Wrapper

Simulates the cisco-ai-defense/skill-scanner interface for development
before the real Cisco repository is cloned.

TODO: Replace this mock with the real Cisco import:
    from cisco_base.scanner import scan_file
See PRD §4.2 for integration strategy.

# ---- Changelog ----
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

from dataclasses import dataclass
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


def scan_file(file_path: str) -> ScanResult:
    """Mock scan_file — always returns SAFE.

    TODO: Replace with real Cisco scanner import:
        from cisco_base.scanner import scan_file as cisco_scan
        return cisco_scan(file_path)

    Args:
        file_path: Path to the file to scan.

    Returns:
        ScanResult with mock results.
    """
    # Mock: always returns safe
    # In production, this calls the Cisco skill-scanner library
    return ScanResult(is_safe=True, threats=[], suspicious_lines=[])
