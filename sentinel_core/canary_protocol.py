"""
TrollGuard — Canary Protocol

Cryptographic canary token generation and validation for the Swarm
Audit pipeline (Layer 3).  Each agent invocation gets a unique token
that it must echo back in its JSON output.  Missing or altered canaries
confirm prompt injection — the malicious code successfully instructed
the agent to "ignore previous instructions."

PRD reference: Section 6.4 — The Canary Protocol

# ---- Changelog ----
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: CanaryProtocol class with generate(), validate(), and
#         build_system_prompt_block() methods.
#   Why:  The canary is both a detection mechanism (confirming
#         compromise) and a training signal (canary failures produce
#         high-value training data for the ML classifier).  Needed
#         before agent_swarm.py can be built.
#   Settings: Token length = 32 hex chars (128 bits of entropy).
#         Chosen to be long enough to resist brute-force guessing
#         but short enough to not waste LLM context.
#   How:  Uses secrets.token_hex() for cryptographic randomness.
#         Validation is strict per PRD: transport → format → canary
#         → verdict, in that order.
# -------------------
"""

from __future__ import annotations

import json
import logging
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger("trollguard.canary_protocol")


class CanaryStatus(str, Enum):
    """Outcome of canary validation."""
    ALIVE = "alive"               # Token present and matches
    DEAD = "dead"                 # Token missing or altered
    FORMAT_ERROR = "format_error" # Response not valid JSON
    TRANSPORT_ERROR = "transport_error"  # HTTP error / timeout


@dataclass
class CanaryResult:
    """Result of validating a canary token.

    Attributes:
        status: Whether the canary survived.
        token_expected: The token that was generated.
        token_received: The token found in the response (if any).
        verdict: The agent's SAFE/UNSAFE verdict (if parseable).
        reasoning: The agent's reasoning text (if parseable).
        data_flow: The agent's data_flow analysis (if parseable).
        behavior_flags: The agent's behavior flags (if parseable).
        raw_response: The full raw response text (for quarantine logging).
        error_detail: Description of what went wrong (if anything).
        validation_time_ms: How long validation took.
    """
    status: CanaryStatus = CanaryStatus.DEAD
    token_expected: str = ""
    token_received: str = ""
    verdict: str = ""
    reasoning: str = ""
    data_flow: str = ""
    behavior_flags: list = None
    raw_response: str = ""
    error_detail: str = ""
    validation_time_ms: float = 0.0

    def __post_init__(self):
        if self.behavior_flags is None:
            self.behavior_flags = []

    @property
    def is_safe(self) -> bool:
        """True only if canary is alive AND agent verdict is SAFE."""
        return (
            self.status == CanaryStatus.ALIVE
            and self.verdict.upper() == "SAFE"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for quarantine logging."""
        return {
            "status": self.status.value,
            "token_expected": self.token_expected,
            "token_received": self.token_received,
            "verdict": self.verdict,
            "reasoning_length": len(self.reasoning),
            "data_flow_length": len(self.data_flow),
            "behavior_flags": self.behavior_flags,
            "error_detail": self.error_detail,
            "validation_time_ms": round(self.validation_time_ms, 2),
        }


class CanaryProtocol:
    """Canary token generation and validation.

    Usage:
        cp = CanaryProtocol()

        # Generate a token for an agent invocation
        token = cp.generate()

        # Build the system prompt instruction for the agent
        prompt_block = cp.build_system_prompt_block(token)

        # ... invoke the agent with the prompt_block included ...

        # Validate the agent's response
        result = cp.validate(token, response_text, http_status=200)

        if result.status == CanaryStatus.DEAD:
            # Prompt injection confirmed
            ...
    """

    def __init__(self, token_length: int = 32):
        """
        Args:
            token_length: Length of hex token string.  32 hex chars =
                         128 bits of entropy — long enough to resist
                         guessing, short enough to not waste context.
        """
        self._token_length = token_length
        self._generation_count = 0
        self._dead_count = 0

    def generate(self) -> str:
        """Generate a cryptographically random canary token.

        Returns:
            A hex string of self._token_length characters.
        """
        self._generation_count += 1
        return secrets.token_hex(self._token_length // 2)

    def build_system_prompt_block(self, token: str) -> str:
        """Build the system prompt instruction mandating canary inclusion.

        This block is appended to the agent's system prompt.  The LLM
        is instructed to include the exact token in its JSON output.

        Args:
            token: The canary token to embed.

        Returns:
            A formatted string to include in the system prompt.
        """
        return (
            "\n\n--- MANDATORY OUTPUT REQUIREMENT ---\n"
            "You MUST include the following exact canary token in your "
            "JSON output under the key \"canary\".\n"
            f"Canary token: {token}\n"
            "Your output MUST be valid JSON with these required fields:\n"
            '  "canary": "<the exact token above>",\n'
            '  "verdict": "SAFE" or "UNSAFE",\n'
            '  "reasoning": "<your analysis>",\n'
            '  "data_flow": "<where data moves in the code>",\n'
            '  "behavior_flags": ["<list of suspicious patterns>"]\n'
            "--- END MANDATORY OUTPUT REQUIREMENT ---\n"
        )

    def validate(
        self,
        expected_token: str,
        response_text: str,
        http_status: int = 200,
    ) -> CanaryResult:
        """Validate a canary token in an agent's response.

        Validation follows the strict order from PRD §6.4:
          1. Transport Check — is the HTTP status OK?
          2. Format Check — is the response valid JSON?
          3. Canary Check — does the canary match?
          4. Verdict Check — read the agent's determination.

        Args:
            expected_token: The token that was generated.
            response_text: The raw response from the agent.
            http_status: The HTTP status code (default 200).

        Returns:
            CanaryResult with full validation details.
        """
        start = time.time()

        # 1. Transport Check
        if http_status != 200:
            elapsed = (time.time() - start) * 1000.0
            return CanaryResult(
                status=CanaryStatus.TRANSPORT_ERROR,
                token_expected=expected_token,
                raw_response=response_text[:500],
                error_detail=f"HTTP {http_status}",
                validation_time_ms=elapsed,
            )

        # 2. Format Check — must be valid JSON
        try:
            parsed = json.loads(response_text)
        except (json.JSONDecodeError, TypeError) as e:
            elapsed = (time.time() - start) * 1000.0
            logger.warning("Canary format check failed: %s", e)
            return CanaryResult(
                status=CanaryStatus.FORMAT_ERROR,
                token_expected=expected_token,
                raw_response=response_text[:500],
                error_detail=f"Model Output Destabilized: {e}",
                validation_time_ms=elapsed,
            )

        # 3. Canary Check — exact match required
        received_token = parsed.get("canary", "")
        if received_token != expected_token:
            elapsed = (time.time() - start) * 1000.0
            self._dead_count += 1
            logger.warning(
                "CANARY DIED: expected=%s received=%s",
                expected_token[:8] + "...",
                str(received_token)[:8] + "..." if received_token else "<missing>",
            )
            return CanaryResult(
                status=CanaryStatus.DEAD,
                token_expected=expected_token,
                token_received=str(received_token),
                raw_response=response_text[:500],
                error_detail="Prompt Injection Detected — Canary Died",
                validation_time_ms=(time.time() - start) * 1000.0,
            )

        # 4. Verdict Check — extract the agent's determination
        verdict = str(parsed.get("verdict", "")).strip()
        reasoning = str(parsed.get("reasoning", "")).strip()
        data_flow = str(parsed.get("data_flow", "")).strip()
        behavior_flags = parsed.get("behavior_flags", [])

        if not isinstance(behavior_flags, list):
            behavior_flags = [str(behavior_flags)]

        elapsed = (time.time() - start) * 1000.0

        return CanaryResult(
            status=CanaryStatus.ALIVE,
            token_expected=expected_token,
            token_received=received_token,
            verdict=verdict,
            reasoning=reasoning,
            data_flow=data_flow,
            behavior_flags=behavior_flags,
            raw_response=response_text[:2000],
            validation_time_ms=elapsed,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Protocol telemetry."""
        return {
            "tokens_generated": self._generation_count,
            "canaries_died": self._dead_count,
            "kill_rate": (
                self._dead_count / self._generation_count
                if self._generation_count > 0 else 0.0
            ),
        }
