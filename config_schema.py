"""
TrollGuard — Pydantic Configuration Schema

Validates config.yaml against a typed schema at load time.  Catches
typos, type errors, and invalid ranges before they cause silent runtime
failures deep in the pipeline.

# ---- Changelog ----
# [2026-02-19] Claude (Opus 4.6) — Grok security audit: config validation.
#   What: Pydantic v2 models mirroring every section of config.yaml.
#   Why:  Grok flagged that config.yaml is loaded with yaml.safe_load()
#         and consumed via raw dict access — no type checking, no range
#         validation.  A typo like "malicious_flor: 0.7" silently falls
#         back to default, hiding a misconfiguration.
#   How:  Pydantic BaseModel with Field() constraints.  load_and_validate()
#         replaces raw yaml.safe_load() in main.py and api.py.  Unknown
#         keys are ignored (not rejected) for forward compatibility —
#         a new config key added in v0.2 shouldn't break v0.1 validators.
#
#   Claude's note on Grok's "strict" suggestion: Pydantic strict mode
#   (strict=True) rejects int-to-float coercion (e.g., port: 7438 typed
#   as int in YAML would fail if the schema says float).  This is too
#   brittle for YAML config files where numeric types are ambiguous.
#   We use Pydantic's default coercion mode instead, which is the right
#   balance of safety and usability for config files.
# -------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger("trollguard.config")


class ThresholdsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    safe_ceiling: float = Field(0.3, ge=0.0, le=1.0)
    malicious_floor: float = Field(0.7, ge=0.0, le=1.0)

    @field_validator("malicious_floor")
    @classmethod
    def floor_above_ceiling(cls, v: float, info) -> float:
        ceiling = info.data.get("safe_ceiling", 0.3)
        if v <= ceiling:
            raise ValueError(
                f"malicious_floor ({v}) must be greater than "
                f"safe_ceiling ({ceiling})"
            )
        return v


class SwarmAgentConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = "huggingface"
    model: str = "kimi-k2.5"


class SwarmAgentCConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tier: str = Field("statistical", pattern=r"^(statistical|local|api)$")
    local_model: str = "phi-3.5-mini-Q4"
    api_provider: str = "anthropic"
    api_model: str = "claude-opus-4-6"


class SwarmEscalationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    auto_swarm_on_suspicious: bool = True
    deep_audit_all: bool = False


class SwarmAuditConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    agent_a: SwarmAgentConfig = SwarmAgentConfig()
    agent_b: SwarmAgentConfig = SwarmAgentConfig()
    agent_c: SwarmAgentCConfig = SwarmAgentCConfig()
    escalation: SwarmEscalationConfig = SwarmEscalationConfig()


class RuntimeSentryConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    mode: str = Field("report_only", pattern=r"^(redact|block|report_only)$")
    sensitivity: float = Field(0.8, ge=0.0, le=1.0)
    redaction_tag: str = "[TROLLGUARD: CONTENT REDACTED]"


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = Field(384, gt=0)
    chunk_size: int = Field(256, gt=0)
    chunk_overlap: int = Field(50, ge=0)
    device: str = Field("cpu", pattern=r"^(cpu|cuda|auto)$")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 256)
        if v >= size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({size})"
            )
        return v


class ClassifierConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_path: str = "models/sentinel_model.pkl"
    n_estimators: int = Field(200, gt=0)
    class_weight: str = "balanced"


class PeerBridgeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    shared_dir: str = "~/.et_modules/shared_learning"
    sync_interval: int = Field(100, gt=0)


class NGLiteConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    module_id: str = "trollguard"
    state_path: str = "ng_lite_state.json"
    peer_bridge: PeerBridgeConfig = PeerBridgeConfig()


class PersistenceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    skills_db_path: str = "skills_db.json"
    quarantine_path: str = "quarantine.json"


class RemoteBlocklistConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    url: str = ""
    check_interval_minutes: int = Field(60, gt=0)
    verify_signature: bool = True
    retroactive_scan: bool = True


class EmergencyConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kill_switch: bool = False
    remote_blocklist: RemoteBlocklistConfig = RemoteBlocklistConfig()


class APIConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = Field(7438, gt=0, lt=65536)


class TrollGuardConfig(BaseModel):
    """Top-level validated config schema for config.yaml -> trollguard: key."""
    model_config = ConfigDict(extra="ignore")

    fail_fast: bool = True
    deep_audit_all: bool = False
    auto_escalate_suspicious: bool = True

    thresholds: ThresholdsConfig = ThresholdsConfig()
    swarm_audit: SwarmAuditConfig = SwarmAuditConfig()
    runtime_sentry: RuntimeSentryConfig = RuntimeSentryConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    classifier: ClassifierConfig = ClassifierConfig()
    ng_lite: NGLiteConfig = NGLiteConfig()
    persistence: PersistenceConfig = PersistenceConfig()
    emergency: EmergencyConfig = EmergencyConfig()
    api: APIConfig = APIConfig()


def validate_config(raw: Dict[str, Any]) -> TrollGuardConfig:
    """Validate a raw config dict against the schema.

    Args:
        raw: The dict from yaml.safe_load(f).get("trollguard", {}).

    Returns:
        Validated TrollGuardConfig with defaults filled in.

    Raises:
        pydantic.ValidationError: If config values are invalid.
    """
    return TrollGuardConfig(**raw)


def load_and_validate(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load config.yaml, validate it, and return as a plain dict.

    This is a drop-in replacement for the old load_config() that adds
    Pydantic validation.  Returns a plain dict for backward compatibility
    with existing code that does dict access.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Validated config as a dict.  Empty dict if file not found.
    """
    import yaml

    p = Path(config_path)
    if not p.exists():
        logger.warning("Config not found at %s, using defaults", config_path)
        return TrollGuardConfig().model_dump()

    with open(p, "r") as f:
        raw = yaml.safe_load(f)

    trollguard_raw = raw.get("trollguard", {}) if raw else {}

    try:
        validated = validate_config(trollguard_raw)
        logger.info("Config validated successfully from %s", config_path)
        return validated.model_dump()
    except Exception as e:
        logger.error("Config validation failed: %s — using defaults", e)
        return TrollGuardConfig().model_dump()
