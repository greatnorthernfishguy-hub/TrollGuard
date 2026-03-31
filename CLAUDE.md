# TrollGuard Repository
## Claude Code Onboarding — Repo-Specific

**You have already read the global `CLAUDE.md` and `ARCHITECTURE.md`.**
**If you have not, stop. Go read them. The Laws defined there govern this repo.**
**This document adds TrollGuard-specific rules on top of those Laws.**

---
## Vault Context
For full ecosystem context, read these from the Obsidian vault (`~/docs/`):
- **Module page:** `~/docs/modules/TrollGuard.md`
- **Concepts:** `~/docs/concepts/Autonomic State.md`, `~/docs/concepts/The River.md`, `~/docs/concepts/Vendored Files.md`
- **Systems:** `~/docs/systems/NG-Lite.md`, `~/docs/systems/NG Peer Bridge.md`
- **Audit:** `~/docs/audits/ecosystem-test-suite-audit-2026-03-23.md`

Each vault page has a Context Map at the top linking to related docs. Follow those links for ripple effects and dependencies.

---

## What This Repo Is

TrollGuard is the skin and perimeter of the E-T Systems digital organism. It validates AI Agent Skills and monitors runtime I/O against adversarial attacks using a **Zero Trust for Text** philosophy: every string of text entering the system is treated as a potential prompt injection until mathematically proven otherwise.

TrollGuard is one of only three modules (alongside Immunis and Cricket) authorized to write `SYMPATHETIC` to the autonomic state. This is a security privilege, not a general capability.

**Status: Deployed.** v0.1.0. Vendored files synced to NeuroGraph canonical. Registered in `~/.et_modules/`. Port 7438. Also runs as a TID sidecar for pre-route scanning.

---

## 1. Repository Structure

```
~/TrollGuard/
├── trollguard_hook.py          # OpenClaw skill entry point (TrollGuardHook singleton)
├── main.py                     # Pipeline entry point + CLI (TrollGuardPipeline)
├── api.py                      # API endpoint
├── et_module.json              # Module manifest (v2 schema)
├── config.yaml                 # All configurable settings
├── config_schema.py            # Configuration schema
├── train_model.py              # ML model training script
├── cisco_wrapper_mock.py       # Cisco scanner mock (development)
├── sentinel_core/              # 4-layer defense pipeline
│   ├── vector_sentry.py        # Layer 4: Runtime Vector Sentry (The Bodyguard)
│   ├── ml_classifier.py        # Layer 2: Sentinel ML classification
│   ├── agent_swarm.py          # Layer 3: Swarm Audit + Semantic Air Gap
│   ├── canary_protocol.py      # Cryptographic canary tokens
│   ├── quarantine_logger.py    # Incident logging
│   └── ast_extractor.py        # Python AST text extraction
├── et_modules/                 # ET Module Manager integration
│   └── __init__.py
├── tests/                      # Test suite
│   └── __init__.py
├── test_dangerous.py           # Dangerous input tests
├── test_injection.py           # Injection attack tests
├── test_skill.py               # Skill scanning tests
├── ng_lite.py                  # VENDORED — canonical from NeuroGraph
├── ng_peer_bridge.py           # VENDORED — canonical from NeuroGraph
├── ng_tract_bridge.py          # VENDORED — canonical from NeuroGraph (v0.3+, preferred)
├── ng_ecosystem.py             # VENDORED — canonical from NeuroGraph
├── ng_autonomic.py             # VENDORED — canonical from NeuroGraph
├── ng_embed.py                 # VENDORED — canonical from NeuroGraph
├── ng_updater.py               # VENDORED — auto-update + vendored file sync
├── openclaw_adapter.py         # VENDORED — canonical from NeuroGraph
├── install.sh                  # One-click installer
└── requirements.txt            # Python dependencies
```

---

## 2. The 4-Layer Defense Pipeline

TrollGuard uses a **Fail Fast** architecture: if any layer returns UNSAFE, all subsequent layers are skipped.

| Layer | Engine | File | Role | Speed |
|-------|--------|------|------|-------|
| 0 | Emergency Stop | `main.py` | Kill Switch (blocklist) | Instant |
| 1 | Cisco Skill-Scanner | `cisco_wrapper_mock.py` | Static Analysis (The Bouncer) | < 0.1s |
| 2 | Sentinel ML Pipeline | `sentinel_core/ml_classifier.py` | Vector Embeddings + Classifier (The Mind Reader) | ~0.5-1.0s |
| 3 | Swarm Audit (A, B, C) | `sentinel_core/agent_swarm.py` | Semantic Air Gap (The Interrogation Room) | 10-45s |
| 4 | Runtime Vector Sentry | `sentinel_core/vector_sentry.py` | Sliding Window on Live I/O (The Bodyguard) | Real-time |

### The Semantic Air Gap (Layer 3)

The core innovation. Agent A's text output is destroyed and converted to vector embeddings before Agent B sees it. Malicious instructions that compromised Agent A cannot propagate — they no longer exist as parseable text. An air gap built from linear algebra.

### Runtime Vector Sentry (Layer 4)

Real-time firewall on streaming text. Uses sliding-window vectorization with per-chunk ML classification. Integrates with NG-Lite for adaptive threat-pattern learning — records scan outcomes so the Hebbian substrate learns which vector patterns correspond to true positives vs false positives.

---

## 3. Key Architectural Constraint: TrollGuard Writes Autonomic State

TrollGuard is an **autonomic writer**. When a confirmed threat is detected, TrollGuard writes `SYMPATHETIC` to `ng_autonomic.py`. This is one of TrollGuard's core responsibilities as the organism's perimeter defense.

Only three modules may write autonomic state: **Immunis, TrollGuard, and Cricket**. All other modules are read-only.

---

## 4. Law 7 Compliance: Content-Derived Identifiers

TrollGuard's substrate learning uses content-derived identifiers, not category labels. When recording scan outcomes:

```python
target_id = f"scan:{sha256(embedding)[:16]}"
```

**Not:** `target_id = "threat:MALICIOUS"` (this was a Law 7 violation, fixed 2026-03-18)

The substrate learns from the actual threat pattern (the specific embedding), not from labels. All threats of the same label are no longer indistinguishable to the substrate.

---

## 5. Thresholds

```yaml
safe_ceiling: 0.3     # Below this → SAFE
malicious_floor: 0.7   # Above this → MALICIOUS
                        # Between → SUSPICIOUS (escalate to Layer 3)
```

These are initial values requiring calibration (PRD 5.4, 11.4). The runtime sentry ships in `report_only` mode. These thresholds are candidates for the competence model (Apprentice -> Journeyman -> Master) but that has not been wired in yet.

---

## 6. Vendored Files

Seven vendored files synced to NeuroGraph canonical:

| File | Purpose |
|------|---------|
| `ng_lite.py` | Tier 1 learning substrate |
| `ng_peer_bridge.py` | Tier 2 legacy fallback (JSONL-based) |
| `ng_tract_bridge.py` | Tier 2 preferred (per-pair directional tracts, v0.3+) |
| `ng_ecosystem.py` | Tier management lifecycle |
| `ng_autonomic.py` | Autonomic state (**TrollGuard: read AND write**) |
| `ng_embed.py` | Centralized embedding (Snowflake/snowflake-arctic-embed-m-v1.5, 768-dim) |
| `openclaw_adapter.py` | OpenClaw skill base class |

**Do not modify vendored files.** If TrollGuard needs different behavior, that behavior lives in TrollGuard-specific code (`sentinel_core/`, `trollguard_hook.py`, `main.py`), not in vendored files.

---

## 7. TID Sidecar Integration

TrollGuard runs as a sidecar within TID's request pipeline (`inference_difference/trollguard.py`). It scans incoming messages for threats at the pre-route hook stage. TrollGuard is a sidecar, not a gatekeeper — it filters alongside the flow, it does not dam it.

---

## 8. What TrollGuard Does NOT Do

- TrollGuard **never** repairs — THC's domain
- TrollGuard **never** monitors substrate health — Elmer's domain
- TrollGuard **never** detects host-level threats (process compromise, filesystem attacks) — Immunis's domain
- TrollGuard **never** calls other modules directly — Law 1
- TrollGuard **never** classifies experience before feeding it to the substrate — Law 7

TrollGuard's domain is **text-level threat filtering**. If a threat is not in the text stream, it is not TrollGuard's concern.

---

## 9. What Claude Code May and May Not Do

### Without Josh's Approval

**Permitted:**
- Read any file in the repo
- Run the test suite
- Edit TrollGuard-specific files (sentinel_core/, trollguard_hook.py, main.py, api.py, config_schema.py)
- Add or modify tests
- Update documentation

**Not permitted without explicit Josh approval:**
- Modify any vendored file
- Delete any file
- Change the pipeline layer order in main.py (the Fail Fast sequence matters)
- Modify autonomic write logic
- Restart any service
- Change the embedding model or dimension

---

## 10. Environment and Paths

| What | Where |
|------|-------|
| Repo root | `~/TrollGuard/` |
| Configuration | `~/TrollGuard/config.yaml` |
| Module manifest | `~/TrollGuard/et_module.json` |
| Module data (runtime) | `~/.et_modules/trollguard/` |
| Shared learning JSONL | `~/.et_modules/shared_learning/trollguard.jsonl` |
| Peer registry | `~/.et_modules/shared_learning/_peer_registry.json` |
| Virtual environment | `~/TrollGuard/venv/` |
| API port | 7438 |

---

*E-T Systems / TrollGuard*
*Last updated: 2026-03-26*
*Maintained by Josh — do not edit without authorization*
*Parent documents: `~/.claude/CLAUDE.md` (global), `~/.claude/ARCHITECTURE.md`*
