# TrollGuard

**The Open-Source Immune System for AI Agents**

*(Formerly ClawGuard)*

TrollGuard is a local, open-source security layer that validates AI Agent Skills (OpenClaw/MoltBot) and monitors runtime I/O against adversarial attacks. It operates on a **"Zero Trust for Text"** philosophy: every string of text entering the system is treated as a potential prompt injection until mathematically proven otherwise.

## Architecture

Four-layer defense pipeline with **Fail Fast** architecture:

| Layer | Engine | Role | Speed |
|-------|--------|------|-------|
| 0 | Emergency Stop | Kill Switch | Instant |
| 1 | Cisco Skill-Scanner | Static Analysis (The Bouncer) | < 0.1s |
| 2 | Sentinel ML Pipeline | Vector Embeddings + Classifier (The Mind Reader) | ~0.5-1.0s |
| 3 | Swarm Audit (A, B, C) | Semantic Air Gap (The Interrogation Room) | 10-45s |
| 4 | Runtime Vector Sentry | Sliding Window on Live I/O (The Bodyguard) | Real-time |

### Core Innovation: The Semantic Air Gap

Agent A's text output is destroyed and converted to vector embeddings before Agent B sees it. Malicious instructions that compromised Agent A cannot propagate — they no longer exist as parseable text. An air gap built from linear algebra.

## E-T Systems Module Ecosystem

TrollGuard is part of the E-T Systems module ecosystem alongside [NeuroGraph](https://github.com/greatnorthernfishguy-hub/NeuroGraph) and [The-Inference-Difference](https://github.com/greatnorthernfishguy-hub/The-Inference-Difference).

### Three-Tier Learning Integration

- **Tier 1 (Standalone):** Each module vendors `ng_lite.py` for local Hebbian learning
- **Tier 2 (Peer):** `NGPeerBridge` connects co-located modules for shared learning — no full NeuroGraph required
- **Tier 3 (SaaS):** `NGSaaSBridge` connects to full NeuroGraph Foundation for cross-module STDP, hyperedges, and predictive coding

### Unified Module Management

The ET Module Manager (`et_modules/`) coordinates all modules:

```
et-modules update --all    # Update everything at once
et-modules status          # Check all module health
```

## Quick Start

```bash
# Clone
git clone https://github.com/greatnorthernfishguy-hub/TrollGuard.git
cd TrollGuard

# Install
./install.sh

# Train the ML classifier (requires datasets download)
python train_model.py

# Scan a file
python main.py scan /path/to/skill.py

# Real-time text scanning (stdin)
echo "suspicious text" | python main.py scan-text
```

## Project Structure

```
TrollGuard/
├── sentinel_core/           # 4-layer defense pipeline
│   ├── vector_sentry.py     # Layer 4: Runtime protection
│   ├── ml_classifier.py     # Layer 2: ML classification
│   ├── agent_swarm.py       # Layer 3: Swarm Audit + Semantic Air Gap
│   ├── canary_protocol.py   # Cryptographic canary tokens
│   ├── quarantine_logger.py # Incident logging
│   └── ast_extractor.py     # Python AST text extraction
├── et_modules/              # Unified module manager
│   └── manager.py           # Discovery, registration, updates
├── ng_lite.py               # Vendored NG-Lite learning substrate
├── ng_peer_bridge.py        # Tier 2 peer-to-peer module learning
├── main.py                  # Pipeline entry point + CLI
├── train_model.py           # ML model training script
├── config.yaml              # All configurable settings
├── cisco_wrapper_mock.py    # Cisco scanner mock (for development)
├── et_module.json           # ET Module Manager manifest
├── install.sh               # One-click installer
└── requirements.txt         # Python dependencies
```

## Configuration

All settings in `config.yaml`. Start with `report_only` mode to observe behavior before enabling automated blocking.

## License

GNU Affero General Public License v3.0
