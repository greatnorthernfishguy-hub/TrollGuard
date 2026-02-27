---
name: trollguard
description: "TrollGuard is a 4-layer security pipeline that protects AI agents from prompt injection and adversarial attacks. It implements a 'Zero Trust for Text' philosophy: every string of text entering the system is treated as a potential prompt injection until mathematically proven otherwise."
metadata: { "openclaw": { "emoji": "🛡️", "requires": {} } }
hook: trollguard_hook.py::get_instance
---

# TrollGuard Security

## Metadata
version: 0.1.0
autoload: true

## Architecture

### Layer 1: Static Analysis (Cisco Skill-Scanner)
- YARA rules, regex, and known malware signatures
- Catches ~90% of basic attacks instantly

### Layer 2: Sentinel ML Pipeline
- Converts text to 384‑D embeddings using all‑MiniLM‑L6‑v2
- Random Forest classifier trained on prompt‑injection datasets
- Platt‑scaled confidence scores (0.0–1.0)
- Thresholds: <0.3 SAFE, 0.3–0.7 SUSPICIOUS, >0.7 MALICIOUS

### Layer 3: Swarm Audit (Semantic Air Gap)
- **Agent A**: Sacrificial LLM reads suspicious code (can be compromised)
- **Semantic Air Gap**: Agent A's textual output destroyed, only embeddings passed
- **Agent B**: Judges Agent A's intent via vector similarity to known‑safe patterns
- **Agent C**: Meta‑auditor analyzing pipeline telemetry
- **Canary Protocol**: Unique token per agent; missing token = prompt injection detected

### Layer 4: Runtime Vector Sentry
- Sliding‑window real‑time firewall for live I/O
- Redacts or blocks malicious content in streaming text

## Integration with NeuroGraph
- Uses NG‑Lite for adaptive learning
- Peer Bridge shares threat intelligence across instances
- OpenClaw adapter provides standard `on_message()`, `recall()`, `stats()` interface

## Environment Variables
- `TROLLGUARD_WORKSPACE_DIR` — Workspace directory (default: `~/.openclaw/trollguard`)

## Usage
TrollGuard auto‑loads as an OpenClaw skill and scans all incoming messages.
Manual scanning:
```bash
python main.py scan /path/to/file.py
python main.py scan-url https://example.com/suspicious.txt
```

## Configuration
See `config.yaml` for thresholds, LLM backends, runtime sentry mode, and emergency stop settings.