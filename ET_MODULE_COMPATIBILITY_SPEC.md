# E-T Systems Module Compatibility Specification v1.0

**Status:** Normative — all modules MUST comply.
**Maintained by:** TrollGuard repository (canonical source of truth).
**Last updated:** 2026-02-19

---

## Purpose

This document defines the exact contract that every E-T Systems module must
implement to participate in the unified ecosystem: shared learning, coordinated
updates, cross-module discovery, and future GUI updater integration.

**If you are a Claude Code instance starting work on a new or existing module,
this is your primary integration reference.** Follow it precisely. The existing
`ET_MODULE_INTEGRATION_PRIMER.md` is a tutorial; this document is the law.

---

## Table of Contents

1. [Module Identity: et_module.json](#1-module-identity-et_modulejson)
2. [Directory Layout Contract](#2-directory-layout-contract)
3. [Shared Filesystem Contract](#3-shared-filesystem-contract)
4. [NG-Lite & Peer Bridge Integration](#4-ng-lite--peer-bridge-integration)
5. [Installer Requirements](#5-installer-requirements)
6. [Update Mechanism Compliance](#6-update-mechanism-compliance)
7. [API Conventions (if module exposes HTTP)](#7-api-conventions-if-module-exposes-http)
8. [Changelog Format](#8-changelog-format)
9. [Graceful Degradation Rules](#9-graceful-degradation-rules)
10. [GUI Updater Readiness](#10-gui-updater-readiness)
11. [Known Modules & Reserved Identifiers](#11-known-modules--reserved-identifiers)
12. [Compliance Checklist](#12-compliance-checklist)

---

## 1. Module Identity: et_module.json

Every module MUST have an `et_module.json` file in its project root. This is
the machine-readable manifest that the ET Module Manager, GUI updater, and
peer bridge use to discover and coordinate modules.

### 1.1 Schema (all fields required unless marked optional)

```json
{
  "module_id":        "string — unique lowercase identifier, snake_case",
  "display_name":     "string — human-readable name",
  "version":          "string — semver (MAJOR.MINOR.PATCH)",
  "description":      "string — one-line description",
  "install_path":     "string — absolute path, populated at install time (empty in repo)",
  "git_remote":       "string — HTTPS clone URL for updates",
  "git_branch":       "string — branch to track (typically 'main')",
  "entry_point":      "string — main executable relative to install_path",
  "ng_lite_version":  "string — version of vendored ng_lite.py",
  "dependencies":     ["string array — module_ids this module requires"],
  "service_name":     "string — systemd unit name, empty if no service",
  "api_port":         "integer — HTTP port, 0 if no API"
}
```

### 1.2 Field Rules

| Field | Constraint | Example |
|-------|-----------|---------|
| `module_id` | `[a-z][a-z0-9_]*`, max 32 chars, globally unique across ecosystem | `"trollguard"`, `"inference_difference"`, `"neurograph"` |
| `version` | Strict semver. Bump MINOR for features, PATCH for fixes, MAJOR for breaking changes. | `"0.1.0"` |
| `install_path` | MUST be empty string (`""`) in the committed repo. Populated by installer at deploy time. | `""` in repo, `"/home/user/TrollGuard"` after install |
| `git_remote` | HTTPS only. SSH URLs break automated update checks. | `"https://github.com/greatnorthernfishguy-hub/TrollGuard.git"` |
| `git_branch` | The branch `ETModuleManager.update_all()` will `git pull`. Usually `"main"`. | `"main"` |
| `ng_lite_version` | Must match the `__version__` string in the vendored `ng_lite.py`. Used for compatibility checks. | `"1.0.0"` |
| `dependencies` | Other `module_id` values. Empty array if standalone. Manager will warn (not block) if deps are missing. | `[]` or `["neurograph"]` |
| `service_name` | Must match the systemd unit filename (without `.service`). Empty string if module has no daemon. | `"trollguard"` |
| `api_port` | Port the module's HTTP API binds to. `0` if module has no API. No two modules may claim the same port. | `7438` |

### 1.3 Reserved Ports

| Port | Module |
|------|--------|
| 7438 | TrollGuard |
| 7439 | The-Inference-Difference |
| 7440 | NeuroGraph |
| 7441-7449 | Reserved for future modules |

If your module needs an API, pick the next unreserved port from the table
above and add it here via PR.

### 1.4 Example (TrollGuard's actual manifest)

```json
{
  "module_id": "trollguard",
  "display_name": "TrollGuard",
  "version": "0.1.0",
  "description": "The Open-Source Immune System for AI Agents",
  "install_path": "",
  "git_remote": "https://github.com/greatnorthernfishguy-hub/TrollGuard.git",
  "git_branch": "main",
  "entry_point": "main.py",
  "ng_lite_version": "1.0.0",
  "dependencies": [],
  "service_name": "trollguard",
  "api_port": 7438
}
```

---

## 2. Directory Layout Contract

### 2.1 Required Files (every module)

```
<project_root>/
├── et_module.json          # Module manifest (Section 1)
├── ng_lite.py              # Vendored NG-Lite v1.0.0 (Section 4)
├── ng_peer_bridge.py       # Vendored NGPeerBridge (Section 4)
├── install.sh              # Installer with ET registration (Section 5)
├── requirements.txt        # Python dependencies
└── <entry_point>           # Whatever et_module.json says (e.g., main.py)
```

### 2.2 Optional But Recommended

```
<project_root>/
├── config.yaml             # Module config (Section 7)
├── et_modules/             # Copy of ET Module Manager package
│   ├── __init__.py
│   └── manager.py
└── <module_hook>.py        # OpenClaw integration hook (if applicable)
```

### 2.3 File Provenance Rules

| File | Source | Update Mechanism |
|------|--------|-----------------|
| `ng_lite.py` | Vendored from NeuroGraph canonical repo | `et-modules update --all` re-vendors from NeuroGraph, OR manual copy |
| `ng_peer_bridge.py` | Vendored from TrollGuard canonical repo | Same as above |
| `et_modules/manager.py` | Vendored from TrollGuard canonical repo | Same as above |
| `et_module.json` | Authored by each module | Each module maintains its own |
| `install.sh` | Authored by each module | Each module maintains its own |

**Critical:** `ng_lite.py` MUST be identical across all modules on the same
host. Version mismatches between modules will cause peer bridge embedding
dimension mismatches and silent learning failures. The `ng_lite_version`
field in `et_module.json` exists specifically so the manager can detect and
warn about this.

---

## 3. Shared Filesystem Contract

### 3.1 Root Directory

```
~/.et_modules/                              # ET_MODULES_DIR
```

Environment variable override: `ET_MODULES_DIR`.
Default: `$HOME/.et_modules/`.
All modules MUST use this same root. No exceptions.

### 3.2 Directory Structure

```
~/.et_modules/
├── registry.json                           # Module registry (Section 3.3)
├── shared_learning/                        # NGPeerBridge event logs (Section 3.4)
│   ├── trollguard.jsonl                    # TrollGuard's events
│   ├── inference_difference.jsonl          # TID's events
│   ├── neurograph.jsonl                    # NeuroGraph's events
│   ├── <module_id>.jsonl                   # Any module's events
│   └── _peer_registry.json                # Active peer discovery
└── modules/                                # Optional: alternative install location
    └── <module_id>/                        # Module installed here if not in ~/
```

### 3.3 Registry Schema (~/.et_modules/registry.json)

```json
{
  "last_updated": 1708300000.0,
  "modules": {
    "<module_id>": {
      "module_id": "string",
      "display_name": "string",
      "version": "string",
      "description": "string",
      "install_path": "string — absolute path",
      "git_remote": "string",
      "git_branch": "string",
      "entry_point": "string",
      "ng_lite_version": "string",
      "dependencies": [],
      "service_name": "string",
      "api_port": 0,
      "registered_at": 1708300000.0
    }
  }
}
```

**Who writes this:** Each module's `install.sh` writes its own entry.
`ETModuleManager.discover()` also updates it when scanning known locations.

**Concurrency:** Multiple installers may run simultaneously. Use
read-modify-write with file locking (fcntl on POSIX). See Section 4.5.

### 3.4 Shared Learning Event Schema (<module_id>.jsonl)

Each line is a self-contained JSON object:

```json
{
  "timestamp": 1708300000.0,
  "module_id": "trollguard",
  "target_id": "MALICIOUS",
  "success": true,
  "embedding": [0.123, -0.456, ...],
  "metadata": {}
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `timestamp` | float | YES | Unix epoch (time.time()) |
| `module_id` | string | YES | Must match the writing module's `module_id` |
| `target_id` | string | YES | What was decided (module-specific meaning) |
| `success` | bool | YES | Whether the outcome was correct |
| `embedding` | float[] | YES | The input embedding vector (dimensionality must match `ng_lite.py` config) |
| `metadata` | object | NO | Arbitrary module-specific context |

**Dimensionality rule:** All embeddings in shared_learning/ MUST have the
same dimensionality. The current standard is **384** (all-MiniLM-L6-v2).
If a module uses a different embedding model, it MUST project to 384 dims
before writing to shared_learning/. Cross-module similarity computation
is meaningless with mismatched dimensions.

### 3.5 Peer Registry Schema (_peer_registry.json)

```json
{
  "modules": {
    "<module_id>": {
      "registered_at": 1708300000.0,
      "event_file": "/home/user/.et_modules/shared_learning/trollguard.jsonl",
      "pid": 12345
    }
  }
}
```

Written by `NGPeerBridge.__init__()` at startup. Used for active peer discovery.
The `pid` field enables stale-peer detection (check if process is still alive).

---

## 4. NG-Lite & Peer Bridge Integration

### 4.1 Tier Architecture

```
Tier 1 (Standalone):  ng_lite.py alone — local Hebbian learning, zero deps.
Tier 2 (Peer):        ng_lite.py + ng_peer_bridge.py — shared learning via filesystem.
Tier 3 (SaaS/Full):   ng_lite.py + NGSaaSBridge — full NeuroGraph SNN.
```

Every module MUST support Tier 1. Every module SHOULD support Tier 2.
Tier 3 is optional (requires NeuroGraph to be running).

The module code MUST NOT care which tier is active. The `NGBridge` ABC
is the universal interface. Tier transitions are transparent.

### 4.2 Required Integration Pattern

```python
from ng_lite import NGLite
from ng_peer_bridge import NGPeerBridge

# 1. Create NG-Lite with YOUR module's ID
ng = NGLite(module_id="<your_module_id>")

# 2. Load persisted state if available
state_path = "<your_module_id>_ng_state.json"  # or from config
if Path(state_path).exists():
    ng.load(state_path)

# 3. Connect peer bridge (Tier 2)
try:
    bridge = NGPeerBridge(
        module_id="<your_module_id>",
        shared_dir="~/.et_modules/shared_learning",
        sync_interval=100,
    )
    ng.connect_bridge(bridge)
except Exception:
    pass  # Tier 1 fallback — local learning continues

# 4. Use ng.record_outcome() / ng.get_recommendations() / ng.detect_novelty()
#    exactly as documented in ng_lite.py.  The bridge is transparent.

# 5. On shutdown: save state
ng.save(state_path)
```

### 4.3 Module ID Conventions for NG-Lite

The `module_id` passed to `NGLite()` and `NGPeerBridge()` MUST match the
`module_id` in `et_module.json`. This is how peer bridge events are attributed
and how cross-module similarity filtering works.

### 4.4 Vendoring Rules

1. Copy `ng_lite.py` and `ng_peer_bridge.py` into your project root.
2. Do NOT modify them. If you need changes, upstream to the canonical repos.
3. The canonical source for `ng_lite.py` is NeuroGraph:
   `https://github.com/greatnorthernfishguy-hub/NeuroGraph`
4. The canonical source for `ng_peer_bridge.py` is TrollGuard:
   `https://github.com/greatnorthernfishguy-hub/TrollGuard`
5. When `et-modules update --all` runs, it SHOULD re-vendor these files
   from the canonical sources to keep all modules in sync.

### 4.5 File Locking (POSIX)

All JSONL and JSON file operations in shared_learning/ MUST use `fcntl`
advisory locks to prevent corruption from concurrent module writers.

```python
import fcntl

# Writing (exclusive lock)
with open(path, "a") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    f.write(json.dumps(record) + "\n")
    f.flush()
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

# Reading (shared lock)
with open(path, "r") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
    data = f.read()
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

On non-POSIX systems (Windows), locks should be no-ops with a debug log.
Use the pattern:

```python
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False
```

This pattern is already implemented in TrollGuard's `quarantine_logger.py`,
`ng_peer_bridge.py`, `main.py`, and `trollguard_hook.py`. Copy the pattern.

---

## 5. Installer Requirements

Every module's `install.sh` (or equivalent) MUST perform these steps.

### 5.1 Mandatory Steps

```
1. Detect environment (Python version, GPU, peer modules)
2. Install Python dependencies
3. Deploy files to install directory
4. Create ~/.et_modules/shared_learning/ directory
5. Register in ~/.et_modules/registry.json
6. Set install_path in deployed et_module.json
7. (Optional) Install systemd service
```

### 5.2 Registration Code (copy verbatim into your install.sh)

```bash
# --- ET Module Manager Registration ---
ET_MODULES_DIR="${ET_MODULES_DIR:-$HOME/.et_modules}"
SHARED_LEARNING_DIR="$ET_MODULES_DIR/shared_learning"
mkdir -p "$SHARED_LEARNING_DIR"

python3 -c "
import json, time, os

registry_path = '$ET_MODULES_DIR/registry.json'

# Read existing registry (with locking if available)
try:
    with open(registry_path, 'r') as f:
        registry = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    registry = {'modules': {}}

# Register this module
registry['modules']['YOUR_MODULE_ID'] = {
    'module_id': 'YOUR_MODULE_ID',
    'display_name': 'Your Module Name',
    'version': 'X.Y.Z',
    'description': 'One-line description',
    'install_path': '$INSTALL_DIR',
    'git_remote': 'https://github.com/greatnorthernfishguy-hub/YourRepo.git',
    'git_branch': 'main',
    'entry_point': 'main.py',
    'ng_lite_version': '1.0.0',
    'dependencies': [],
    'service_name': 'your_service_name',
    'api_port': YOUR_PORT,
    'registered_at': time.time(),
}
registry['last_updated'] = time.time()

with open(registry_path, 'w') as f:
    json.dump(registry, f, indent=2)

print('Registered YOUR_MODULE_ID in ET Module Manager')
" 2>/dev/null || echo "[WARN] ET Module Manager registration failed (non-critical)"
```

Replace `YOUR_MODULE_ID`, `Your Module Name`, etc. with actual values.

### 5.3 Peer Module Detection (recommended)

During install, detect sibling modules to inform the user:

```bash
# Check for peer modules
[ -d "$HOME/NeuroGraph" ]               && echo "[INFO] NeuroGraph detected"
[ -d "$HOME/The-Inference-Difference" ] && echo "[INFO] The-Inference-Difference detected"
[ -d "$HOME/TrollGuard" ]              && echo "[INFO] TrollGuard detected"
```

### 5.4 install_path Stamping

After deploying files, update the installed `et_module.json` with the
actual absolute path:

```bash
python3 -c "
import json
with open('$INSTALL_DIR/et_module.json', 'r') as f:
    m = json.load(f)
m['install_path'] = '$INSTALL_DIR'
with open('$INSTALL_DIR/et_module.json', 'w') as f:
    json.dump(m, f, indent=2)
"
```

### 5.5 Uninstaller Rules

- MUST stop and disable systemd service if present.
- MUST remove module files.
- MUST NOT delete `~/.et_modules/shared_learning/`. Learning data survives uninstall.
- SHOULD remove the module's entry from `registry.json`.
- SHOULD preserve the module's `<module_id>.jsonl` event file (other modules may reference it).

---

## 6. Update Mechanism Compliance

### 6.1 How Updates Work

`ETModuleManager.update_all()` does this for each registered module:

```
1. git -C <install_path> fetch --dry-run       → check if updates exist
2. git -C <install_path> pull origin <branch>   → pull updates
3. sudo systemctl restart <service_name>        → restart if service exists
```

### 6.2 What Your Module Must Do to Be Updatable

1. **Be a git repository.** The install directory must be a git clone
   (not a tarball extract or manual copy).
2. **Have `git_remote` set correctly** in `et_module.json`.
3. **Have `git_branch` set correctly** (usually `"main"`).
4. **Not have uncommitted local changes** that would block `git pull`.
   The installer should deploy via `git clone`, not file copy.
5. **Handle service restart gracefully.** If your module has a systemd
   service, it must tolerate `systemctl restart` at any time without
   data corruption. Save state on SIGTERM.

### 6.3 Version Bumping Protocol

When you release an update:

1. Bump `version` in `et_module.json` (semver).
2. Update changelogs in affected files (Section 8).
3. Commit and push to the tracked branch.
4. `ETModuleManager.update_all()` will pick it up on next run.

### 6.4 CLI Entry Point (to be implemented)

The ecosystem needs a `et-modules` CLI command. Until a standalone package
exists, each module that vendors `et_modules/manager.py` can expose it.
The canonical interface:

```
et-modules discover       # List all found modules
et-modules status         # Health check all modules
et-modules update --all   # Pull + restart all modules
et-modules update <id>    # Pull + restart one module
```

**For now:** Any module's `main.py` can add an `et-update` subcommand that
instantiates `ETModuleManager()` and calls `update_all()`. TrollGuard will
add this. Other modules SHOULD too, so whichever module the user happens to
invoke can trigger ecosystem-wide updates.

---

## 7. API Conventions (if module exposes HTTP)

### 7.1 Required Endpoints (if your module has an API)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check. Must return JSON with at least `{"status": "healthy"}` |
| `/stats` | GET | Telemetry. Should include NG-Lite stats if connected. |

### 7.2 Recommended Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/retrain/status` | GET | Active learning readiness (if module has ML) |

### 7.3 Binding Rules

- Default bind: `127.0.0.1` (localhost only). Never default to `0.0.0.0`.
- Port: Use the reserved port from Section 1.3.
- Config: Host and port MUST be overridable via `config.yaml` or env vars.

### 7.4 Systemd Service Template

```ini
[Unit]
Description=<Module Display Name> API
After=network.target

[Service]
Type=simple
User=<deploying_user>
WorkingDirectory=<install_path>
ExecStart=python3 -m uvicorn <entry_module>:app --host <host> --port <port>
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=<install_path> <ET_MODULES_DIR>
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

---

## 8. Changelog Format

Every Python file MUST have an inline changelog block in its module docstring.
This is how we maintain institutional memory across Claude instances working
on different modules at different times.

### 8.1 Format

```python
# ---- Changelog ----
# [YYYY-MM-DD] Author (Model) — Short title.
#   What: What was changed (factual, specific).
#   Why:  Why it was changed (motivation, what problem it solves).
#   Settings: Any defaults chosen and the reasoning behind them.
#   How:  Implementation approach and key design decisions.
#
# [YYYY-MM-DD] Author (Model) — Previous entry.
#   ...
# -------------------
```

### 8.2 Rules

1. Most recent entry first (reverse chronological).
2. `Author` is the entity that made the change. For AI: `"Claude (Opus 4.6)"`.
   For humans: name or handle.
3. `Settings` section is required when defaults are chosen. Explain why
   that value and not another.
4. When you disagree with an external recommendation (e.g., from Grok,
   from a code review), note your reasoning in the changelog. This creates
   a living decision log.
5. The `# ---- Changelog ----` / `# -------------------` delimiters are
   required for grep-ability.

---

## 9. Graceful Degradation Rules

These are non-negotiable. Every integration point must degrade cleanly.

### 9.1 Degradation Hierarchy

```
Full ecosystem (all modules + NeuroGraph SNN)
  ↓ NeuroGraph offline
Peer learning (Tier 2 via shared_learning/)
  ↓ No peer modules installed
Standalone learning (Tier 1 via local ng_lite.py)
  ↓ ng_lite.py fails to load
Module operates without learning (static behavior)
  ↓ Module's own dependencies missing
Clear error message + safe exit
```

### 9.2 Implementation Rules

1. **Never crash on missing peers.** Wrap all bridge/peer operations in
   try/except. Log at WARNING, not ERROR.

2. **Never block on missing peers.** If peer sync fails, continue with
   local state. Never retry synchronously.

3. **Never require the registry to exist.** If `~/.et_modules/registry.json`
   is missing, create it. If it's corrupt, recreate it.

4. **Never require shared_learning/ to exist.** Create it on first write.

5. **Import guards for all cross-module dependencies:**
   ```python
   try:
       from ng_lite import NGLite
       _HAS_NG_LITE = True
   except ImportError:
       _HAS_NG_LITE = False
       # Module continues without learning
   ```

6. **Config missing = use defaults.** Never crash on missing config.yaml.
   Every config value must have a hardcoded default.

7. **Fail open for availability, fail closed for security.** Learning and
   telemetry degrade gracefully (fail open). Security decisions (TrollGuard
   verdicts) must NOT degrade to "allow all" — if the classifier is missing,
   flag as SUSPICIOUS, not SAFE.

---

## 10. GUI Updater Readiness

The NeuroGraph GUI updater does not yet have a communication protocol with
modules. To ensure modules are ready when it does, comply with the following.

### 10.1 Discovery Contract

The GUI updater will find modules by:
1. Reading `~/.et_modules/registry.json`
2. Scanning `KNOWN_LOCATIONS` (Section 11.2) for `et_module.json` files
3. Checking each module's `/health` endpoint (if `api_port > 0`)

**Your module is discoverable if:** it has a valid `et_module.json` AND
is registered in `registry.json` AND `install_path` points to an existing
directory.

### 10.2 Update Trigger Contract

The GUI updater will trigger updates by:
1. Calling `ETModuleManager.update_module(module_id)` via a Python subprocess
2. OR hitting `POST /admin/update` on the module's API (when implemented)
3. OR running `et-modules update <module_id>` CLI command (when implemented)

**Your module supports this if:** it is a git clone with correct `git_remote`
and `git_branch`, and its service tolerates restart.

### 10.3 Status Reporting Contract

The GUI updater will poll status via:
1. `ETModuleManager.status()` — returns `ModuleStatus` with health, version,
   update_available
2. `GET /health` on the module's API

**Your module reports correctly if:** its `et_module.json` version matches
reality, its service responds to `systemctl is-active`, and its `/health`
endpoint returns `{"status": "healthy"}` when operational.

### 10.4 What Each Module Must Implement Now (pre-GUI)

Even before the GUI updater exists, implement these so it works on day one:

1. **Valid `et_module.json`** with all fields populated correctly.
2. **Registration in `~/.et_modules/registry.json`** via `install.sh`.
3. **`/health` endpoint** returning at minimum `{"status": "healthy"}`.
4. **Service that tolerates `systemctl restart`** without data loss.
5. **State saved on SIGTERM** (NG-Lite state, any in-flight data).
6. **No uncommitted changes** in the install directory that block `git pull`.

---

## 11. Known Modules & Reserved Identifiers

### 11.1 Current Modules

| module_id | Display Name | Repo | Port | Status |
|-----------|-------------|------|------|--------|
| `trollguard` | TrollGuard | greatnorthernfishguy-hub/TrollGuard | 7438 | Active |
| `inference_difference` | The-Inference-Difference | greatnorthernfishguy-hub/The-Inference-Difference | 7439 | Active |
| `neurograph` | NeuroGraph | greatnorthernfishguy-hub/NeuroGraph | 7440 | Active |
| `cricket` | Cricket | TBD | 7441 | Planned |
| `faux_clawdbot` | Faux_Clawdbot | TBD | 7442 | Planned |

### 11.2 Known Install Locations

The ET Module Manager scans these paths. If your module installs to a
non-standard location, it MUST register via `registry.json` to be found.

```python
KNOWN_LOCATIONS = [
    "~/NeuroGraph",
    "~/.openclaw/skills/neurograph",
    "~/The-Inference-Difference",
    "/opt/inference-difference",
    "~/TrollGuard",
    "/opt/trollguard",
    "~/.et_modules/modules",
]
```

To add a new module's install location: add it to `KNOWN_LOCATIONS` in
`et_modules/manager.py` in the TrollGuard repo AND in your own vendored
copy. Submit a PR to TrollGuard so the canonical copy stays in sync.

---

## 12. Compliance Checklist

Use this checklist when integrating a new or existing module. Every item
must be checked before the module is considered ecosystem-compatible.

### Files & Structure
- [ ] `et_module.json` exists in project root with all required fields
- [ ] `ng_lite.py` vendored in project root (v1.0.0, unmodified)
- [ ] `ng_peer_bridge.py` vendored in project root (unmodified)
- [ ] `install.sh` exists and is executable

### Installer
- [ ] Creates `~/.et_modules/shared_learning/` directory
- [ ] Registers module in `~/.et_modules/registry.json`
- [ ] Sets `install_path` in deployed `et_module.json`
- [ ] Detects peer modules and reports them
- [ ] Uninstaller preserves `~/.et_modules/shared_learning/`

### NG-Lite Integration
- [ ] `NGLite(module_id=<matching et_module.json module_id>)` used
- [ ] `NGPeerBridge` connected at startup with `shared_dir="~/.et_modules/shared_learning"`
- [ ] NG-Lite state saved on shutdown / SIGTERM
- [ ] NG-Lite state loaded on startup if file exists
- [ ] All bridge operations wrapped in try/except (graceful degradation)
- [ ] Embeddings written to shared_learning/ are 384-dimensional

### File Locking
- [ ] All JSONL writes use `fcntl.LOCK_EX` (with non-POSIX fallback)
- [ ] All JSONL reads use `fcntl.LOCK_SH` (with non-POSIX fallback)
- [ ] Registry JSON read-modify-writes use `fcntl.LOCK_EX`

### Update Readiness
- [ ] Install directory is a git clone (not file copy)
- [ ] `git_remote` in `et_module.json` is correct HTTPS URL
- [ ] `git_branch` is correct (usually `"main"`)
- [ ] Service handles `systemctl restart` without corruption
- [ ] No committed `install_path` in repo copy of `et_module.json` (must be `""`)

### API (if applicable)
- [ ] `GET /health` returns `{"status": "healthy", ...}`
- [ ] Binds to `127.0.0.1` by default (not `0.0.0.0`)
- [ ] Uses reserved port from Section 1.3
- [ ] Port is configurable via config.yaml or env var

### Changelog
- [ ] Every Python file has `# ---- Changelog ----` block
- [ ] Entries include What, Why, Settings (if applicable), How
- [ ] Most recent entry first

### Graceful Degradation
- [ ] Module works without any peer modules installed
- [ ] Module works without `~/.et_modules/` existing
- [ ] Module works without `config.yaml` (uses hardcoded defaults)
- [ ] Module works without NG-Lite (if ng_lite.py fails to import)
- [ ] No operation blocks on missing/unavailable peers

---

## Appendix A: Quick-Start for a New Module

If you are a Claude Code instance starting a brand-new E-T Systems module
from scratch, do the following in order:

1. Create the project directory: `~/YourModuleName/`
2. `git init` and set remote to `https://github.com/greatnorthernfishguy-hub/YourModuleName.git`
3. Copy `ng_lite.py` from any sibling module (they're all identical v1.0.0)
4. Copy `ng_peer_bridge.py` from TrollGuard
5. Create `et_module.json` per Section 1 (pick next available port from Section 1.3)
6. Create `install.sh` per Section 5
7. In your module's initialization code, follow Section 4.2 exactly
8. Add `# ---- Changelog ----` blocks to every Python file
9. Run through the compliance checklist in Section 12
10. Add your module to Section 11.1 of this spec via PR to TrollGuard

## Appendix B: Module-Specific Instructions

### For The-Inference-Difference

Your module already vendors `ng_lite.py` and uses it in the routing engine.
You need to:

1. **Add `et_module.json`** to project root with `module_id: "inference_difference"`,
   `api_port: 7439`, `service_name: "inference-difference"`.
2. **Vendor `ng_peer_bridge.py`** from TrollGuard into your project root.
3. **Connect `NGPeerBridge`** in your app initialization, wherever
   `NGLite(module_id="inference_difference")` is currently created.
   Follow Section 4.2 exactly.
4. **Update your `install.sh`** (or `deploy.sh`) to:
   - Create `~/.et_modules/shared_learning/`
   - Register in `~/.et_modules/registry.json` (use Section 5.2 template)
   - Set `install_path` in deployed `et_module.json`
5. **Add fcntl locking** to any JSONL file operations per Section 4.5.
6. **Add `/health` endpoint** to your API returning
   `{"status": "healthy", "ng_lite_connected": bool, "peer_bridge_connected": bool}`.
7. Your routing engine's `report_outcome()` already calls `ng.record_outcome()` —
   no changes needed there. The peer bridge intercepts transparently.
8. **Verify embedding dimensionality** is 384. If your routing embeddings use
   a different model/dimension, project to 384 before writing to NG-Lite.

### For NeuroGraph

NeuroGraph is the canonical source of `ng_lite.py` and the Tier 3 SNN.
It has a unique role:

1. **Add `et_module.json`** to project root with `module_id: "neurograph"`,
   `api_port: 7440`.
2. **Update `deploy.sh`** to register with ET Module Manager (Section 5.2).
3. **Write events to shared_learning/** even though NeuroGraph is Tier 3.
   Tier 2 modules (TrollGuard, TID) read from `~/.et_modules/shared_learning/`
   and benefit from NeuroGraph's intelligence even without a Tier 3 bridge.
   In `openclaw_hook.py` or wherever outcomes are recorded, also append to
   `~/.et_modules/shared_learning/neurograph.jsonl` using the event schema
   from Section 3.4.
4. **Bridge upgrade offering**: When NeuroGraph detects peer modules via
   `_peer_registry.json`, it can offer to upgrade their connections from
   Tier 2 (peer) to Tier 3 (full SNN). This is optional but valuable —
   it means installing NeuroGraph automatically improves all co-located modules.
5. **ng_lite.py versioning**: When you update `ng_lite.py`, bump its
   `__version__` and update `ng_lite_version` in your `et_module.json`.
   All sibling modules will need to re-vendor. The manager will warn about
   version mismatches.
6. **GUI updater**: If NeuroGraph already has GUI components, the updater
   should use `ETModuleManager.discover()` and `ETModuleManager.update_all()`
   for ecosystem-wide updates. The registry and manifest system is ready.

---

*End of specification.*
