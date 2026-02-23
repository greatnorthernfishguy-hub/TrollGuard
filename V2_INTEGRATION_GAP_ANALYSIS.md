# TrollGuard v2 Integration Spec — Gap Analysis

**Date:** 2026-02-23
**Baseline:** Integration-Specs/ folder (v2.0, dated 2026-02-22)
**Analyzed by:** Claude (Opus 4.6)

---

## Summary

TrollGuard was the v1 reference implementation and is now behind the v2 unified
spec.  There are **4 critical gaps**, **2 important gaps**, and **1 minor gap**.

The security pipeline itself (sentinel_core/, 4-layer defense, semantic air gap)
is unaffected — all gaps are in the ecosystem integration layer.

---

## CRITICAL GAPS (must fix for v2 compliance)

### 1. `ng_ecosystem.py` — MISSING

The v2 spec's centerpiece file.  Replaces manual "copy the TrollGuard init
pattern" with a single vendorable orchestrator handling Tier 1→2→3 lifecycle
automatically.  The spec says "Always required" for every module.

- **Spec source:** `Integration-Specs/ng ecosystem.pdf` (~14 KB implementation)
- **Current state:** File does not exist in TrollGuard
- **Impact:** No auto-upgrade to Tier 3, no standardized ecosystem API
  (`record_outcome`, `get_recommendations`, `detect_novelty`, `get_context`),
  other v2-compliant modules can't interop cleanly with TrollGuard's learning

### 2. `openclaw_adapter.py` — MISSING

Standardized OpenClaw skill base class.  Since TrollGuard IS an OpenClaw skill,
this is required.

- **Spec source:** `Integration-Specs/openclaw adapter.pdf` (~12 KB implementation)
- **Current state:** File does not exist in TrollGuard
- **Impact:** `trollguard_hook.py` reimplements what this base class provides
  (singleton, event logging, workspace management, embedding fallback)

### 3. `et_module.json` — v1 schema, needs v2

**Current (v1 — flat keys):**
```json
{
  "module_id": "trollguard",
  "ng_lite_version": "1.0.0",
  "dependencies": [],
  "service_name": "trollguard",
  "api_port": 7438
}
```

**Required (v2 — `_schema` + `ecosystem` block):**
```json
{
  "_schema": "et_module/2.0",
  "module_id": "trollguard",
  "ecosystem": {
    "ng_lite": true,
    "ng_lite_version": "1.0.0",
    "peer_bridge": true,
    "tier3_upgrade": true,
    "ng_ecosystem_version": "1.0.0",
    "openclaw_adapter": "trollguard_hook.py",
    "openclaw_skill_name": "TrollGuard Security",
    "shared_learning_writes": true,
    "capabilities": ["security", "threat-detection", "prompt-injection-defense"]
  }
}
```

**Missing fields:** `_schema`, entire `ecosystem` block, `capabilities`,
`provides`, `openclaw_adapter`, `openclaw_skill_name`, `tier3_upgrade`,
`ng_ecosystem_version`, `shared_learning_writes`

### 4. `trollguard_hook.py` — needs refactor to subclass `OpenClawAdapter`

**Current:** 523-line bespoke `TrollGuardFilter` class with:
- Manual NGLite + NGPeerBridge init (`_init_components()`)
- Custom event logging (`_write_event()` → `security/events.jsonl`)
- Custom singleton pattern
- Custom workspace/config management
- Mixes security logic with ecosystem wiring

**Required per spec** (`Integration-Specs/trollguard hook example.pdf`):
Subclass `OpenClawAdapter` with only TrollGuard-specific logic:
- `MODULE_ID = "trollguard"`
- `SKILL_NAME = "TrollGuard Security"`
- `_embed(text)` — embedder with hash fallback
- `_module_on_message(text, embedding)` — security scan
- `_module_stats()` — scan/threat counters

All ecosystem wiring delegated to `OpenClawAdapter` → `ng_ecosystem.py`.

**Note:** Existing `sanitize()`, `scan_file()`, `scan_url()`, `scan_repo()`
public API is TrollGuard-specific and not part of OpenClawAdapter.  These
need to be preserved alongside the adapter pattern.

---

## IMPORTANT GAPS (should fix)

### 5. `install.sh` — deploys v1 files, registers v1 manifest

- **`deploy_files()`** (line 143–191): Does not copy `ng_ecosystem.py` or
  `openclaw_adapter.py`
- **`setup_shared_learning()`** (line 194–237): Registers flat v1 manifest
  in `registry.json` instead of v2 schema with `ecosystem` block
- **Fix:** Add new files to copy list; update registry snippet to v2

### 6. `ET_MODULE_INTEGRATION_PRIMER.md` — documents v1 pattern

- **Current:** Manual `NGLite` + `NGPeerBridge` init, v1 manifest schema,
  no mention of `ng_ecosystem.py` or `openclaw_adapter.py`
- **Spec v2:** `Integration-Specs/ET MODULE INTEGRATION PRIMER v2.pdf` —
  completely rewritten with `ng_ecosystem.init()` as standard
- **Fix:** Replace with v2 content or add note pointing to Integration-Specs

---

## MINOR GAP

### 7. `main.py` — manual tier init should migrate to `ng_ecosystem.init()`

- **Current:** `_init_ng_lite()` manually creates `NGLite()` + `NGPeerBridge()`
- **v2 pattern:** `eco = ng_ecosystem.init(module_id="trollguard", ...)`
- **Impact:** Lower priority — CLI/pipeline entry, not the OpenClaw skill hook

---

## WHAT'S ALREADY COMPLIANT

| Component | Status |
|---|---|
| `ng_lite.py` (v1.0.0 vendored) | ✓ Compliant |
| `ng_peer_bridge.py` (vendored) | ✓ Compliant |
| `~/.et_modules/` directory structure | ✓ Compliant |
| `~/.et_modules/shared_learning/` events | ✓ Compliant |
| `~/.et_modules/registry.json` registration | ✓ Compliant (schema needs v2) |
| Graceful degradation principle | ✓ Compliant |
| Living changelog convention | ✓ Compliant |
| Thread safety (RLock, fcntl) | ✓ Compliant |
| 4-layer security pipeline | ✓ Unaffected |
| `sentinel_core/` modules | ✓ Unaffected |
| `et_modules/manager.py` | ✓ Compliant |

---

## Recommended Fix Order

1. Vendor `ng_ecosystem.py` from spec / NeuroGraph canonical source
2. Vendor `openclaw_adapter.py` from spec / NeuroGraph canonical source
3. Upgrade `et_module.json` to v2 schema
4. Refactor `trollguard_hook.py` to subclass `OpenClawAdapter`
   (preserving TrollGuard-specific public API)
5. Update `install.sh` to deploy new files + register v2 manifest
6. Update `ET_MODULE_INTEGRATION_PRIMER.md` to v2
7. Migrate `main.py` init to `ng_ecosystem.init()`
