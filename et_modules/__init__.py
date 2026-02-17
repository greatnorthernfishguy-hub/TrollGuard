"""
ET Module Manager — Unified Update & Coordination for E-T Systems Modules

Provides a single command to install, update, and manage all E-T Systems
modules (NeuroGraph, The-Inference-Difference, TrollGuard, Cricket, etc.)
as a cohesive ecosystem.

Instead of each module having its own independent install/update script,
the ET Module Manager knows about all registered modules and can:
  - Update all modules at once (`et-modules update --all`)
  - Check status of all modules (`et-modules status`)
  - Manage the shared learning directory for NGPeerBridge
  - Coordinate version compatibility between modules

The manager uses a manifest system: each module declares its identity,
version, dependencies, and update source in an `et_module.json` manifest
file.  The manager discovers modules by scanning known install locations.

# ---- Changelog ----
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   What: ET Module Manager package with ModuleManifest, ModuleRegistry,
#         and ETModuleManager classes.
#   Why:  The user explicitly requested that modules "act as one when
#         used together, instead of having to update each module
#         separately."  With 6+ planned modules, independent update
#         scripts are impractical.  This manager is the "convenience
#         thing" they asked for.
#   Settings: Module root defaults to ~/.et_modules/ — a shared
#         location that all E-T Systems modules agree on.  Each module
#         registers itself here on install.  shared_learning/ subdirectory
#         is used by NGPeerBridge for cross-module learning.
#   How:  Manifest-based discovery.  Each module drops an et_module.json
#         in its install directory.  The manager scans known locations
#         (and the registry) to find all modules.  Updates pull from
#         each module's declared git remote.
# -------------------
"""

__version__ = "0.1.0"
