#!/usr/bin/env bash
# ============================================================================
# TrollGuard — One-Click Installer
#
# Installs TrollGuard as an E-T Systems module with:
#   - Python dependency installation
#   - NG-Lite peer bridge setup (shared learning directory)
#   - ET Module Manager registration
#   - Optional systemd service (for runtime sentry API)
#
# Usage:
#   ./install.sh              # Full installation
#   ./install.sh --deps-only  # Install dependencies only
#   ./install.sh --no-service # Install without systemd service
#   ./install.sh --uninstall  # Remove TrollGuard (preserves learning data)
#   ./install.sh --status     # Check installation status
#
# Environment variable overrides:
#   TG_INSTALL_DIR   — Installation path (default: ~/TrollGuard)
#   TG_PORT          — API port (default: 7438)
#   TG_HOST          — Bind host (default: 127.0.0.1)
#
# Follows the same patterns as The-Inference-Difference's install.sh
# for consistency across the E-T Systems module ecosystem.
#
# Changelog:
# [2026-02-17] Claude (Opus 4.6) — Initial creation.
#   Modeled after The-Inference-Difference's install.sh.
#   Added ET Module Manager registration and NGPeerBridge shared
#   learning directory setup — the cross-module coordination that
#   TID's installer doesn't yet have.
# ============================================================================

set -euo pipefail

# --- Configuration (overridable via environment) ---
INSTALL_DIR="${TG_INSTALL_DIR:-$HOME/TrollGuard}"
API_PORT="${TG_PORT:-7438}"
API_HOST="${TG_HOST:-127.0.0.1}"
SERVICE_NAME="trollguard"
ET_MODULES_DIR="${ET_MODULES_DIR:-$HOME/.et_modules}"
SHARED_LEARNING_DIR="$ET_MODULES_DIR/shared_learning"

# --- Colors ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# --- Detect environment ---
detect_environment() {
    info "Detecting environment..."

    # Python version
    if command -v python3 &>/dev/null; then
        PYTHON="python3"
        PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
        info "Python: $PY_VERSION"
    else
        error "Python 3 not found. TrollGuard requires Python 3.10+"
        exit 1
    fi

    # Check Python 3.10+
    PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
    if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
        error "Python 3.10+ required (found $PY_VERSION)"
        exit 1
    fi

    # GPU detection
    if command -v nvidia-smi &>/dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "")
        if [ -n "$GPU_INFO" ]; then
            info "GPU detected: $GPU_INFO"
            HAS_GPU=true
        else
            HAS_GPU=false
        fi
    else
        HAS_GPU=false
        info "No NVIDIA GPU detected (CPU-only mode)"
    fi

    # Check for peer modules via registry (sole source of truth — no hardcoded paths)
    HAS_NEUROGRAPH=false
    HAS_TID=false
    if [ -f "$ET_MODULES_DIR/registry.json" ]; then
        PEER_INFO=$($PYTHON -c "
import json
with open('$ET_MODULES_DIR/registry.json') as f:
    reg = json.load(f)
mods = reg.get('modules', {})
if 'neurograph' in mods:
    print('neurograph:' + mods['neurograph'].get('install_path', ''))
if 'inference_difference' in mods:
    print('inference_difference:' + mods['inference_difference'].get('install_path', ''))
" 2>/dev/null || true)
        if echo "$PEER_INFO" | grep -q "^neurograph:"; then
            NG_PATH=$(echo "$PEER_INFO" | grep "^neurograph:" | cut -d: -f2)
            if [ -n "$NG_PATH" ] && [ -d "$NG_PATH" ]; then
                info "NeuroGraph detected at $NG_PATH (via registry)"
                HAS_NEUROGRAPH=true
            fi
        fi
        if echo "$PEER_INFO" | grep -q "^inference_difference:"; then
            TID_PATH=$(echo "$PEER_INFO" | grep "^inference_difference:" | cut -d: -f2)
            if [ -n "$TID_PATH" ] && [ -d "$TID_PATH" ]; then
                info "The-Inference-Difference detected at $TID_PATH (via registry)"
                HAS_TID=true
            fi
        fi
    fi
}

# --- Install dependencies ---
install_deps() {
    info "Installing Python dependencies..."

    $PYTHON -m pip install --upgrade pip 2>/dev/null || true

    # Core dependencies (always required)
    $PYTHON -m pip install numpy pyyaml rich python-dotenv 2>/dev/null

    # ML dependencies
    $PYTHON -m pip install scikit-learn 2>/dev/null

    # Sentence transformers (largest dependency — CPU build)
    info "Installing sentence-transformers (this may take a minute)..."
    if [ "$HAS_GPU" = true ]; then
        $PYTHON -m pip install sentence-transformers 2>/dev/null
    else
        $PYTHON -m pip install sentence-transformers 2>/dev/null
    fi

    # API server
    $PYTHON -m pip install fastapi uvicorn 2>/dev/null

    # HuggingFace client (for Swarm Audit LLM calls)
    $PYTHON -m pip install huggingface_hub 2>/dev/null

    info "Dependencies installed."
}

# --- Deploy files ---
deploy_files() {
    info "Deploying TrollGuard to $INSTALL_DIR..."

    # Resolve where install.sh lives (the git clone)
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    # If the clone IS the install dir, skip file copies — they're already there.
    # Just ensure data directories and manifest are set up.
    if [ "$(realpath "$SCRIPT_DIR")" = "$(realpath "$INSTALL_DIR")" ]; then
        info "Clone directory is the install directory — skipping file copy."
    else
        # Create install directory (use sudo only if outside $HOME)
        if [[ "$INSTALL_DIR" == "$HOME"* ]]; then
            mkdir -p "$INSTALL_DIR"
        else
            sudo mkdir -p "$INSTALL_DIR"
            sudo chown "$(whoami):$(whoami)" "$INSTALL_DIR"
        fi

        # Copy project files
        cp "$SCRIPT_DIR/main.py" "$INSTALL_DIR/"
        cp "$SCRIPT_DIR/api.py" "$INSTALL_DIR/"
        cp "$SCRIPT_DIR/trollguard_hook.py" "$INSTALL_DIR/"
        cp "$SCRIPT_DIR/config.yaml" "$INSTALL_DIR/"
        cp "$SCRIPT_DIR/ng_lite.py" "$INSTALL_DIR/"
        cp "$SCRIPT_DIR/ng_peer_bridge.py" "$INSTALL_DIR/"
        cp "$SCRIPT_DIR/cisco_wrapper_mock.py" "$INSTALL_DIR/"
        cp "$SCRIPT_DIR/et_module.json" "$INSTALL_DIR/"
        cp "$SCRIPT_DIR/requirements.txt" "$INSTALL_DIR/"

        # Copy sentinel_core package
        mkdir -p "$INSTALL_DIR/sentinel_core"
        cp "$SCRIPT_DIR/sentinel_core/"*.py "$INSTALL_DIR/sentinel_core/"

        # Copy et_modules package
        mkdir -p "$INSTALL_DIR/et_modules"
        cp "$SCRIPT_DIR/et_modules/"*.py "$INSTALL_DIR/et_modules/"
    fi

    # Create data directories
    mkdir -p "$INSTALL_DIR/models"
    mkdir -p "$INSTALL_DIR/training_data"
    mkdir -p "$INSTALL_DIR/cisco_base"

    # Update manifest with actual install path
    $PYTHON -c "
import json
with open('$INSTALL_DIR/et_module.json', 'r') as f:
    m = json.load(f)
m['install_path'] = '$INSTALL_DIR'
with open('$INSTALL_DIR/et_module.json', 'w') as f:
    json.dump(m, f, indent=2)
"

    info "Files deployed to $INSTALL_DIR"
}

# --- Setup shared learning (NGPeerBridge) ---
setup_shared_learning() {
    info "Setting up shared learning directory..."

    mkdir -p "$SHARED_LEARNING_DIR"
    mkdir -p "$ET_MODULES_DIR"

    info "Shared learning directory: $SHARED_LEARNING_DIR"

    # Register with ET Module Manager
    info "Registering with ET Module Manager..."

    $PYTHON -c "
import json, os, time
registry_path = '$ET_MODULES_DIR/registry.json'

try:
    with open(registry_path, 'r') as f:
        registry = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    registry = {'modules': {}}

registry['modules']['trollguard'] = {
    'module_id': 'trollguard',
    'display_name': 'TrollGuard',
    'version': '0.1.0',
    'description': 'The Open-Source Immune System for AI Agents',
    'install_path': '$INSTALL_DIR',
    'git_remote': 'https://github.com/greatnorthernfishguy-hub/TrollGuard.git',
    'git_branch': 'main',
    'entry_point': 'main.py',
    'ng_lite_version': '1.0.0',
    'dependencies': [],
    'service_name': 'trollguard',
    'api_port': $API_PORT,
    'registered_at': time.time(),
}
registry['last_updated'] = time.time()

with open(registry_path, 'w') as f:
    json.dump(registry, f, indent=2)

print('Registered TrollGuard in ET Module Manager')
" 2>/dev/null || warn "ET Module Manager registration failed (non-critical)"
}

# --- Install systemd service ---
install_service() {
    info "Installing systemd service..."

    sudo tee /etc/systemd/system/${SERVICE_NAME}.service > /dev/null << SERVICEEOF
[Unit]
Description=TrollGuard Runtime Sentry API
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$INSTALL_DIR
ExecStart=$PYTHON -m uvicorn api:app --host $API_HOST --port $API_PORT
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=$INSTALL_DIR $ET_MODULES_DIR
PrivateTmp=true

[Install]
WantedBy=multi-user.target
SERVICEEOF

    sudo systemctl daemon-reload
    sudo systemctl enable ${SERVICE_NAME}
    sudo systemctl start ${SERVICE_NAME}

    info "Service installed and started: $SERVICE_NAME"
}

# --- Status check ---
check_status() {
    info "TrollGuard Status"
    echo "========================"

    if [ -d "$INSTALL_DIR" ]; then
        echo -e "Installed: ${GREEN}Yes${NC} ($INSTALL_DIR)"
    else
        echo -e "Installed: ${RED}No${NC}"
    fi

    if systemctl is-active --quiet ${SERVICE_NAME} 2>/dev/null; then
        echo -e "Service:   ${GREEN}Running${NC}"
    else
        echo -e "Service:   ${YELLOW}Stopped${NC}"
    fi

    if [ -d "$SHARED_LEARNING_DIR" ]; then
        PEER_FILES=$(ls "$SHARED_LEARNING_DIR"/*.jsonl 2>/dev/null | wc -l)
        echo -e "Peer Bridge: ${GREEN}Active${NC} ($PEER_FILES module event files)"
    else
        echo -e "Peer Bridge: ${YELLOW}Not configured${NC}"
    fi

    if [ -f "$ET_MODULES_DIR/registry.json" ]; then
        MODULE_COUNT=$($PYTHON -c "
import json
with open('$ET_MODULES_DIR/registry.json') as f:
    r = json.load(f)
print(len(r.get('modules', {})))
" 2>/dev/null || echo "?")
        echo -e "ET Modules:  ${GREEN}$MODULE_COUNT registered${NC}"
    fi
}

# --- Uninstall ---
uninstall() {
    warn "Uninstalling TrollGuard..."

    # Stop service
    if systemctl is-active --quiet ${SERVICE_NAME} 2>/dev/null; then
        sudo systemctl stop ${SERVICE_NAME}
        sudo systemctl disable ${SERVICE_NAME}
        sudo rm -f /etc/systemd/system/${SERVICE_NAME}.service
        sudo systemctl daemon-reload
        info "Service removed"
    fi

    # Remove files (preserve learning data)
    if [ -d "$INSTALL_DIR" ]; then
        warn "Preserving learning data in $SHARED_LEARNING_DIR"
        sudo rm -rf "$INSTALL_DIR"
        info "Removed $INSTALL_DIR"
    fi

    info "TrollGuard uninstalled. Learning data preserved."
}

# --- Main ---
main() {
    echo "============================================"
    echo "  TrollGuard Installer v0.1.0"
    echo "  The Open-Source Immune System for AI Agents"
    echo "============================================"
    echo ""

    case "${1:-}" in
        --deps-only)
            detect_environment
            install_deps
            ;;
        --no-service)
            detect_environment
            install_deps
            deploy_files
            setup_shared_learning
            info "Installation complete (no service). Run manually:"
            info "  cd $INSTALL_DIR && python3 main.py scan /path/to/file.py"
            ;;
        --uninstall)
            uninstall
            ;;
        --status)
            check_status
            ;;
        *)
            detect_environment
            install_deps
            deploy_files
            setup_shared_learning
            install_service
            info ""
            info "============================================"
            info "  TrollGuard installed successfully!"
            info "============================================"
            info ""
            info "  Install dir:  $INSTALL_DIR"
            info "  API endpoint: http://$API_HOST:$API_PORT"
            info "  Service:      systemctl status $SERVICE_NAME"
            info ""
            info "  Quick start:"
            info "    python3 $INSTALL_DIR/main.py scan /path/to/skill.py"
            info ""
            info "  Peer modules detected:"
            [ "$HAS_NEUROGRAPH" = true ] && info "    - NeuroGraph"
            [ "$HAS_TID" = true ]        && info "    - The-Inference-Difference"
            info ""
            ;;
    esac
}

main "$@"
