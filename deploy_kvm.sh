#!/usr/bin/env bash
# ---------------------------------------------------------------
# CryptoResearchLab -- Deploy to KVM
# Packages the trained model + execution engine for 24/7 operation
# on a remote KVM server.
# ---------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- Colors --
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
header(){ echo -e "\n${BOLD}${CYAN}$*${NC}"; }

# -- Parse arguments --
KVM_HOST="${1:-}"
KVM_USER="${2:-root}"
KVM_DIR="${3:-/opt/cryptolab}"

if [[ -z "$KVM_HOST" ]]; then
    echo "Usage: ./deploy_kvm.sh <kvm-host> [user] [deploy-dir]"
    echo ""
    echo "Example:"
    echo "  ./deploy_kvm.sh 192.168.1.100 ubuntu /opt/cryptolab"
    echo ""
    echo "Prerequisites on KVM:"
    echo "  - Python 3.11+"
    echo "  - pip"
    echo "  - SSH access"
    echo ""
    echo "This script will:"
    echo "  1. Package the project (models + code)"
    echo "  2. Upload to KVM via rsync/scp"
    echo "  3. Install dependencies on KVM"
    echo "  4. Create a systemd service for 24/7 operation"
    exit 1
fi

header "Deploying CryptoResearchLab to KVM"
echo "  Host: ${KVM_USER}@${KVM_HOST}"
echo "  Dir:  ${KVM_DIR}"

# -- Check prerequisites --
info "Checking trained model..."
if [[ ! -f "${SCRIPT_DIR}/models/xrp_lstm_v1.keras" ]]; then
    fail "No trained model found. Run ./train.sh first."
fi
ok "Model found: models/xrp_lstm_v1.keras"

# -- Create deployment package --
DEPLOY_DIR="${SCRIPT_DIR}/.deploy_package"
rm -rf "$DEPLOY_DIR"
mkdir -p "$DEPLOY_DIR"

info "Packaging project..."

# Copy essential files (no venv, no __pycache__, no checkpoints)
rsync -a --exclude='venv/' \
         --exclude='__pycache__/' \
         --exclude='.deploy_package/' \
         --exclude='checkpoints/' \
         --exclude='*.pyc' \
         --exclude='.DS_Store' \
         --exclude='*.tar.gz' \
         "${SCRIPT_DIR}/" "${DEPLOY_DIR}/"

# Create KVM-specific requirements (no tensorflow-metal on Linux)
cat > "${DEPLOY_DIR}/requirements_kvm.txt" << 'KVMREQ'
# CryptoResearchLab -- KVM Deployment (Linux x86_64)
numpy==2.0.2
pandas==2.2.3
scipy==1.14.1
scikit-learn==1.5.2
joblib==1.4.2
tensorflow==2.18.1
# NO tensorflow-metal on Linux
matplotlib==3.9.3
seaborn==0.13.2
tqdm==4.67.1
ccxt==4.4.54
pyarrow==18.1.0
requests==2.32.3
python-dateutil==2.9.0
pytz==2024.2
KVMREQ

# Create systemd service file
cat > "${DEPLOY_DIR}/cryptolab.service" << SVCEOF
[Unit]
Description=CryptoResearchLab Trading Engine
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=${KVM_USER}
WorkingDirectory=${KVM_DIR}
ExecStart=${KVM_DIR}/venv/bin/python3 ${KVM_DIR}/run.py --skip-research
Restart=on-failure
RestartSec=60
StandardOutput=append:${KVM_DIR}/logs/service.log
StandardError=append:${KVM_DIR}/logs/service_error.log
Environment="PYTHONUNBUFFERED=1"

# Resource limits
LimitNOFILE=65536
MemoryMax=4G
CPUQuota=80%

[Install]
WantedBy=multi-user.target
SVCEOF

# Create remote setup script
cat > "${DEPLOY_DIR}/setup_kvm.sh" << 'SETUPEOF'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[INFO] Setting up CryptoResearchLab on KVM..."

# Create venv
python3.11 -m venv "${SCRIPT_DIR}/venv" 2>/dev/null || python3 -m venv "${SCRIPT_DIR}/venv"
source "${SCRIPT_DIR}/venv/bin/activate"

# Install deps
pip install --upgrade pip setuptools wheel --quiet
pip install -r "${SCRIPT_DIR}/requirements_kvm.txt"

# Create directories
mkdir -p "${SCRIPT_DIR}/data"
mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/reports"

# Verify
python3 -c "
import tensorflow as tf
print(f'TensorFlow: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs: {len(gpus)}')
import ccxt
print(f'CCXT: {ccxt.__version__}')
print('KVM setup OK')
"

echo "[OK] Setup complete!"
echo "To start: sudo systemctl enable cryptolab && sudo systemctl start cryptolab"
SETUPEOF
chmod +x "${DEPLOY_DIR}/setup_kvm.sh"

PACKAGE_SIZE=$(du -sh "$DEPLOY_DIR" | cut -f1)
ok "Package ready: ${PACKAGE_SIZE}"

# -- Upload to KVM --
info "Uploading to ${KVM_USER}@${KVM_HOST}:${KVM_DIR}..."

ssh "${KVM_USER}@${KVM_HOST}" "mkdir -p ${KVM_DIR}" 2>/dev/null || true

rsync -avz --progress \
    "${DEPLOY_DIR}/" \
    "${KVM_USER}@${KVM_HOST}:${KVM_DIR}/"

ok "Upload complete"

# -- Remote setup --
info "Running remote setup..."
ssh "${KVM_USER}@${KVM_HOST}" "cd ${KVM_DIR} && bash setup_kvm.sh"

# -- Install systemd service --
info "Installing systemd service..."
ssh "${KVM_USER}@${KVM_HOST}" "
    sudo cp ${KVM_DIR}/cryptolab.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable cryptolab
    echo 'Service installed. Start with: sudo systemctl start cryptolab'
" 2>/dev/null || warn "Could not install systemd service (may need sudo)"

# Cleanup
rm -rf "$DEPLOY_DIR"

echo ""
header "Deployment Complete!"
echo ""
info "On the KVM server:"
echo "  Start:   sudo systemctl start cryptolab"
echo "  Status:  sudo systemctl status cryptolab"
echo "  Logs:    tail -f ${KVM_DIR}/logs/service.log"
echo "  Stop:    sudo systemctl stop cryptolab"
echo ""
warn "IMPORTANT: Configure API keys on the KVM before starting live trading!"
echo "  Edit ${KVM_DIR}/config.py and set exchange credentials."
echo ""
ok "Done!"
