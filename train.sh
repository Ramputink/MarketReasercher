#!/usr/bin/env bash
# ---------------------------------------------------------------
# CryptoResearchLab -- Master Training Script
# Runs the full pipeline: Data -> MiroFish -> TF Model -> AutoResearch
# Optimized for Mac M2 Pro + Metal GPU
# ---------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

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
MODE="full"           # full | model-only | pipeline-only | walkforward | optimize | evolve
EXPERIMENTS=12
SKIP_LLM=false
VERBOSE=1
OPT_STRATEGY="all"
EVOLVE_HOURS=14
EVOLVE_POP=40
EVOLVE_CORES=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-only)      MODE="model-only"; shift ;;
        --pipeline-only)   MODE="pipeline-only"; shift ;;
        --walkforward)     MODE="walkforward"; shift ;;
        --optimize)        MODE="optimize"; shift ;;
        --evolve)          MODE="evolve"; shift ;;
        --evolve-hours)    EVOLVE_HOURS="$2"; shift 2 ;;
        --evolve-pop)      EVOLVE_POP="$2"; shift 2 ;;
        --cores)           EVOLVE_CORES="$2"; shift 2 ;;
        --opt-strategy)    OPT_STRATEGY="$2"; shift 2 ;;
        --experiments)     EXPERIMENTS="$2"; shift 2 ;;
        --no-llm)          SKIP_LLM=true; shift ;;
        --quiet)           VERBOSE=0; shift ;;
        --help|-h)
            echo "Usage: ./train.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-only       Only train TF model (skip pipeline)"
            echo "  --pipeline-only    Only run research pipeline (skip TF model)"
            echo "  --walkforward      Run walk-forward model training"
            echo "  --optimize         Run parallel strategy parameter optimizer"
            echo "  --evolve           Run autonomous evolution (default: 14h)"
            echo "  --evolve-hours N   Hours for evolution (default: 14)"
            echo "  --evolve-pop N     Population size (default: 40)"
            echo "  --cores N          CPU cores for evolution (0=auto, default: 0)"
            echo "  --opt-strategy S   Strategy to optimize: vb, mr, tf, all (default: all)"
            echo "  --experiments N    Number of autoresearch experiments (default: 12)"
            echo "  --no-llm           Skip Ollama LLM augmentation"
            echo "  --quiet            Minimal output"
            echo "  --help             Show this help"
            exit 0
            ;;
        *) warn "Unknown option: $1"; shift ;;
    esac
done

# -- Check venv --
if [[ ! -d "$VENV_DIR" ]]; then
    fail "Virtual environment not found. Run ./setup_venv.sh first."
fi

# Activate venv
source "${VENV_DIR}/bin/activate"

header "================================================="
header " CryptoResearchLab -- Training Pipeline"
header " Mode: ${MODE}"
header " Experiments: ${EXPERIMENTS}"
header "================================================="
echo ""

# -- Check Metal GPU + set threading --
info "Checking Metal GPU and configuring parallelism..."
NCORES=$(python3 -c "import os; print(os.cpu_count())" 2>/dev/null || echo "4")
info "CPU cores detected: ${NCORES}"

python3 -c "
import os, tensorflow as tf
# Configure TF to use all available cores for CPU ops
ncores = os.cpu_count() or 4
tf.config.threading.set_intra_op_parallelism_threads(ncores)
tf.config.threading.set_inter_op_parallelism_threads(ncores)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f'  Metal GPU: {len(gpus)} device(s) detected')
    print(f'  CPU threads: intra={ncores}, inter={ncores}')
else:
    print(f'  No GPU detected -- using CPU with {ncores} threads')
" 2>/dev/null || warn "Could not check GPU status"

# Export for child processes
export TF_NUM_INTRAOP_THREADS="${NCORES}"
export TF_NUM_INTEROP_THREADS="${NCORES}"
export OMP_NUM_THREADS="${NCORES}"
export MKL_NUM_THREADS="${NCORES}"

# -- Check Ollama --
if [[ "$SKIP_LLM" == "false" ]]; then
    info "Checking Ollama..."
    if command -v ollama &>/dev/null; then
        if ollama list 2>/dev/null | grep -q "llama3.1"; then
            ok "Ollama + llama3.1:8b ready"
            LLM_FLAG=""
        else
            warn "llama3.1:8b not found. Run: ollama pull llama3.1:8b"
            LLM_FLAG="--no-llm"
        fi
    else
        warn "Ollama not running. LLM features disabled."
        LLM_FLAG="--no-llm"
    fi
else
    LLM_FLAG="--no-llm"
    info "LLM augmentation disabled (--no-llm)"
fi

# -- Create output directories --
mkdir -p "${SCRIPT_DIR}/data"
mkdir -p "${SCRIPT_DIR}/models"
mkdir -p "${SCRIPT_DIR}/checkpoints"
mkdir -p "${SCRIPT_DIR}/logs"
mkdir -p "${SCRIPT_DIR}/reports"

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------
# STAGE 1: Full research pipeline (strategies + backtests)
# ---------------------------------------------------------------
if [[ "$MODE" == "full" || "$MODE" == "pipeline-only" ]]; then
    header "[STAGE 1/3] Research Pipeline (MiroFish + AutoResearch + Backtests)"

    PIPELINE_ARGS="--experiments ${EXPERIMENTS}"
    if [[ "${SKIP_LLM}" == "true" || "${LLM_FLAG:-}" == "--no-llm" ]]; then
        PIPELINE_ARGS="${PIPELINE_ARGS} --skip-research"
    fi

    python3 run.py ${PIPELINE_ARGS} 2>&1 | tee "logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
    ok "Research pipeline complete"
fi

# ---------------------------------------------------------------
# STAGE 2: TensorFlow model training
# ---------------------------------------------------------------
if [[ "$MODE" == "full" || "$MODE" == "model-only" ]]; then
    header "[STAGE 2/3] TensorFlow Model Training (XRP/USDT)"

    python3 -c "
import sys, os, logging
sys.path.insert(0, '.')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('train')

from config import LabConfig, TimeFrame, DataConfig
from engine.data_ingestion import DataIngestionEngine
from engine.env_loader import has_binance_keys
from engine.features import build_all_features
from engine.tf_model import train_model, setup_metal_gpu

config = LabConfig()

# ---- DATA: real Binance data only ----
if not has_binance_keys():
    logger.error('ERROR: No Binance API keys found.')
    logger.error('Configure .env with BINANCE_API_KEY and BINANCE_API_SECRET')
    sys.exit(1)

logger.info('Fetching real XRP/USDT data from Binance (180 days, H1)...')
data_cfg = DataConfig(
    timeframes=[TimeFrame.H1],
    history_days=180,
    cross_exchanges=[],
)
engine = DataIngestionEngine(data_cfg)
raw = engine.fetch_ohlcv(
    'XRP/USDT', TimeFrame.H1,
    since_days=180, force_refresh=True,
)

if raw is None or len(raw) < 100:
    logger.error(f'ERROR: Insufficient data from Binance ({len(raw) if raw is not None else 0} candles).')
    logger.error('Check your API keys and network connection.')
    sys.exit(1)

logger.info(f'Real data: {len(raw)} candles')

# Build features
logger.info('Computing features...')
df = build_all_features(raw)
logger.info(f'Features: {len(df.columns)} columns, {len(df)} rows')

# Train model
logger.info('Starting TF model training on Metal GPU...')
model, history, test_metrics, scaler = train_model(
    df,
    config=config.tf_model,
    model_name='xrp_lstm_v1',
    verbose=${VERBOSE},
)

# Summary
logger.info('')
logger.info('=== Training Summary ===')
for k, v in test_metrics.items():
    if isinstance(v, float):
        logger.info(f'  {k}: {v:.4f}')
    elif isinstance(v, (int, str)):
        logger.info(f'  {k}: {v}')
logger.info('========================')
logger.info(f'Model saved to: models/xrp_lstm_v1.keras')
" 2>&1 | tee "logs/tf_training_$(date +%Y%m%d_%H%M%S).log"

    ok "TF model training complete"
fi

# ---------------------------------------------------------------
# STAGE 3: Walk-forward validation (optional)
# ---------------------------------------------------------------
if [[ "$MODE" == "walkforward" ]]; then
    header "[STAGE 3/3] Walk-Forward Model Training"

    python3 -c "
import sys, os, logging
sys.path.insert(0, '.')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('walkforward')

from config import LabConfig, TimeFrame, DataConfig
from engine.data_ingestion import DataIngestionEngine
from engine.env_loader import has_binance_keys
from engine.features import build_all_features
from engine.tf_model import walk_forward_train

config = LabConfig()

# ---- DATA: real Binance data only ----
if not has_binance_keys():
    logger.error('ERROR: No Binance API keys found.')
    sys.exit(1)

logger.info('Fetching real XRP/USDT data for walk-forward (365 days, H1)...')
data_cfg = DataConfig(
    timeframes=[TimeFrame.H1],
    history_days=365,
    cross_exchanges=[],
)
engine = DataIngestionEngine(data_cfg)
raw = engine.fetch_ohlcv(
    'XRP/USDT', TimeFrame.H1,
    since_days=365, force_refresh=True,
)

if raw is None or len(raw) < 500:
    logger.error(f'ERROR: Insufficient data ({len(raw) if raw is not None else 0} candles, need 500+).')
    sys.exit(1)

logger.info(f'Real data: {len(raw)} candles')
df = build_all_features(raw)

logger.info('Starting walk-forward training...')
fold_results, aggregate = walk_forward_train(
    df, config=config.tf_model,
    train_days=90, val_days=20, test_days=30,
    model_name_prefix='xrp_wf',
)

logger.info('')
logger.info('=== Walk-Forward Summary ===')
for k, v in aggregate.items():
    logger.info(f'  {k}: {v}')
logger.info('============================')
" 2>&1 | tee "logs/walkforward_$(date +%Y%m%d_%H%M%S).log"

    ok "Walk-forward training complete"
fi

# ---------------------------------------------------------------
# STAGE 4: Strategy Parameter Optimizer (parallel)
# ---------------------------------------------------------------
if [[ "$MODE" == "optimize" ]]; then
    header "[STAGE 4] Strategy Parameter Optimizer (multicore)"

    python3 optimize_strategies.py \
        --strategy "${OPT_STRATEGY}" \
        --cores "${NCORES}" \
        --days 365 \
        2>&1 | tee "logs/optimize_$(date +%Y%m%d_%H%M%S).log"

    ok "Optimization complete"
fi

# ---------------------------------------------------------------
# STAGE 5: Autonomous Evolution (genetic algorithm)
# ---------------------------------------------------------------
if [[ "$MODE" == "evolve" ]]; then
    # Use user-specified cores or auto-detect
    if [[ "${EVOLVE_CORES}" -gt 0 ]]; then
        ECORES="${EVOLVE_CORES}"
    else
        ECORES="${NCORES}"
    fi

    header "[STAGE 5] Autonomous Evolution Engine (${EVOLVE_HOURS}h)"
    info "Population: ${EVOLVE_POP}, Cores: ${ECORES}/${NCORES}, Duration: ${EVOLVE_HOURS} hours"
    info "Strategies: 11 (TF, Donchian, DualMA, Keltner, VolSqueeze, Ichimoku, KAMA, Fisher, ChaosTrend, VolRegimeArb, LSTMPattern)"
    info "This will run autonomously. Check logs/ for progress."
    echo ""

    python3 auto_evolve.py \
        --hours "${EVOLVE_HOURS}" \
        --pop-size "${EVOLVE_POP}" \
        --cores "${ECORES}" \
        --days 365 \
        2>&1 | tee "logs/evolve_$(date +%Y%m%d_%H%M%S).log"

    ok "Evolution complete! Check reports/evolution_report_*.json for results"
fi

# ---------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------
echo ""
header "================================================="
header " Training Complete!"
header "================================================="
echo ""
info "Artifacts:"
echo "  Models:      ${SCRIPT_DIR}/models/"
echo "  Checkpoints: ${SCRIPT_DIR}/checkpoints/"
echo "  Reports:     ${SCRIPT_DIR}/reports/"
echo "  Logs:        ${SCRIPT_DIR}/logs/"
echo ""
info "Next steps:"
echo "  1. Review reports in reports/"
echo "  2. Optimize strategies: ./train.sh --optimize"
echo "  3. Optimize single:     ./train.sh --optimize --opt-strategy vb"
echo "  4. Run walk-forward:    ./train.sh --walkforward"
echo "  5. EVOLVE (14h auto):   ./train.sh --evolve"
echo "  6. Evolve (custom):     ./train.sh --evolve --evolve-hours 8 --evolve-pop 60"
echo "  7. Deploy to KVM:       ./deploy_kvm.sh"
echo ""
ok "Done!"
