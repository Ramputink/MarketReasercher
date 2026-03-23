#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
# CryptoResearchLab — Setup Virtual Environment
# Optimizado para Mac M2 Pro + Metal GPU
# ═══════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
REQ_FILE="${SCRIPT_DIR}/requirements_m2.txt"
PYTHON_BIN="python3.11"

# ── Colores ───────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }

# ── 1. Verificar Python 3.11 ─────────────────────────────
info "Verificando Python 3.11..."
if ! command -v "$PYTHON_BIN" &>/dev/null; then
    warn "python3.11 no encontrado en PATH."
    warn "Intentando con 'python3'..."
    PYTHON_BIN="python3"
    if ! command -v "$PYTHON_BIN" &>/dev/null; then
        fail "No se encontró Python 3.11. Instálalo con: brew install python@3.11"
    fi
fi

PY_VER=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [[ "$PY_VER" != "3.11" ]]; then
    warn "Python detectado: $PY_VER (esperado: 3.11)"
    warn "TF 2.18.1 + Metal 1.2.0 está verificado con Python 3.11"
    read -p "¿Continuar de todos modos? (y/N): " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || fail "Abortado. Instala Python 3.11 con: brew install python@3.11"
fi
ok "Python: $PY_VER ($PYTHON_BIN)"

# ── 2. Verificar que estamos en Apple Silicon ─────────────
info "Verificando arquitectura..."
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    warn "Arquitectura detectada: $ARCH (esperada: arm64 para M2)"
    warn "tensorflow-metal solo funciona en Apple Silicon"
fi
ok "Arquitectura: $ARCH"

# ── 3. Verificar Ollama ──────────────────────────────────
info "Verificando Ollama..."
if command -v ollama &>/dev/null; then
    OLLAMA_VER=$(ollama --version 2>/dev/null || echo "desconocida")
    ok "Ollama instalado: $OLLAMA_VER"
else
    warn "Ollama no encontrado. Instálalo desde: https://ollama.ai"
    warn "Después ejecuta: ollama pull llama3.1:8b"
    warn "El sistema funcionará sin LLM pero MiroFish será solo cuantitativo."
fi

# ── 4. Crear virtual environment ─────────────────────────
if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment ya existe en: $VENV_DIR"
    read -p "¿Recrear desde cero? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Eliminando venv existente..."
        rm -rf "$VENV_DIR"
    else
        info "Reutilizando venv existente."
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creando virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    ok "venv creado en: $VENV_DIR"
fi

# ── 5. Activar venv e instalar dependencias ───────────────
info "Activando venv..."
source "${VENV_DIR}/bin/activate"

info "Actualizando pip..."
pip install --upgrade pip setuptools wheel --quiet

info "Instalando dependencias desde requirements_m2.txt..."
info "Esto puede tardar unos minutos (TensorFlow es grande)..."
pip install -r "$REQ_FILE"

# ── 6. Verificar instalación ─────────────────────────────
info "Verificando instalación..."

echo ""
info "─── Verificación de paquetes ───"

python3 -c "
import sys
print(f'  Python:     {sys.version}')

import numpy as np
print(f'  NumPy:      {np.__version__}')

import pandas as pd
print(f'  Pandas:     {pd.__version__}')

import scipy
print(f'  SciPy:      {scipy.__version__}')

import sklearn
print(f'  Sklearn:    {sklearn.__version__}')

import tensorflow as tf
print(f'  TensorFlow: {tf.__version__}')

# Verificar Metal GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'  Metal GPU:  ✓ Detectada(s) — {len(gpus)} dispositivo(s)')
    for gpu in gpus:
        print(f'              {gpu.name}')
else:
    print(f'  Metal GPU:  ✗ No detectada (se usará CPU)')

# Verificar ccxt
import ccxt
print(f'  CCXT:       {ccxt.__version__}')

# Verificar ollama
try:
    import ollama
    print(f'  Ollama SDK: ✓ instalado')
except ImportError:
    print(f'  Ollama SDK: ✗ no disponible')

print()
print('  ═══ Test rápido TensorFlow Metal ═══')
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
    device_used = 'GPU (Metal)' if gpus else 'CPU'
    print(f'  MatMul 1000x1000 en {device_used}: OK (shape={c.shape})')
"

echo ""
ok "═══ Setup completado ═══"
echo ""
info "Para activar el entorno:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
info "Para entrenar el modelo XRP:"
echo "  ./train.sh"
echo ""
info "Para ejecutar el pipeline completo:"
echo "  python run.py"
echo ""

# ── 7. Descargar modelo Ollama si está disponible ────────
if command -v ollama &>/dev/null; then
    info "Verificando modelo Ollama llama3.1:8b..."
    if ollama list 2>/dev/null | grep -q "llama3.1:8b"; then
        ok "Modelo llama3.1:8b ya descargado"
    else
        info "Descargando llama3.1:8b (~4.7 GB)..."
        info "Esto solo se hace una vez."
        ollama pull llama3.1:8b || warn "No se pudo descargar. Hazlo manualmente: ollama pull llama3.1:8b"
    fi
fi

echo ""
ok "¡Todo listo! Tu laboratorio de investigación cuantitativa está configurado."
