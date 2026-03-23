"""
CryptoResearchLab -- Environment & Credentials Loader
Loads API keys and config from .env file.
NEVER import secrets directly in code -- always use this module.
"""
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Find .env relative to this file's location
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_FILE = _PROJECT_ROOT / ".env"


def _parse_env_file(path: Path) -> dict:
    """Parse a .env file into a dictionary."""
    env_vars = {}
    if not path.exists():
        return env_vars

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Remove surrounding quotes if present
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                env_vars[key] = value
    return env_vars


def load_env():
    """
    Load .env file into os.environ.
    Does NOT override existing environment variables.
    """
    env_vars = _parse_env_file(_ENV_FILE)
    loaded = 0
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
            loaded += 1

    if loaded > 0:
        logger.info(f"Loaded {loaded} variables from .env")
    elif not _ENV_FILE.exists():
        logger.warning(
            f".env file not found at {_ENV_FILE}. "
            "Copy .env.example to .env and fill in your API keys."
        )

    return env_vars


def get_binance_credentials() -> dict:
    """
    Get Binance API credentials.
    Returns dict with 'apiKey' and 'secret', or empty values if not set.
    """
    load_env()
    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    if not api_key or api_key == "your_api_key_here":
        logger.warning(
            "BINANCE_API_KEY not configured. "
            "Using public endpoints only (no authenticated data)."
        )
        return {"apiKey": "", "secret": ""}

    # Mask key in logs for security
    masked = api_key[:6] + "..." + api_key[-4:]
    logger.info(f"Binance API key loaded: {masked}")

    return {"apiKey": api_key, "secret": api_secret}


def get_ollama_config() -> dict:
    """Get Ollama configuration."""
    load_env()
    return {
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        "model": os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
    }


def has_binance_keys() -> bool:
    """Check if Binance API keys are configured."""
    load_env()
    key = os.environ.get("BINANCE_API_KEY", "")
    return bool(key) and key != "your_api_key_here"
