"""
sweep.build_data — freeze BTC/ETH multi-timeframe snapshots
===========================================================

Fetches BTC/USDT and ETH/USDT at 15m / 1h / 4h / 1d (HISTORY_DAYS), builds the
same features + per-bar regime label as the rest of the system, and freezes each
to sweep/snapshots/<SYMBOL>_<tf>.parquet with a SHA-256 manifest and a published
lockbox cutoff (final LOCKBOX_DAYS sealed).

Run once (needs network the first time):
    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m sweep.build_data
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

from sweep import SWEEP_SYMBOLS, SWEEP_TIMEFRAMES, HISTORY_DAYS, LOCKBOX_DAYS

SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
MANIFEST = os.path.join(SNAP_DIR, "manifest.json")


def _safe(symbol: str, tf: str) -> str:
    return f"{symbol.replace('/', '')}_{tf}.parquet"


def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build():
    from config import DataConfig, TimeFrame
    from engine.data_ingestion import DataIngestionEngine
    from engine.features import build_all_features
    from mirofish.scenario_engine import classify_regime_quantitative

    os.makedirs(SNAP_DIR, exist_ok=True)
    tf_map = {tf.value: tf for tf in TimeFrame}

    # every base TF plus every confirmation TF must be fetched
    all_tfs = set(SWEEP_TIMEFRAMES.keys())
    for confs in SWEEP_TIMEFRAMES.values():
        all_tfs.update(confs)

    engine = DataIngestionEngine(DataConfig(history_days=HISTORY_DAYS, data_dir="data"))
    entries = {}

    for sym in SWEEP_SYMBOLS:
        for tf in sorted(all_tfs):
            print(f"  fetching {sym} {tf} ...", flush=True)
            raw = engine.fetch_ohlcv(sym, tf_map[tf], since_days=HISTORY_DAYS, force_refresh=False)
            if raw is None or len(raw) == 0:
                raise RuntimeError(f"No data for {sym} {tf}")
            feat = build_all_features(raw).reset_index(drop=True)
            regimes = []
            for i in range(len(feat)):
                try:
                    rc = classify_regime_quantitative(feat, bar_idx=i)
                    regimes.append(rc.regime.value if hasattr(rc.regime, "value") else str(rc.regime))
                except Exception:
                    regimes.append("unknown")
            feat["_regime"] = regimes

            path = os.path.join(SNAP_DIR, _safe(sym, tf))
            feat.to_parquet(path, index=False)
            cutoff = int(feat["timestamp"].iloc[-1]) - LOCKBOX_DAYS * 86_400_000
            entries[f"{sym}|{tf}"] = {
                "symbol": sym, "tf": tf,
                "path": os.path.relpath(path, SNAP_DIR), "sha256": _sha(path),
                "n_bars": int(len(feat)),
                "first_ts": int(feat["timestamp"].iloc[0]),
                "last_ts": int(feat["timestamp"].iloc[-1]),
                "lockbox_cutoff_ts": cutoff,
            }
            print(f"    {len(feat)} bars  sha={entries[f'{sym}|{tf}']['sha256'][:12]}", flush=True)

    manifest = {
        "symbols": SWEEP_SYMBOLS,
        "base_timeframes": list(SWEEP_TIMEFRAMES.keys()),
        "confirmation_map": SWEEP_TIMEFRAMES,
        "history_days": HISTORY_DAYS,
        "lockbox_days": LOCKBOX_DAYS,
        "entries": entries,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nFrozen {len(entries)} (symbol,tf) snapshots -> {SNAP_DIR}")
    return manifest


if __name__ == "__main__":
    build()
