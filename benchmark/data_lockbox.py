"""
benchmark.data_lockbox — Frozen data + the holdout contract
===========================================================

The benchmark is only meaningful if it runs on *exactly the same bytes* every
time, and on data the evolution has *never seen*. This module enforces both:

    FREEZE      Build OHLCV+features+regime snapshots once, write them to
                benchmark/snapshots/, and record a SHA-256 of the content in a
                manifest. Every later run reads the frozen parquet, never the
                live exchange. => reproducibility.

    SPLIT       The final `lockbox_days` of the primary series are the LOCKBOX.
                Evolution must train ONLY on the in-sample part. The benchmark
                judges on the lockbox. => no selection-on-test.

    CONTRACT    `get_evolution_window()` returns the in-sample slice that
                evolution is *allowed* to use. The lockbox cutoff timestamp is
                published so an evolution run can assert it never reads past it.

Offline self-tests use `synthetic_ohlcv()` — a deterministic geometric brownian
motion series — so the harness can be unit-tested with no network.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
MANIFEST_PATH = os.path.join(SNAPSHOT_DIR, "manifest.json")


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot manifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SnapshotManifest:
    timeframe: str
    history_days: int
    lockbox_days: int
    symbols: dict          # symbol -> {"path", "sha256", "n_bars", "first_ts", "last_ts"}
    lockbox_cutoff_ts: int  # primary symbol: evolution must not read bars >= this
    built_at: str

    def fingerprint(self) -> str:
        payload = json.dumps(
            {s: v["sha256"] for s, v in sorted(self.symbols.items())},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(symbol: str, tf: str) -> str:
    return f"{symbol.replace('/', '')}_{tf}.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Build (one-off, requires the project's data pipeline + network)
# ─────────────────────────────────────────────────────────────────────────────

def build_snapshots(spec, force_refresh: bool = False) -> SnapshotManifest:
    """
    Build & freeze snapshots for the primary + cross symbols using the project's
    own data + feature pipeline, so the benchmark sees identical features to
    production. Adds the per-bar `_regime` label. Writes parquet + manifest.

    Requires network (ccxt) on first build. Run via:  python -m benchmark snapshot
    """
    # Imported lazily so the rest of the package works without the heavy stack.
    from datetime import datetime, timezone
    from config import DataConfig, TimeFrame
    from engine.data_ingestion import DataIngestionEngine
    from engine.features import build_all_features
    from mirofish.scenario_engine import classify_regime_quantitative

    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    tf_map = {tf.value: tf for tf in TimeFrame}
    tf_enum = tf_map[spec.timeframe]

    data_cfg = DataConfig(
        symbol=spec.primary_symbol,
        cross_assets=list(spec.cross_symbols),
        history_days=spec.history_days,
        data_dir="data",
    )
    engine = DataIngestionEngine(data_cfg)

    symbols = {}
    primary_cutoff_ts = 0
    all_symbols = [spec.primary_symbol] + list(spec.cross_symbols)

    for sym in all_symbols:
        raw = engine.fetch_ohlcv(sym, tf_enum, since_days=spec.history_days,
                                 force_refresh=force_refresh)
        if raw is None or len(raw) == 0:
            raise RuntimeError(f"No data fetched for {sym}; cannot freeze snapshot.")
        feat = build_all_features(raw).reset_index(drop=True)

        # per-bar regime label (matches what evolution feeds strategies)
        regimes = []
        for i in range(len(feat)):
            try:
                rc = classify_regime_quantitative(feat, bar_idx=i)
                regimes.append(rc.regime.value if hasattr(rc.regime, "value") else str(rc.regime))
            except Exception:
                regimes.append("unknown")
        feat["_regime"] = regimes

        path = os.path.join(SNAPSHOT_DIR, _safe_name(sym, spec.timeframe))
        feat.to_parquet(path, index=False)
        symbols[sym] = {
            "path": os.path.relpath(path, SNAPSHOT_DIR),
            "sha256": _sha256_file(path),
            "n_bars": int(len(feat)),
            "first_ts": int(feat["timestamp"].iloc[0]),
            "last_ts": int(feat["timestamp"].iloc[-1]),
        }
        if sym == spec.primary_symbol:
            cutoff_ms = spec.lockbox_days * 86_400_000
            primary_cutoff_ts = int(feat["timestamp"].iloc[-1]) - cutoff_ms

    manifest = SnapshotManifest(
        timeframe=spec.timeframe,
        history_days=spec.history_days,
        lockbox_days=spec.lockbox_days,
        symbols=symbols,
        lockbox_cutoff_ts=primary_cutoff_ts,
        built_at=datetime.now(timezone.utc).isoformat(),
    )
    with open(MANIFEST_PATH, "w") as f:
        json.dump(asdict(manifest), f, indent=2)
    return manifest


# ─────────────────────────────────────────────────────────────────────────────
# Load (every benchmark run uses this — frozen, offline, hashed)
# ─────────────────────────────────────────────────────────────────────────────

class DataLockbox:
    """Reads frozen snapshots and exposes the in-sample / lockbox split."""

    def __init__(self, spec):
        self.spec = spec
        if not os.path.exists(MANIFEST_PATH):
            raise FileNotFoundError(
                "No frozen data snapshot found. Build it once with:\n"
                "    python -m benchmark snapshot\n"
                "(requires network on first run; afterwards the benchmark is offline & reproducible)."
            )
        with open(MANIFEST_PATH) as f:
            self.manifest = SnapshotManifest(**json.load(f))
        self._verify_integrity()

    def _verify_integrity(self):
        """Refuse to run if the frozen bytes changed since freezing."""
        for sym, meta in self.manifest.symbols.items():
            path = os.path.join(SNAPSHOT_DIR, meta["path"])
            if not os.path.exists(path):
                raise FileNotFoundError(f"Snapshot missing for {sym}: {path}")
            if _sha256_file(path) != meta["sha256"]:
                raise RuntimeError(
                    f"Snapshot for {sym} was modified after freezing "
                    f"(hash mismatch). The benchmark refuses to run on tampered data. "
                    f"Re-freeze with `python -m benchmark snapshot --force`."
                )

    def manifest_hash(self) -> str:
        return self.manifest.fingerprint()

    def _load(self, symbol: str) -> pd.DataFrame:
        meta = self.manifest.symbols[symbol]
        return pd.read_parquet(os.path.join(SNAPSHOT_DIR, meta["path"]))

    # ── primary splits ───────────────────────────────────────────────────────
    def primary_full(self) -> pd.DataFrame:
        return self._load(self.spec.primary_symbol)

    def in_sample(self) -> pd.DataFrame:
        """The slice evolution is ALLOWED to use (everything before the cutoff)."""
        df = self.primary_full()
        return df[df["timestamp"] < self.manifest.lockbox_cutoff_ts].reset_index(drop=True)

    def lockbox(self) -> pd.DataFrame:
        """The held-out slice the benchmark judges on (never given to evolution)."""
        df = self.primary_full()
        return df[df["timestamp"] >= self.manifest.lockbox_cutoff_ts].reset_index(drop=True)

    def lockbox_cutoff_ts(self) -> int:
        return self.manifest.lockbox_cutoff_ts

    # ── cross-asset (also held out: use each coin's lockbox-equivalent tail) ──
    def cross_asset_lockboxes(self) -> dict[str, pd.DataFrame]:
        out = {}
        cutoff_ms = self.spec.lockbox_days * 86_400_000
        for sym in self.spec.cross_symbols:
            if sym not in self.manifest.symbols:
                continue
            df = self._load(sym)
            cutoff = int(df["timestamp"].iloc[-1]) - cutoff_ms
            out[sym] = df[df["timestamp"] >= cutoff].reset_index(drop=True)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# The evolution-side contract
# ─────────────────────────────────────────────────────────────────────────────

def get_evolution_window(spec) -> pd.DataFrame:
    """
    Call this from auto_evolve.py instead of loading the full series. It returns
    ONLY the in-sample data; the lockbox stays sealed. This is the single line
    that converts the current (leaky) setup into an honest one.
    """
    return DataLockbox(spec).in_sample()


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic synthetic data — for offline self-tests of the harness itself
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_ohlcv(
    n_bars: int = 4000,
    seed: int = 7,
    start_price: float = 1.0,
    drift: float = 0.0,
    vol: float = 0.01,
    start_ts_ms: int = 1_600_000_000_000,
    bar_ms: int = 3_600_000,
) -> pd.DataFrame:
    """Deterministic GBM OHLCV with the feature columns the harness/strategies need."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n_bars)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n_bars)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n_bars)))
    volume = rng.uniform(1e5, 5e5, n_bars)
    ts = start_ts_ms + np.arange(n_bars) * bar_ms

    df = pd.DataFrame({
        "timestamp": ts, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })
    # minimal feature set used by the harness + simple strategies
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - np.concatenate([[start_price], close[:-1]])),
        np.abs(low - np.concatenate([[start_price], close[:-1]])),
    ])
    df["atr_14"] = pd.Series(tr).rolling(14, min_periods=1).mean()
    df["adx_14"] = 25.0
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20, min_periods=1).mean()
    df["_regime"] = "trend"
    return df
