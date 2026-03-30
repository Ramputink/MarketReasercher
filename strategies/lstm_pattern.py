"""
CryptoResearchLab — Strategy: LSTM Multi-Variant Pattern Recognition

Combines clustering + LSTM sequence prediction with evolvable variant selection.

The evolution system can select between 7 clustering variants:
  - kmeans_8/20/50: Flat K-Means with increasing granularity
  - hier_20/50: 2-level hierarchical (3 macro direction × N micro shape)
  - bisect_20/50: BisectingKMeans (divisive tree, top-down splits)

Performance-critical: ALL cluster labels AND LSTM predictions are precomputed
in a single batch pass per backtest. The per-bar strategy function only does
dict lookups — no model inference, no O(n) operations.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

logger = logging.getLogger("lstm_pattern_strategy")

PARAMS = {
    # Model selection (evolvable!)
    "cluster_variant": "kmeans_50",   # Which clustering variant to use
    "window_size": 10,                # Must match trained model
    "seq_length": 20,                 # Must match trained LSTM

    # Signal thresholds (evolvable)
    "min_cluster_prob": 0.55,         # Min P(direction) from cluster profile
    "min_lstm_confidence": 0.45,      # Min P(direction) from LSTM
    "min_combined_confidence": 0.55,  # Min weighted combination
    "cluster_weight": 0.4,            # Weight of cluster signal vs LSTM
    "lstm_weight": 0.6,               # Weight of LSTM signal vs cluster

    # Filters (evolvable)
    "adx_min": 15.0,                  # Minimum ADX for trend confirmation
    "require_volume": True,
    "volume_threshold": 1.0,          # Min volume ratio
    "rsi_oversold": 25.0,             # RSI filter
    "rsi_overbought": 75.0,

    # Risk parameters (evolvable)
    "stop_loss_atr_mult": 2.5,
    "take_profit_atr_mult": 4.0,
    "time_stop_hours": 36,

    # Regime filter
    "require_regime": False,
    "allowed_regimes": ["trend", "breakout", "lateral"],
}

# ── Module-level model cache ──
_models_loaded = False
_loaded_key = None       # (variant_id, timeframe) tuple
_cluster_model = None    # Can be MiniBatchKMeans, BisectingKMeans, or HierarchicalClusterModel
_scaler = None
_cluster_profiles = None
_lstm_model = None
_n_clusters = 0
_bar_cluster = None       # np.ndarray of cluster IDs per bar
_bar_lstm_preds = None    # np.ndarray of shape (n_bars, 3) — prob_down/neutral/up per bar
_cache_df_id = None


def _detect_timeframe(df: pd.DataFrame) -> str:
    """Auto-detect timeframe from median bar spacing in the data."""
    if len(df) < 10 or "timestamp" not in df.columns:
        return "1h"
    diffs = np.diff(df["timestamp"].values[:100])
    median_ms = np.median(diffs[diffs > 0])
    if median_ms > 10_000_000:  # > ~2.8h → 4H
        return "4h"
    return "1h"


def _load_models(variant_id: str, timeframe: str):
    """
    Load pre-trained clustering variant + matching LSTM.
    Caches by (variant_id, timeframe) — reloads on change.
    """
    global _models_loaded, _loaded_key, _cluster_model, _scaler
    global _cluster_profiles, _lstm_model, _n_clusters

    key = (variant_id, timeframe)
    if _models_loaded and _loaded_key == key:
        return True

    # Log variant switch (important for debugging 10h+ evolution runs)
    if _loaded_key is not None and _loaded_key != key:
        logger.debug(f"Switching variant: {_loaded_key} → {key}")

    # Reset cache on model change
    _reset_cache()

    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

    try:
        # Try variant-specific model first
        from engine.multi_clustering import load_variant_model
        _cluster_model, _scaler, model_info = load_variant_model(
            variant_id, timeframe, model_dir,
        )
        _cluster_profiles = {p.cluster_id: p for p in model_info.cluster_profiles}
        _n_clusters = model_info.n_clusters

        from engine.lstm_pattern_model import load_lstm_pattern_model
        lstm_name = f"lstm_{variant_id}_{timeframe}"
        _lstm_model = load_lstm_pattern_model(model_dir, lstm_name)

        _models_loaded = True
        _loaded_key = key
        logger.info(f"Loaded variant={variant_id}, tf={timeframe}, k={_n_clusters}")
        return True

    except FileNotFoundError:
        # Fallback: try legacy model names (backward compat)
        try:
            from engine.pattern_clustering import load_cluster_model
            prefix = f"pattern_clusters_{timeframe}"
            _cluster_model, _scaler, model_info = load_cluster_model(model_dir, prefix)
            _cluster_profiles = {p.cluster_id: p for p in model_info.cluster_profiles}
            _n_clusters = model_info.n_clusters

            from engine.lstm_pattern_model import load_lstm_pattern_model
            lstm_name = f"lstm_pattern_{timeframe}"
            _lstm_model = load_lstm_pattern_model(model_dir, lstm_name)

            _models_loaded = True
            _loaded_key = key
            logger.info(f"Loaded LEGACY model (tf={timeframe}, k={_n_clusters})")
            return True
        except Exception as e2:
            logger.warning(f"Could not load any pattern models: {e2}")
            _models_loaded = False
            return False

    except Exception as e:
        logger.warning(f"Could not load variant {variant_id}/{timeframe}: {e}")
        _models_loaded = False
        return False


def _reset_cache():
    """Reset precomputed caches when switching models/datasets."""
    global _bar_cluster, _bar_lstm_preds, _cache_df_id
    _bar_cluster = None
    _bar_lstm_preds = None
    _cache_df_id = None


def _precompute_all(df: pd.DataFrame, window_size: int, seq_length: int):
    """
    Precompute BOTH cluster labels AND LSTM predictions for ALL bars in one pass.

    This is the key performance optimization: instead of calling model.predict()
    per bar during the backtest (~10ms × 8,760 bars = 87s), we do ONE batch
    prediction for all bars at once (~0.5s total).

    After this, the per-bar strategy function does only dict lookups — O(1).
    """
    global _bar_cluster, _bar_lstm_preds, _cache_df_id

    cache_key = (id(df), _loaded_key)
    if _cache_df_id == cache_key:
        return
    _cache_df_id = cache_key

    from engine.pattern_clustering import extract_candle_windows

    n_bars = len(df)

    # ── Step 1: Cluster all windows ──
    _bar_cluster = np.full(n_bars, -1, dtype=int)

    features, indices = extract_candle_windows(df, window_size=window_size, step=1)
    if len(features) == 0:
        _bar_lstm_preds = np.full((n_bars, 3), 1.0 / 3, dtype=np.float32)
        return

    features_scaled = _scaler.transform(features)
    labels = _cluster_model.predict(features_scaled)
    for label, idx in zip(labels, indices):
        if 0 <= idx < n_bars:
            _bar_cluster[idx] = label

    # Forward-fill cluster gaps
    last_valid = -1
    for i in range(n_bars):
        if _bar_cluster[i] >= 0:
            last_valid = _bar_cluster[i]
        elif last_valid >= 0:
            _bar_cluster[i] = last_valid

    # ── Step 2: Build ALL LSTM sequences at once ──
    tech_cols = []
    available = df.columns.tolist()
    for col in ["rsi_14", "adx_14", "volume_ratio", "ret_zscore_20",
                 "bb_pct_b", "pressure_imbalance", "gk_vol", "vol_ratio"]:
        if col in available:
            tech_cols.append(col)

    n_features = _n_clusters + len(tech_cols)

    # Validate shape against model
    expected_input = _lstm_model.input_shape  # (None, seq_length, n_features)
    if expected_input[-1] is not None and expected_input[-1] != n_features:
        logger.error(
            f"LSTM shape mismatch: model expects {expected_input[-1]} features "
            f"but variant has {_n_clusters} clusters + {len(tech_cols)} tech = {n_features}. "
            f"Using uniform predictions."
        )
        _bar_lstm_preds = np.full((n_bars, 3), 1.0 / 3, dtype=np.float32)
        return

    # Determine which bars have valid sequences (bar_idx >= seq_length + window_size)
    min_bar = seq_length + window_size
    valid_bars = list(range(min_bar, n_bars))

    if not valid_bars:
        _bar_lstm_preds = np.full((n_bars, 3), 1.0 / 3, dtype=np.float32)
        return

    # Build batch: one sequence per valid bar
    # Pre-extract tech feature arrays for speed
    tech_arrays = []
    for col in tech_cols:
        arr = df[col].values.astype(np.float32)
        # Replace NaN/Inf with 0
        arr = np.where(np.isfinite(arr), arr, 0.0)
        tech_arrays.append(arr)

    batch_size = len(valid_bars)
    batch = np.zeros((batch_size, seq_length, n_features), dtype=np.float32)

    for batch_idx, bar_idx in enumerate(valid_bars):
        for t in range(seq_length):
            bi = bar_idx - seq_length + t
            cid = int(_bar_cluster[bi]) if 0 <= bi < n_bars and _bar_cluster[bi] >= 0 else -1
            if 0 <= cid < _n_clusters:
                batch[batch_idx, t, cid] = 1.0
            for j, arr in enumerate(tech_arrays):
                if 0 <= bi < n_bars:
                    batch[batch_idx, t, _n_clusters + j] = arr[bi]

    # ── Step 3: Single batch LSTM prediction ──
    # This replaces ~8,000 individual model.predict() calls with ONE call
    try:
        preds = _lstm_model.predict(batch, batch_size=256, verbose=0)
        dir_probs = preds[0]  # shape (batch_size, 3) — prob_down/neutral/up
    except Exception as e:
        logger.error(f"LSTM batch prediction failed: {e}. Using uniform probs.")
        dir_probs = np.full((batch_size, 3), 1.0 / 3, dtype=np.float32)

    # Store predictions indexed by bar
    _bar_lstm_preds = np.full((n_bars, 3), 1.0 / 3, dtype=np.float32)
    for batch_idx, bar_idx in enumerate(valid_bars):
        _bar_lstm_preds[bar_idx] = dir_probs[batch_idx]

    _cache_df_id = (id(df), _loaded_key)
    logger.debug(f"Precomputed {batch_size} LSTM predictions in batch")


def lstm_pattern_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    lookback = max(p["window_size"], p["seq_length"]) + 10
    if bar_idx < lookback:
        return None
    if position is not None:
        return None
    if p["require_regime"] and regime not in p["allowed_regimes"]:
        return None

    # ── Auto-detect timeframe, then load the selected variant ──
    detected_tf = _detect_timeframe(df)
    variant_id = p["cluster_variant"]
    if not _load_models(variant_id, detected_tf):
        return None

    # ── Precompute clusters + LSTM predictions (once per backtest+variant) ──
    _precompute_all(df, p["window_size"], p["seq_length"])

    current = df.iloc[bar_idx]
    atr = current.get("atr_14", 0)
    adx = current.get("adx_14", 0)
    rsi = current.get("rsi_14", 50)
    vol_ratio = current.get("volume_ratio", 1.0)

    if pd.isna(atr) or atr <= 0:
        return None

    # ── ADX filter ──
    if not pd.isna(adx) and adx < p["adx_min"]:
        return None

    # ── Volume filter ──
    if p["require_volume"] and not pd.isna(vol_ratio):
        if vol_ratio < p["volume_threshold"]:
            return None

    # ── Get current cluster ──
    current_cluster = _bar_cluster[bar_idx] if bar_idx < len(_bar_cluster) else -1
    if current_cluster < 0:
        return None

    # ── Cluster profile signal ──
    profile = _cluster_profiles.get(current_cluster)
    if profile is None or profile.n_samples < 10:
        return None

    cluster_prob_up = profile.prob_up
    cluster_prob_down = profile.prob_down

    # ── LSTM prediction (precomputed — just a lookup!) ──
    lstm_probs = _bar_lstm_preds[bar_idx]
    lstm_prob_down = float(lstm_probs[0])
    lstm_prob_up = float(lstm_probs[2])
    lstm_confidence = max(lstm_prob_up, lstm_prob_down)

    # ── Confidence check ──
    if lstm_confidence < p["min_lstm_confidence"]:
        return None

    # ── Combine signals ──
    cw = p["cluster_weight"]
    lw = p["lstm_weight"]
    total_w = cw + lw

    combined_up = (cluster_prob_up * cw + lstm_prob_up * lw) / total_w
    combined_down = (cluster_prob_down * cw + lstm_prob_down * lw) / total_w

    # ── Decision ──
    close = current["close"]
    min_combined = p["min_combined_confidence"]

    if combined_up > combined_down and combined_up > min_combined:
        if not pd.isna(rsi) and rsi > p["rsi_overbought"]:
            return None
        if cluster_prob_up < p["min_cluster_prob"]:
            return None

        strength = min(combined_up, 1.0)
        return Signal(
            timestamp=int(current["timestamp"]),
            side="long",
            strength=strength,
            strategy="lstm_pattern",
            reason=(f"v={variant_id} c={current_cluster} P(up)={cluster_prob_up:.2f}, "
                    f"LSTM={lstm_prob_up:.2f}, comb={combined_up:.2f}"),
            stop_loss=close - atr * p["stop_loss_atr_mult"],
            take_profit=close + atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    elif combined_down > combined_up and combined_down > min_combined:
        if not pd.isna(rsi) and rsi < p["rsi_oversold"]:
            return None
        if cluster_prob_down < p["min_cluster_prob"]:
            return None

        strength = min(combined_down, 1.0)
        return Signal(
            timestamp=int(current["timestamp"]),
            side="short",
            strength=strength,
            strategy="lstm_pattern",
            reason=(f"v={variant_id} c={current_cluster} P(dn)={cluster_prob_down:.2f}, "
                    f"LSTM={lstm_prob_down:.2f}, comb={combined_down:.2f}"),
            stop_loss=close + atr * p["stop_loss_atr_mult"],
            take_profit=close - atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    return None
