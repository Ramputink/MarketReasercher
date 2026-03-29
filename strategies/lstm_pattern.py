"""
CryptoResearchLab — Strategy: LSTM Multi-Variant Pattern Recognition

Combines clustering + LSTM sequence prediction with evolvable variant selection.

The evolution system can select between 7 clustering variants:
  - kmeans_8/20/50: Flat K-Means with increasing granularity
  - hier_20/50: 2-level hierarchical (3 macro direction × N micro shape)
  - bisect_20/50: BisectingKMeans (divisive tree, top-down splits)

This creates a "genealogical tree" where:
  - Different granularity levels compete (8 vs 20 vs 50 clusters)
  - Different algorithms compete (flat vs hierarchical vs divisive)
  - Evolution discovers which variant produces the best trading signals
  - The hierarchical variants explicitly model pattern sub-categories:
    e.g., "slow bullish channel" vs "sharp bullish breakout"

Flow per bar:
  1. Load variant model (selected by evolvable `cluster_variant` param)
  2. Classify current N-candle window into a cluster
  3. Look up cluster profile (historical forward return distribution)
  4. Feed cluster sequence to LSTM for transition prediction
  5. Combine cluster probability + LSTM confidence → generate signal

Evolvable: cluster_variant, confidence thresholds, filters, risk params.
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
_bar_cluster = None
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
    """Reset precomputed cluster cache when switching models/datasets."""
    global _bar_cluster, _cache_df_id
    _bar_cluster = None
    _cache_df_id = None


def _precompute_clusters(df: pd.DataFrame, window_size: int):
    """Precompute cluster labels for all bars (once per backtest+variant combo)."""
    global _bar_cluster, _cache_df_id

    # Cache key includes both df identity AND loaded model key
    # so switching variants on the same df triggers recomputation
    cache_key = (id(df), _loaded_key)
    if _cache_df_id == cache_key:
        return
    _cache_df_id = cache_key  # Will be set again at end, but set early to avoid recursion

    from engine.pattern_clustering import extract_candle_windows

    n_bars = len(df)
    _bar_cluster = np.full(n_bars, -1, dtype=int)

    features, indices = extract_candle_windows(df, window_size=window_size, step=1)
    if len(features) > 0:
        features_scaled = _scaler.transform(features)
        labels = _cluster_model.predict(features_scaled)
        for label, idx in zip(labels, indices):
            if 0 <= idx < n_bars:
                _bar_cluster[idx] = label

    # Forward-fill gaps
    last_valid = -1
    for i in range(n_bars):
        if _bar_cluster[i] >= 0:
            last_valid = _bar_cluster[i]
        elif last_valid >= 0:
            _bar_cluster[i] = last_valid

    _cache_df_id = (id(df), _loaded_key)


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

    # ── Precompute clusters (once per backtest) ──
    _precompute_clusters(df, p["window_size"])

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

    # ── LSTM prediction ──
    from engine.lstm_pattern_model import predict_pattern_direction
    lstm_pred = predict_pattern_direction(
        _lstm_model, df, bar_idx, _bar_cluster, _n_clusters,
        seq_length=p["seq_length"],
    )
    if lstm_pred is None:
        return None

    lstm_prob_up = lstm_pred["prob_up"]
    lstm_prob_down = lstm_pred["prob_down"]
    lstm_confidence = lstm_pred["confidence"]

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
