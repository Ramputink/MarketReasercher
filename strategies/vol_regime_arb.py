"""
CryptoResearchLab — Strategy: Volatility Regime Arbitrage (VolRegimeArb)
Adapted volatility arbitrage for single-asset spot trading.

Core idea: Exploit mean-reversion in realized volatility.
When realized vol is abnormally LOW relative to its historical distribution,
expect expansion → position for a breakout.
When realized vol is abnormally HIGH, expect contraction → fade the move.

Unlike volatility_squeeze (which detects BB inside Keltner),
this strategy uses a statistical z-score of Garman-Klass volatility
across multiple timeframes to detect regime extremes.

Two modes:
  1. VOL_EXPANSION: Low vol → expect breakout → trade with momentum direction
  2. VOL_CONTRACTION: Extreme high vol → expect reversion → fade with RSI extremes

Performance: GK vol and z-scores are precomputed once via vectorized rolling
operations, avoiding O(n²) per-bar recalculation.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal


PARAMS = {
    # Volatility measurement
    "gk_fast_period": 10,            # Fast GK vol lookback (hours)
    "gk_slow_period": 50,            # Slow GK vol lookback (hours)
    "gk_baseline_period": 100,       # Baseline for z-score calculation

    # Vol expansion mode (low vol → breakout)
    "expansion_zscore_threshold": -1.2,  # GK vol z-score below this = compressed
    "expansion_min_bars_compressed": 5,  # Bars vol must stay compressed
    "expansion_momentum_period": 10,     # Momentum lookback for direction on release
    "expansion_adx_min": 18.0,           # Min ADX on expansion signal

    # Vol contraction mode (extreme vol → mean-revert)
    "contraction_zscore_threshold": 2.0, # GK vol z-score above this = extreme
    "contraction_rsi_oversold": 30.0,    # RSI threshold for long on contraction
    "contraction_rsi_overbought": 70.0,  # RSI threshold for short on contraction
    "enable_contraction_mode": True,     # Enable vol contraction trades

    # Volume confirmation
    "require_volume": True,
    "volume_threshold": 1.1,

    # Vol ratio filter (short-term vs long-term vol)
    "vol_ratio_min": 0.0,           # Min fast/slow vol ratio
    "vol_ratio_max": 3.0,           # Max fast/slow vol ratio

    # Risk parameters
    "stop_loss_atr_mult": 2.5,
    "take_profit_atr_mult": 4.5,
    "time_stop_hours": 36,

    # Regime filter
    "require_regime": False,
    "allowed_regimes": ["lateral", "breakout", "mean_reversion"],
}

# ── Module-level cache for precomputed GK columns ──
_cache_id = None       # id(df) to detect new DataFrames
_cache_params = None   # (fast, slow, baseline) to detect param changes


def _precompute_gk_columns(df: pd.DataFrame, fast: int, slow: int, baseline: int):
    """
    Vectorized Garman-Klass volatility at multiple scales + rolling z-score.
    Computed ONCE per backtest run, then accessed O(1) per bar.
    """
    global _cache_id, _cache_params

    cache_key = (id(df), fast, slow, baseline)
    if _cache_id == id(df) and _cache_params == (fast, slow, baseline):
        # Already computed for this exact df + params
        if "_gk_fast" in df.columns:
            return

    # Garman-Klass per-bar variance component
    with np.errstate(divide='ignore', invalid='ignore'):
        log_hl = np.log(df["high"] / df["low"]) ** 2
        log_co = np.log(df["close"] / df["open"]) ** 2

    gk_bar = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    # Replace inf/nan with NaN for clean rolling
    gk_bar = gk_bar.where(np.isfinite(gk_bar), np.nan)

    # Annualized GK vol = sqrt(rolling_mean(gk_bar) * 24 * 365)
    ann = 24 * 365

    gk_fast_var = gk_bar.rolling(fast, min_periods=max(5, fast // 2)).mean()
    gk_slow_var = gk_bar.rolling(slow, min_periods=max(5, slow // 2)).mean()

    # Clip negative variance to 0 before sqrt
    df["_gk_fast"] = np.sqrt(gk_fast_var.clip(lower=0) * ann)
    df["_gk_slow"] = np.sqrt(gk_slow_var.clip(lower=0) * ann)

    # Rolling z-score of fast GK vol over baseline window
    gk_f = df["_gk_fast"]
    rolling_mean = gk_f.rolling(baseline, min_periods=20).mean()
    rolling_std = gk_f.rolling(baseline, min_periods=20).std()
    # Guard against division by near-zero std
    safe_std = rolling_std.where(rolling_std > 1e-10, np.nan)
    df["_gk_zscore"] = (gk_f - rolling_mean) / safe_std

    # Fast/slow ratio
    safe_slow = df["_gk_slow"].where(df["_gk_slow"] > 1e-10, np.nan)
    df["_gk_ratio"] = df["_gk_fast"] / safe_slow

    _cache_id = id(df)
    _cache_params = (fast, slow, baseline)


def vol_regime_arb_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    lookback = p["gk_baseline_period"] + p["expansion_min_bars_compressed"] + 10
    if bar_idx < lookback:
        return None
    if position is not None:
        return None
    if p["require_regime"] and regime not in p["allowed_regimes"]:
        return None

    # ── Precompute vectorized GK columns (once per backtest) ──
    _precompute_gk_columns(df, p["gk_fast_period"], p["gk_slow_period"], p["gk_baseline_period"])

    current = df.iloc[bar_idx]
    atr = current.get("atr_14", 0)
    adx = current.get("adx_14", 0)
    rsi = current.get("rsi_14", 50)
    vol_ratio_feat = current.get("volume_ratio", 1.0)

    if pd.isna(atr) or atr <= 0:
        return None

    # ── Read precomputed values O(1) ──
    gk_fast = current.get("_gk_fast", np.nan)
    gk_slow = current.get("_gk_slow", np.nan)
    vol_zscore = current.get("_gk_zscore", np.nan)
    fast_slow_ratio = current.get("_gk_ratio", np.nan)

    if pd.isna(gk_fast) or pd.isna(vol_zscore):
        return None

    # ── Vol ratio filter ──
    if not pd.isna(fast_slow_ratio):
        if fast_slow_ratio < p["vol_ratio_min"] or fast_slow_ratio > p["vol_ratio_max"]:
            return None

    # ── Volume filter ──
    if p["require_volume"] and not pd.isna(vol_ratio_feat):
        if vol_ratio_feat < p["volume_threshold"]:
            return None

    close = current["close"]

    # ════════════════════════════════════════════════════
    # MODE 1: VOL EXPANSION (low vol → breakout expected)
    # ════════════════════════════════════════════════════
    if vol_zscore < p["expansion_zscore_threshold"]:
        # Check vol has been compressed for N bars using precomputed z-scores
        compressed_count = 0
        zscores = df["_gk_zscore"]
        for j in range(1, p["expansion_min_bars_compressed"] + 1):
            idx = bar_idx - j
            if idx < 0:
                break
            past_z = zscores.iat[idx]
            if not pd.isna(past_z) and past_z < p["expansion_zscore_threshold"]:
                compressed_count += 1

        if compressed_count < p["expansion_min_bars_compressed"] - 1:
            return None

        # ADX confirmation
        if pd.isna(adx) or adx < p["expansion_adx_min"]:
            return None

        # Direction from momentum
        mom_closes = df["close"].iloc[
            bar_idx - p["expansion_momentum_period"]:bar_idx + 1
        ].values
        x = np.arange(len(mom_closes))
        slope = np.polyfit(x, mom_closes, 1)[0]

        strength = min(abs(vol_zscore) / 3.0, 1.0)

        if slope > 0:
            return Signal(
                timestamp=int(current["timestamp"]),
                side="long",
                strength=strength,
                strategy="vol_regime_arb",
                reason=f"VOL_EXPANSION: z={vol_zscore:.2f}, GK={gk_fast:.4f}, slope>0",
                stop_loss=close - atr * p["stop_loss_atr_mult"],
                take_profit=close + atr * p["take_profit_atr_mult"],
                time_stop_hours=p["time_stop_hours"],
            )
        elif slope < 0:
            return Signal(
                timestamp=int(current["timestamp"]),
                side="short",
                strength=strength,
                strategy="vol_regime_arb",
                reason=f"VOL_EXPANSION: z={vol_zscore:.2f}, GK={gk_fast:.4f}, slope<0",
                stop_loss=close + atr * p["stop_loss_atr_mult"],
                take_profit=close - atr * p["take_profit_atr_mult"],
                time_stop_hours=p["time_stop_hours"],
            )

    # ════════════════════════════════════════════════════
    # MODE 2: VOL CONTRACTION (extreme vol → fade expected)
    # ════════════════════════════════════════════════════
    if p["enable_contraction_mode"] and vol_zscore > p["contraction_zscore_threshold"]:
        if pd.isna(rsi):
            return None

        strength = min(abs(vol_zscore) / 4.0, 1.0)

        # Extreme vol + oversold RSI → long (fade the crash)
        if rsi < p["contraction_rsi_oversold"]:
            return Signal(
                timestamp=int(current["timestamp"]),
                side="long",
                strength=strength,
                strategy="vol_regime_arb",
                reason=f"VOL_CONTRACTION: z={vol_zscore:.2f}, RSI={rsi:.1f} oversold",
                stop_loss=close - atr * p["stop_loss_atr_mult"],
                take_profit=close + atr * p["take_profit_atr_mult"],
                time_stop_hours=p["time_stop_hours"],
            )
        # Extreme vol + overbought RSI → short (fade the spike)
        elif rsi > p["contraction_rsi_overbought"]:
            return Signal(
                timestamp=int(current["timestamp"]),
                side="short",
                strength=strength,
                strategy="vol_regime_arb",
                reason=f"VOL_CONTRACTION: z={vol_zscore:.2f}, RSI={rsi:.1f} overbought",
                stop_loss=close + atr * p["stop_loss_atr_mult"],
                take_profit=close - atr * p["take_profit_atr_mult"],
                time_stop_hours=p["time_stop_hours"],
            )

    return None
