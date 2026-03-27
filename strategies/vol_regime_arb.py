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


def _garman_klass_vol(df: pd.DataFrame, end_idx: int, period: int) -> float:
    """
    Garman-Klass volatility estimator over a rolling window.
    GK = sqrt( mean(0.5 * ln(H/L)^2 - (2ln2 - 1) * ln(C/O)^2) * annualization )
    Returns annualized vol (assuming hourly bars, 24*365).
    """
    start = max(0, end_idx - period + 1)
    if end_idx - start + 1 < max(5, period // 2):
        return np.nan

    h = df["high"].iloc[start:end_idx + 1].values
    l = df["low"].iloc[start:end_idx + 1].values
    o = df["open"].iloc[start:end_idx + 1].values
    c = df["close"].iloc[start:end_idx + 1].values

    # Avoid log(0) or negative values
    with np.errstate(divide='ignore', invalid='ignore'):
        log_hl = np.log(h / l) ** 2
        log_co = np.log(c / o) ** 2

    valid = np.isfinite(log_hl) & np.isfinite(log_co)
    if valid.sum() < 5:
        return np.nan

    gk_var = np.mean(0.5 * log_hl[valid] - (2 * np.log(2) - 1) * log_co[valid])
    if gk_var < 0:
        gk_var = 0.0

    # Annualize for hourly data
    return float(np.sqrt(gk_var * 24 * 365))


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

    current = df.iloc[bar_idx]
    atr = current.get("atr_14", 0)
    adx = current.get("adx_14", 0)
    rsi = current.get("rsi_14", 50)
    vol_ratio = current.get("volume_ratio", 1.0)

    if pd.isna(atr) or atr <= 0:
        return None

    # ── Compute Garman-Klass volatility at multiple scales ──
    gk_fast = _garman_klass_vol(df, bar_idx, p["gk_fast_period"])
    gk_slow = _garman_klass_vol(df, bar_idx, p["gk_slow_period"])
    gk_baseline = _garman_klass_vol(df, bar_idx, p["gk_baseline_period"])

    if np.isnan(gk_fast) or np.isnan(gk_slow) or np.isnan(gk_baseline):
        return None

    # ── Vol z-score: how unusual is current vol vs baseline? ──
    # Compute rolling vol series for z-score
    vol_series = []
    for i in range(max(0, bar_idx - p["gk_baseline_period"]), bar_idx + 1):
        v = _garman_klass_vol(df, i, p["gk_fast_period"])
        if not np.isnan(v):
            vol_series.append(v)

    if len(vol_series) < 20:
        return None

    vol_arr = np.array(vol_series)
    vol_mean = np.mean(vol_arr)
    vol_std = np.std(vol_arr)

    if vol_std < 1e-10:
        return None

    vol_zscore = (gk_fast - vol_mean) / vol_std

    # ── Vol ratio filter ──
    if gk_slow > 1e-10:
        fast_slow_ratio = gk_fast / gk_slow
    else:
        fast_slow_ratio = 1.0

    if fast_slow_ratio < p["vol_ratio_min"] or fast_slow_ratio > p["vol_ratio_max"]:
        return None

    # ── Volume filter ──
    if p["require_volume"] and not pd.isna(vol_ratio):
        if vol_ratio < p["volume_threshold"]:
            return None

    close = current["close"]

    # ════════════════════════════════════════════════════
    # MODE 1: VOL EXPANSION (low vol → breakout expected)
    # ════════════════════════════════════════════════════
    if vol_zscore < p["expansion_zscore_threshold"]:
        # Check vol has been compressed for N bars
        compressed_count = 0
        for j in range(1, p["expansion_min_bars_compressed"] + 1):
            if bar_idx - j < p["gk_fast_period"]:
                break
            past_gk = _garman_klass_vol(df, bar_idx - j, p["gk_fast_period"])
            if not np.isnan(past_gk):
                past_z = (past_gk - vol_mean) / vol_std
                if past_z < p["expansion_zscore_threshold"]:
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
                reason=f"VOL_EXPANSION: z={vol_zscore:.2f}, GK_fast={gk_fast:.4f}, slope>0",
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
                reason=f"VOL_EXPANSION: z={vol_zscore:.2f}, GK_fast={gk_fast:.4f}, slope<0",
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
