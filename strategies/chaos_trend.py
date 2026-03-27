"""
CryptoResearchLab — Strategy: Chaos Trend (Hurst + Fractal Dimension)
Adaptive trend-following for chaotic environments.

Uses the Hurst exponent to measure market persistence:
  - H > 0.5 → persistent (trend-following regime, follow momentum)
  - H ≈ 0.5 → random walk (skip)
  - H < 0.5 → antipersistent / mean-reverting (skip or fade)

Combined with fractal dimension D = 2 - H as confirmation.
Only trades when the market shows statistically significant persistence
AND a directional bias is confirmed by momentum + ADX.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal


PARAMS = {
    # Hurst exponent parameters
    "hurst_window": 100,              # Rolling window for Hurst estimation
    "hurst_min": 0.55,                # Minimum H for trend persistence
    "hurst_max": 0.85,                # Maximum H (>0.85 may indicate regime shift)

    # Trend confirmation
    "ema_fast": 12,                   # Fast EMA for direction
    "ema_slow": 26,                   # Slow EMA for direction
    "adx_min": 20.0,                  # Minimum ADX for trend strength
    "require_ema_alignment": True,    # Require fast > slow (long) or fast < slow (short)

    # Momentum filter
    "momentum_period": 14,            # Lookback for linear regression slope
    "min_momentum_strength": 0.0,     # Minimum |slope|/ATR ratio

    # Volume filter
    "require_volume_surge": True,
    "volume_threshold": 1.2,          # Minimum volume ratio vs 20-SMA

    # Fractal dimension filter
    "use_fractal_filter": True,       # Use D = 2 - H as secondary confirmation
    "fractal_max": 1.45,              # D < 1.45 → trending; D > 1.5 → noisy

    # Risk parameters
    "stop_loss_atr_mult": 3.0,
    "take_profit_atr_mult": 5.5,
    "time_stop_hours": 48,

    # Regime filter
    "require_regime": True,
    "allowed_regimes": ["trend", "breakout"],
}


def _estimate_hurst(series: np.ndarray) -> float:
    """
    Estimate Hurst exponent using Rescaled Range (R/S) analysis.

    H = log(R/S) / log(n)

    For speed, uses a simplified single-scale R/S on the full window.
    More robust than DFA for our bar-by-bar use case.
    """
    n = len(series)
    if n < 20:
        return 0.5  # Not enough data, assume random walk

    # Log returns
    returns = np.diff(np.log(series))
    if len(returns) < 10:
        return 0.5

    # Multi-scale R/S for better estimate
    scales = []
    rs_values = []

    for chunk_size in [16, 32, 64]:
        if chunk_size > len(returns):
            continue
        n_chunks = len(returns) // chunk_size
        if n_chunks < 1:
            continue

        rs_list = []
        for i in range(n_chunks):
            chunk = returns[i * chunk_size:(i + 1) * chunk_size]
            mean_chunk = np.mean(chunk)
            deviations = np.cumsum(chunk - mean_chunk)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(chunk, ddof=1)
            if S > 1e-12:
                rs_list.append(R / S)

        if rs_list:
            scales.append(np.log(chunk_size))
            rs_values.append(np.log(np.mean(rs_list)))

    if len(scales) < 2:
        # Fallback: single-scale R/S
        mean_r = np.mean(returns)
        deviations = np.cumsum(returns - mean_r)
        R = np.max(deviations) - np.min(deviations)
        S = np.std(returns, ddof=1)
        if S < 1e-12:
            return 0.5
        rs = R / S
        if rs <= 0:
            return 0.5
        return np.clip(np.log(rs) / np.log(n), 0.0, 1.0)

    # Linear regression: log(R/S) = H * log(n) + c
    coeffs = np.polyfit(scales, rs_values, 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))


def chaos_trend_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    lookback = max(p["hurst_window"], p["ema_slow"], p["momentum_period"]) + 10
    if bar_idx < lookback:
        return None
    if position is not None:
        return None
    if p["require_regime"] and regime not in p["allowed_regimes"]:
        return None

    current = df.iloc[bar_idx]
    atr = current.get("atr_14", 0)
    adx = current.get("adx_14", 0)
    vol_ratio = current.get("volume_ratio", 1.0)

    if pd.isna(atr) or atr <= 0:
        return None
    if pd.isna(adx) or adx < p["adx_min"]:
        return None

    # ── Hurst Exponent ──
    closes = df["close"].iloc[bar_idx - p["hurst_window"] + 1:bar_idx + 1].values
    if len(closes) < p["hurst_window"]:
        return None

    hurst = _estimate_hurst(closes)

    # Filter: only trade in persistent regimes
    if hurst < p["hurst_min"] or hurst > p["hurst_max"]:
        return None

    # ── Fractal dimension filter ──
    if p["use_fractal_filter"]:
        fractal_dim = 2.0 - hurst
        if fractal_dim > p["fractal_max"]:
            return None  # Too noisy / choppy

    # ── EMA direction ──
    fast_ema = df["close"].iloc[bar_idx - p["ema_fast"]:bar_idx + 1].ewm(
        span=p["ema_fast"], adjust=False
    ).mean().iloc[-1]
    slow_ema = df["close"].iloc[bar_idx - p["ema_slow"]:bar_idx + 1].ewm(
        span=p["ema_slow"], adjust=False
    ).mean().iloc[-1]

    if pd.isna(fast_ema) or pd.isna(slow_ema):
        return None

    bullish = fast_ema > slow_ema
    bearish = fast_ema < slow_ema

    if p["require_ema_alignment"] and not (bullish or bearish):
        return None

    # ── Momentum confirmation ──
    mom_closes = df["close"].iloc[bar_idx - p["momentum_period"]:bar_idx + 1].values
    x = np.arange(len(mom_closes))
    slope = np.polyfit(x, mom_closes, 1)[0]
    mom_strength = abs(slope) / atr

    if mom_strength < p["min_momentum_strength"]:
        return None

    # Direction must agree: EMA alignment + momentum slope
    if bullish and slope <= 0:
        return None
    if bearish and slope >= 0:
        return None

    # ── Volume filter ──
    if p["require_volume_surge"] and not pd.isna(vol_ratio):
        if vol_ratio < p["volume_threshold"]:
            return None

    close = current["close"]
    strength = min(mom_strength * 5, 1.0)

    if bullish and slope > 0:
        return Signal(
            timestamp=int(current["timestamp"]),
            side="long",
            strength=strength,
            strategy="chaos_trend",
            reason=f"Hurst={hurst:.3f}, D={2-hurst:.3f}, slope={slope:.6f}",
            stop_loss=close - atr * p["stop_loss_atr_mult"],
            take_profit=close + atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )
    elif bearish and slope < 0:
        return Signal(
            timestamp=int(current["timestamp"]),
            side="short",
            strength=strength,
            strategy="chaos_trend",
            reason=f"Hurst={hurst:.3f}, D={2-hurst:.3f}, slope={slope:.6f}",
            stop_loss=close + atr * p["stop_loss_atr_mult"],
            take_profit=close - atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    return None
