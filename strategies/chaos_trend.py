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

Performance: EMAs are precomputed once via vectorized pandas operations.
Hurst is computed per-bar (inherently sequential) but optimized with
numpy-only operations and minimal allocations.
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

# ── Module-level cache for precomputed EMA columns ──
_cache_id = None
_cache_params = None


def _estimate_hurst(closes: np.ndarray) -> float:
    """
    Estimate Hurst exponent using multi-scale Rescaled Range (R/S) analysis.
    Optimized: numpy-only, no pandas, minimal allocations.
    """
    n = len(closes)
    if n < 20:
        return 0.5

    # Log returns (avoid log(0) with clip)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratios = closes[1:] / closes[:-1]
    ratios = np.where(np.isfinite(ratios) & (ratios > 0), ratios, 1.0)
    returns = np.log(ratios)

    if len(returns) < 10:
        return 0.5

    # Multi-scale R/S
    scales = []
    rs_values = []

    for chunk_size in [16, 32, 64]:
        if chunk_size > len(returns):
            continue
        n_chunks = len(returns) // chunk_size
        if n_chunks < 1:
            continue

        rs_sum = 0.0
        rs_count = 0
        for i in range(n_chunks):
            chunk = returns[i * chunk_size:(i + 1) * chunk_size]
            mean_c = np.mean(chunk)
            std_c = np.std(chunk, ddof=1)
            if std_c < 1e-12:
                continue
            deviations = np.cumsum(chunk - mean_c)
            R = np.max(deviations) - np.min(deviations)
            rs_sum += R / std_c
            rs_count += 1

        if rs_count > 0:
            scales.append(np.log(chunk_size))
            rs_values.append(np.log(rs_sum / rs_count))

    if len(scales) < 2:
        # Fallback: single-scale R/S on full series
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        if std_r < 1e-12:
            return 0.5
        deviations = np.cumsum(returns - mean_r)
        R = np.max(deviations) - np.min(deviations)
        rs = R / std_r
        if rs <= 0:
            return 0.5
        return float(np.clip(np.log(rs) / np.log(len(returns)), 0.0, 1.0))

    # Linear regression: log(R/S) = H * log(n) + c
    coeffs = np.polyfit(scales, rs_values, 1)
    return float(np.clip(coeffs[0], 0.0, 1.0))


def _precompute_emas(df: pd.DataFrame, fast: int, slow: int):
    """Precompute EMA columns once per backtest run."""
    global _cache_id, _cache_params

    if _cache_id == id(df) and _cache_params == (fast, slow):
        if "_ema_fast" in df.columns:
            return

    df["_ema_fast"] = df["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    df["_ema_slow"] = df["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()

    _cache_id = id(df)
    _cache_params = (fast, slow)


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

    # ── Precompute EMAs once per backtest ──
    _precompute_emas(df, p["ema_fast"], p["ema_slow"])

    current = df.iloc[bar_idx]
    atr = current.get("atr_14", 0)
    adx = current.get("adx_14", 0)
    vol_ratio = current.get("volume_ratio", 1.0)

    if pd.isna(atr) or atr <= 0:
        return None
    if pd.isna(adx) or adx < p["adx_min"]:
        return None

    # ── Hurst Exponent (sequential, but numpy-optimized) ──
    closes = df["close"].values[bar_idx - p["hurst_window"] + 1:bar_idx + 1]
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
            return None

    # ── EMA direction (precomputed, O(1) access) ──
    fast_ema = df["_ema_fast"].iat[bar_idx]
    slow_ema = df["_ema_slow"].iat[bar_idx]

    if pd.isna(fast_ema) or pd.isna(slow_ema):
        return None

    bullish = fast_ema > slow_ema
    bearish = fast_ema < slow_ema

    if p["require_ema_alignment"] and not (bullish or bearish):
        return None

    # ── Momentum confirmation ──
    mom_closes = df["close"].values[bar_idx - p["momentum_period"]:bar_idx + 1]
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

    if bullish:
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
    elif bearish:
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
