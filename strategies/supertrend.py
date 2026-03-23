"""
CryptoResearchLab — Strategy: SuperTrend
ATR-based trailing stop that flips direction on price cross.
Excellent trend-following with dynamic exits.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Evolved via genetic algorithm (2026-03-22, gen ~300, 1019 evals)
    # Walk-forward ROBUST: Sharpe=1.20, WF_Sharpe=1.10, 69 trades, $361 PnL
    "atr_period": 14,                     # Evolved: standard
    "multiplier": 3.8957,                 # Evolved: wide multiplier = fewer flips
    "adx_min": 17.233,                    # Evolved: low ADX floor
    "require_volume": True,               # Evolved: volume required
    "volume_threshold": 1.0648,           # Evolved: near-average volume
    "stop_loss_atr_mult": 3.0251,         # Evolved: moderate SL
    "take_profit_atr_mult": 5.8416,       # Evolved: wide TP
    "time_stop_hours": 48,
    "require_trend_regime": True,
    "allowed_regimes": ["trend", "breakout"],
}


def _compute_supertrend(df: pd.DataFrame, bar_idx: int, atr_period: int, mult: float):
    """Compute SuperTrend up to bar_idx (no lookahead)."""
    if bar_idx < atr_period + 1:
        return None, None, None

    lookback = min(bar_idx + 1, 200)
    start = max(0, bar_idx + 1 - lookback)
    sl = df.iloc[start:bar_idx + 1]

    high = sl["high"].values
    low = sl["low"].values
    close = sl["close"].values

    # ATR
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    atr_vals = np.full(len(close), np.nan)
    if len(tr) < atr_period:
        return None, None, None
    atr_vals[atr_period] = np.mean(tr[:atr_period])
    for i in range(atr_period + 1, len(close)):
        atr_vals[i] = (atr_vals[i - 1] * (atr_period - 1) + tr[i - 1]) / atr_period

    # SuperTrend bands
    n = len(close)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 = bullish, -1 = bearish

    for i in range(atr_period, n):
        if np.isnan(atr_vals[i]):
            continue
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + mult * atr_vals[i]
        lower_band[i] = hl2 - mult * atr_vals[i]

        # Clamp bands
        if i > atr_period:
            if lower_band[i] < lower_band[i - 1] and close[i - 1] > lower_band[i - 1]:
                lower_band[i] = lower_band[i - 1]
            if upper_band[i] > upper_band[i - 1] and close[i - 1] < upper_band[i - 1]:
                upper_band[i] = upper_band[i - 1]

        # Direction
        if i == atr_period:
            direction[i] = 1 if close[i] > upper_band[i] else -1
        else:
            if direction[i - 1] == 1:
                direction[i] = -1 if close[i] < lower_band[i] else 1
            else:
                direction[i] = 1 if close[i] > upper_band[i] else -1

        supertrend[i] = lower_band[i] if direction[i] == 1 else upper_band[i]

    last = n - 1
    prev = last - 1
    if prev < atr_period:
        return None, None, None

    return direction[last], direction[prev], supertrend[last]


def supertrend_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    if bar_idx < 60:
        return None
    if position is not None:
        return None
    if p["require_trend_regime"] and regime not in p["allowed_regimes"]:
        return None

    current = df.iloc[bar_idx]
    close = float(current["close"])
    atr_val = current.get("atr_14", 0)
    adx = current.get("adx_14", 0)
    vol_ratio = current.get("volume_ratio", 1.0)

    if pd.isna(atr_val) or atr_val <= 0 or pd.isna(adx):
        return None

    # ADX filter
    if adx < p["adx_min"]:
        return None

    # Volume filter
    if p["require_volume"]:
        if pd.isna(vol_ratio) or vol_ratio < p["volume_threshold"]:
            return None

    # Compute SuperTrend
    dir_now, dir_prev, st_val = _compute_supertrend(
        df, bar_idx, p["atr_period"], p["multiplier"]
    )
    if dir_now is None:
        return None

    # Signal on direction flip
    if dir_now == dir_prev:
        return None  # No flip

    side = "long" if dir_now == 1 else "short"

    strength = 0.6
    if adx > 30:
        strength += 0.15
    if vol_ratio and vol_ratio > 1.5:
        strength += 0.1
    strength = min(strength, 1.0)

    if side == "long":
        stop_loss = close - p["stop_loss_atr_mult"] * atr_val
        take_profit = close + p["take_profit_atr_mult"] * atr_val
    else:
        stop_loss = close + p["stop_loss_atr_mult"] * atr_val
        take_profit = close - p["take_profit_atr_mult"] * atr_val

    return Signal(
        timestamp=int(current["timestamp"]),
        side=side,
        strength=strength,
        strategy="supertrend",
        reason=f"ST_flip_{side}_ADX={adx:.0f}",
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_stop_hours=p["time_stop_hours"],
        metadata={"supertrend_val": st_val, "adx": adx, "regime": regime},
    )
