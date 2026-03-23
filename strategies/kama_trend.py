"""
CryptoResearchLab — Strategy: KAMA (Kaufman Adaptive Moving Average)
Self-adapting MA: fast in trends, slow in noise. No manual regime switching needed.
Backtest with KAMA+ATR showed Sharpe 1.76.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Evolved via genetic algorithm (2026-03-22, gen 296, 4211 evals)
    # WF ROBUST variant: Sharpe=3.32, WF_Sharpe=4.25, PF=8.83, 18 trades, $402 PnL
    # NOTE: low trade count (18) — use with caution, may be overfit
    "er_period": 15,                      # Evolved: longer efficiency ratio window
    "fast_sc": 5,                         # Evolved: slower fast constant
    "slow_sc": 29,                        # Evolved: near default
    "signal_period": 5,                   # Evolved: same
    "slope_threshold_pct": 0.0701,        # Evolved: moderate slope requirement
    "adx_min": 10.4446,                   # Evolved: low ADX floor = more entries
    "require_volume": False,
    "volume_threshold": 0.8,
    "stop_loss_atr_mult": 2.2972,         # Evolved: tighter SL
    "take_profit_atr_mult": 5.3074,       # Evolved: wide TP
    "time_stop_hours": 48,
    "require_trend_regime": True,
    "allowed_regimes": ["trend", "breakout", "lateral"],
}


def _compute_kama(close_arr, er_period, fast_sc, slow_sc):
    """Compute Kaufman Adaptive Moving Average."""
    n = len(close_arr)
    if n < er_period + 1:
        return None

    fast_alpha = 2.0 / (fast_sc + 1)
    slow_alpha = 2.0 / (slow_sc + 1)

    kama = np.full(n, np.nan)
    kama[er_period] = close_arr[er_period]

    for i in range(er_period + 1, n):
        # Efficiency Ratio = direction / volatility
        direction = abs(close_arr[i] - close_arr[i - er_period])
        volatility = sum(abs(close_arr[j] - close_arr[j - 1]) for j in range(i - er_period + 1, i + 1))

        if volatility == 0:
            er = 0
        else:
            er = direction / volatility

        # Smoothing constant = (ER * (fast - slow) + slow)^2
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        kama[i] = kama[i - 1] + sc * (close_arr[i] - kama[i - 1])

    return kama


def kama_trend_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    min_lookback = max(p["er_period"] + p["signal_period"] + 10, 60)
    if bar_idx < min_lookback:
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

    if pd.isna(atr_val) or atr_val <= 0:
        return None
    if pd.isna(adx) or adx < p["adx_min"]:
        return None
    if p["require_volume"] and (pd.isna(vol_ratio) or vol_ratio < p["volume_threshold"]):
        return None

    # Compute KAMA
    lookback = min(bar_idx + 1, 200)
    close_arr = df["close"].iloc[bar_idx + 1 - lookback:bar_idx + 1].values
    kama = _compute_kama(close_arr, p["er_period"], p["fast_sc"], p["slow_sc"])

    if kama is None:
        return None

    last = len(kama) - 1
    sig = p["signal_period"]

    if last < sig + 1 or np.isnan(kama[last]) or np.isnan(kama[last - sig]):
        return None

    # KAMA slope over signal_period
    kama_slope_pct = (kama[last] - kama[last - sig]) / kama[last - sig] * 100

    # Price vs KAMA
    price_above = close > kama[last]
    price_below = close < kama[last]

    # KAMA direction change detection
    kama_rising_now = kama[last] > kama[last - 1]
    kama_rising_prev = kama[last - 1] > kama[last - 2] if last >= 2 else kama_rising_now

    # Signal: price crosses KAMA + KAMA slope is significant
    threshold = p["slope_threshold_pct"]
    side = None

    if price_above and kama_slope_pct > threshold:
        # Bullish: KAMA turning up + price above
        if not kama_rising_prev and kama_rising_now:
            side = "long"  # KAMA direction flip
        elif kama_slope_pct > threshold * 2:
            # Strong upward KAMA
            prev_close = float(df.iloc[bar_idx - 1]["close"])
            if prev_close <= kama[last - 1]:
                side = "long"  # Price just crossed above KAMA

    elif price_below and kama_slope_pct < -threshold:
        if kama_rising_prev and not kama_rising_now:
            side = "short"
        elif kama_slope_pct < -threshold * 2:
            prev_close = float(df.iloc[bar_idx - 1]["close"])
            if prev_close >= kama[last - 1]:
                side = "short"

    if side is None:
        return None

    strength = 0.5
    strength += min(0.25, abs(kama_slope_pct) / 1.0)
    if adx > 25:
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
        strategy="kama_trend",
        reason=f"KAMA_slope={kama_slope_pct:.2f}%_ADX={adx:.0f}",
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_stop_hours=p["time_stop_hours"],
        metadata={"kama_val": kama[last], "kama_slope_pct": kama_slope_pct, "regime": regime},
    )
