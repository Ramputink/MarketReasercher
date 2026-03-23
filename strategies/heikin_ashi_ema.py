"""
CryptoResearchLab — Strategy: Heikin Ashi Smoothed + EMA Crossover
Uses Heikin Ashi candles to smooth noise, combined with EMA cross for entries.
HA color flip + EMA alignment = high-quality trend continuation signals.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Evolved via genetic algorithm (2026-03-22, gen ~300, 1073 evals)
    # Walk-forward ROBUST: Sharpe=1.36, WF_Sharpe=1.07, 53 trades, $346 PnL
    "ema_fast": 21,                       # Evolved: slower fast EMA
    "ema_slow": 45,                       # Evolved: tighter gap
    "require_ha_color_flip": False,       # Evolved: no HA flip required
    "require_consecutive_ha": 2,          # Evolved: 2 consecutive HA bars
    "adx_min": 24.9722,                   # Evolved: moderate ADX
    "require_volume": True,               # Evolved: volume required
    "volume_threshold": 1.0717,           # Evolved: near-average volume
    "stop_loss_atr_mult": 3.5641,         # Evolved: wide SL
    "take_profit_atr_mult": 5.6392,       # Evolved: wide TP
    "time_stop_hours": 48,
    "require_trend_regime": True,
    "allowed_regimes": ["trend", "breakout"],
}


def _compute_heikin_ashi(df: pd.DataFrame, bar_idx: int, lookback: int = 5):
    """Compute Heikin Ashi candles up to bar_idx."""
    start = max(0, bar_idx - lookback - 1)
    sl = df.iloc[start:bar_idx + 1]

    ha_close = (sl["open"].values + sl["high"].values + sl["low"].values + sl["close"].values) / 4
    ha_open = np.zeros(len(sl))
    ha_open[0] = (sl["open"].values[0] + sl["close"].values[0]) / 2

    for i in range(1, len(sl)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2

    # HA is bullish when close > open (green)
    ha_bullish = ha_close > ha_open
    return ha_bullish, ha_close, ha_open


def heikin_ashi_ema_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    min_lookback = max(p["ema_slow"] + 10, 60)
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

    # EMA crossover
    ema_fast_series = df["close"].iloc[max(0, bar_idx - p["ema_slow"] - 10):bar_idx + 1].ewm(
        span=p["ema_fast"], adjust=False).mean()
    ema_slow_series = df["close"].iloc[max(0, bar_idx - p["ema_slow"] - 10):bar_idx + 1].ewm(
        span=p["ema_slow"], adjust=False).mean()

    if len(ema_fast_series) < 2:
        return None

    ema_fast_now = ema_fast_series.iloc[-1]
    ema_slow_now = ema_slow_series.iloc[-1]
    ema_fast_prev = ema_fast_series.iloc[-2]
    ema_slow_prev = ema_slow_series.iloc[-2]

    # Detect EMA cross or existing alignment
    bullish_cross = ema_fast_prev <= ema_slow_prev and ema_fast_now > ema_slow_now
    bearish_cross = ema_fast_prev >= ema_slow_prev and ema_fast_now < ema_slow_now
    bullish_aligned = ema_fast_now > ema_slow_now
    bearish_aligned = ema_fast_now < ema_slow_now

    # Heikin Ashi confirmation
    n_consec = p["require_consecutive_ha"]
    ha_bullish, _, _ = _compute_heikin_ashi(df, bar_idx, lookback=n_consec + 2)

    if len(ha_bullish) < n_consec + 1:
        return None

    # Check for HA color flip + consecutive candles
    ha_now_bull = ha_bullish[-1]
    ha_prev_bull = ha_bullish[-2]

    if p["require_ha_color_flip"]:
        # Need a fresh color flip
        long_ha_ok = not ha_prev_bull and ha_now_bull  # flip to green
        short_ha_ok = ha_prev_bull and not ha_now_bull  # flip to red

        # Also accept N consecutive same-color HA
        if not long_ha_ok and n_consec > 1:
            long_ha_ok = all(ha_bullish[-i] for i in range(1, n_consec + 1))
        if not short_ha_ok and n_consec > 1:
            short_ha_ok = all(not ha_bullish[-i] for i in range(1, n_consec + 1))
    else:
        long_ha_ok = ha_now_bull
        short_ha_ok = not ha_now_bull

    # Combine signals
    side = None
    if (bullish_cross or bullish_aligned) and long_ha_ok:
        side = "long"
    elif (bearish_cross or bearish_aligned) and short_ha_ok:
        side = "short"

    # Fresh cross gets priority
    if bullish_cross and long_ha_ok:
        side = "long"
    elif bearish_cross and short_ha_ok:
        side = "short"
    elif not bullish_cross and not bearish_cross:
        # Only aligned — need HA flip for entry
        if not (p["require_ha_color_flip"] and (
            (side == "long" and not ha_prev_bull and ha_now_bull) or
            (side == "short" and ha_prev_bull and not ha_now_bull)
        )):
            if not bullish_cross and not bearish_cross:
                return None

    if side is None:
        return None

    strength = 0.5
    if bullish_cross or bearish_cross:
        strength += 0.2
    if adx > 30:
        strength += 0.15
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
        strategy="heikin_ashi_ema",
        reason=f"HA_{'bull' if side == 'long' else 'bear'}_EMA{p['ema_fast']}/{p['ema_slow']}_ADX={adx:.0f}",
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_stop_hours=p["time_stop_hours"],
        metadata={"ema_fast": ema_fast_now, "ema_slow": ema_slow_now, "regime": regime},
    )
