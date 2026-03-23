"""
CryptoResearchLab — Strategy: Donchian Channel Breakout
Turtle Trading inspired: enter on N-period high/low breakouts.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Evolved via genetic algorithm (2026-03-22, gen 144, 2739 evals)
    # Walk-forward ROBUST: Sharpe=2.79, WF_Sharpe=2.99, PF=1.89, 132 trades, $1015 PnL
    "entry_period": 41,                   # Evolved: longer channel = bigger breakouts
    "exit_period": 10,                    # Evolved: keep exit tight
    "require_volume_confirmation": True,
    "volume_surge_threshold": 1.5495,     # Evolved: moderate volume bar
    "adx_min": 27.6524,                   # Evolved: strong trend required
    "atr_filter_mult": 0.2,              # Evolved: minimal ATR filter
    "use_volatility_filter": True,
    "stop_loss_atr_mult": 2.681,          # Evolved: medium stops
    "take_profit_atr_mult": 7.8633,       # Evolved: very wide TP = ride big breakouts
    "time_stop_hours": 48,
    "require_breakout_regime": True,
    "allowed_regimes": ["breakout", "trend"],
}

def donchian_breakout_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    lookback = max(p["entry_period"], p["exit_period"]) + 5
    if bar_idx < lookback:
        return None
    if position is not None:
        return None
    if p["require_breakout_regime"] and regime not in p["allowed_regimes"]:
        return None

    current = df.iloc[bar_idx]
    window = df.iloc[bar_idx - p["entry_period"]:bar_idx]
    atr = current.get("atr_14", 0)
    adx = current.get("adx_14", 0)
    vol_ratio = current.get("volume_ratio", 1.0)

    if pd.isna(atr) or atr <= 0 or pd.isna(adx):
        return None
    if adx < p["adx_min"]:
        return None
    if p["require_volume_confirmation"] and vol_ratio < p["volume_surge_threshold"]:
        return None

    channel_high = window["high"].max()
    channel_low = window["low"].min()
    close = current["close"]

    # Volatility filter: channel must be wide enough (not noise)
    if p["use_volatility_filter"]:
        channel_width = channel_high - channel_low
        if channel_width < atr * p["atr_filter_mult"]:
            return None

    signal = None

    # Long: close breaks above channel high
    if close > channel_high:
        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="long",
            strength=min((close - channel_high) / atr, 1.0),
            strategy="donchian_breakout",
            reason=f"Break above {p['entry_period']}-bar high {channel_high:.4f}",
            stop_loss=close - atr * p["stop_loss_atr_mult"],
            take_profit=close + atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    # Short: close breaks below channel low
    elif close < channel_low:
        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="short",
            strength=min((channel_low - close) / atr, 1.0),
            strategy="donchian_breakout",
            reason=f"Break below {p['entry_period']}-bar low {channel_low:.4f}",
            stop_loss=close + atr * p["stop_loss_atr_mult"],
            take_profit=close - atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    return signal
