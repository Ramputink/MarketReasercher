"""
CryptoResearchLab — Strategy: Dual Moving Average Crossover
Classic golden/death cross with modern filters.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Evolved via genetic algorithm (2026-03-22, gen 447, 1330 evals)
    # Walk-forward ROBUST: Sharpe=1.84, WF_Sharpe=3.13, PF=3.21, 41 trades, $35 PnL
    "fast_period": 17,                    # Evolved: slightly longer fast MA
    "slow_period": 27,                    # Evolved: tighter gap between fast/slow
    "signal_type": "sma",                 # Evolved: SMA outperformed EMA
    "require_adx": False,                 # Evolved: no ADX filter
    "adx_min": 28.9929,
    "require_volume": True,               # Evolved: volume required
    "volume_threshold": 1.7066,           # Evolved: high volume bar
    "require_price_above_slow": True,
    "pullback_tolerance_pct": 0.3,
    "stop_loss_atr_mult": 3.5518,         # Evolved: wide stops
    "take_profit_atr_mult": 6.7914,       # Evolved: wide TP
    "time_stop_hours": 36,
    "require_trend_regime": True,
    "allowed_regimes": ["trend", "breakout"],
}

def dual_ma_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    lookback = p["slow_period"] + 5
    if bar_idx < lookback:
        return None
    if position is not None:
        return None
    if p["require_trend_regime"] and regime not in p["allowed_regimes"]:
        return None

    current = df.iloc[bar_idx]
    prev = df.iloc[bar_idx - 1]
    atr = current.get("atr_14", 0)
    adx = current.get("adx_14", 0)
    vol_ratio = current.get("volume_ratio", 1.0)

    if pd.isna(atr) or atr <= 0:
        return None
    if p["require_adx"] and (pd.isna(adx) or adx < p["adx_min"]):
        return None
    if p["require_volume"] and vol_ratio < p["volume_threshold"]:
        return None

    # Compute MAs
    closes = df["close"].iloc[bar_idx - p["slow_period"] - 1:bar_idx + 1]
    if p["signal_type"] == "ema":
        fast_ma = closes.ewm(span=p["fast_period"], adjust=False).mean()
        slow_ma = closes.ewm(span=p["slow_period"], adjust=False).mean()
    else:
        fast_ma = closes.rolling(p["fast_period"]).mean()
        slow_ma = closes.rolling(p["slow_period"]).mean()

    fast_now = fast_ma.iloc[-1]
    fast_prev = fast_ma.iloc[-2]
    slow_now = slow_ma.iloc[-1]
    slow_prev = slow_ma.iloc[-2]

    if pd.isna(fast_now) or pd.isna(slow_now) or pd.isna(fast_prev) or pd.isna(slow_prev):
        return None

    close = current["close"]
    signal = None

    # Golden cross: fast crosses above slow
    if fast_prev <= slow_prev and fast_now > slow_now:
        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="long",
            strength=min(abs(fast_now - slow_now) / atr, 1.0),
            strategy="dual_ma",
            reason=f"Golden cross: fast={fast_now:.4f} > slow={slow_now:.4f}",
            stop_loss=close - atr * p["stop_loss_atr_mult"],
            take_profit=close + atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    # Death cross: fast crosses below slow
    elif fast_prev >= slow_prev and fast_now < slow_now:
        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="short",
            strength=min(abs(slow_now - fast_now) / atr, 1.0),
            strategy="dual_ma",
            reason=f"Death cross: fast={fast_now:.4f} < slow={slow_now:.4f}",
            stop_loss=close + atr * p["stop_loss_atr_mult"],
            take_profit=close - atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    return signal
