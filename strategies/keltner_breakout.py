"""
CryptoResearchLab — Strategy: Keltner Channel Breakout
ATR-based channel breakout with momentum confirmation.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Evolved via genetic algorithm (2026-03-22, gen ~400, 1759 evals)
    # Walk-forward ROBUST: Sharpe=1.08, WF_Sharpe=2.30, 142 trades, $361 PnL
    "ema_period": 19,                     # Evolved: shorter EMA
    "atr_mult": 3.404,                    # Evolved: standard ATR multiplier
    "require_momentum": True,
    "rsi_long_min": 50.9851,              # Evolved: above 55 for longs
    "rsi_short_max": 45.4013,             # Evolved: below 52 for shorts
    "adx_min": 14.7614,                   # Evolved: moderate ADX
    "require_close_outside": True,
    "consecutive_bars_outside": 1,
    "volume_confirmation": True,
    "volume_threshold": 1.98,             # Evolved: high volume bar
    "stop_loss_atr_mult": 2.6146,         # Evolved: moderate SL
    "take_profit_atr_mult": 6.2393,       # Evolved: wide TP
    "time_stop_hours": 36,
    "require_regime": True,
    "allowed_regimes": ["breakout", "trend"],
}

def keltner_breakout_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    lookback = p["ema_period"] + 10
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
    if pd.isna(adx) or adx < p["adx_min"]:
        return None
    if p["volume_confirmation"] and vol_ratio < p["volume_threshold"]:
        return None

    # Compute Keltner Channel
    closes = df["close"].iloc[bar_idx - p["ema_period"]:bar_idx + 1]
    ema_center = closes.ewm(span=p["ema_period"], adjust=False).mean().iloc[-1]
    upper = ema_center + atr * p["atr_mult"]
    lower = ema_center - atr * p["atr_mult"]

    if pd.isna(ema_center):
        return None

    close = current["close"]
    signal = None

    # Long: close breaks above upper Keltner + bullish momentum
    if close > upper:
        if p["require_momentum"] and (pd.isna(rsi) or rsi < p["rsi_long_min"]):
            return None
        # Check consecutive bars (if required)
        if p["consecutive_bars_outside"] > 1 and bar_idx > 0:
            prev_close = df.iloc[bar_idx - 1]["close"]
            prev_ema = closes.ewm(span=p["ema_period"], adjust=False).mean().iloc[-2]
            prev_atr = df.iloc[bar_idx - 1].get("atr_14", atr)
            prev_upper = prev_ema + prev_atr * p["atr_mult"]
            if prev_close <= prev_upper:
                return None

        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="long",
            strength=min((close - upper) / atr, 1.0),
            strategy="keltner_breakout",
            reason=f"Keltner breakout up: {close:.4f} > {upper:.4f}",
            stop_loss=ema_center,
            take_profit=close + atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    # Short: close breaks below lower Keltner + bearish momentum
    elif close < lower:
        if p["require_momentum"] and (pd.isna(rsi) or rsi > p["rsi_short_max"]):
            return None

        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="short",
            strength=min((lower - close) / atr, 1.0),
            strategy="keltner_breakout",
            reason=f"Keltner breakout down: {close:.4f} < {lower:.4f}",
            stop_loss=ema_center,
            take_profit=close - atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    return signal
