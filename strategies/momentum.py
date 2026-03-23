"""
CryptoResearchLab — Strategy: Momentum (RSI + MACD + Rate of Change)
Classic momentum strategy: ride strong moves in the direction of momentum.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Evolved via genetic algorithm (2026-03-22, gen ~500, 1432 evals)
    # Walk-forward ROBUST: Sharpe=0.90, WF_Sharpe=1.58, 234 trades, $409 PnL
    "rsi_period": 14,
    "rsi_long_threshold": 63.4129,        # Evolved: high RSI for momentum confirm
    "rsi_short_threshold": 45.5328,       # Evolved: below 50 for short momentum
    "rsi_overbought": 69.9867,            # Evolved: tight overbought filter
    "rsi_oversold": 19.7349,              # Evolved: extreme oversold
    "roc_period": 12,
    "roc_min_magnitude": 0.5516,          # Evolved: moderate ROC minimum
    "require_volume_confirmation": True,
    "volume_surge_threshold": 1.311,      # Evolved: moderate volume bar
    "adx_min": 19.2893,                   # Evolved: moderate ADX
    "use_acceleration_filter": False,     # Evolved: no acceleration filter
    "stop_loss_atr_mult": 2.844,          # Evolved: moderate SL
    "take_profit_atr_mult": 6.0472,       # Evolved: wide TP
    "time_stop_hours": 36,
    "require_trend_regime": True,
    "allowed_regimes": ["trend", "breakout"],
}

def momentum_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    if bar_idx < 50:
        return None
    if position is not None:
        return None
    if p["require_trend_regime"] and regime not in p["allowed_regimes"]:
        return None

    current = df.iloc[bar_idx]
    rsi = current.get("rsi_14", 50)
    roc = current.get("roc_12", 0)
    adx = current.get("adx_14", 0)
    vol_ratio = current.get("volume_ratio", 1.0)
    accel = current.get("price_acceleration", 0)
    atr = current.get("atr_14", 0)

    if pd.isna(rsi) or pd.isna(roc) or pd.isna(adx) or atr <= 0:
        return None
    if adx < p["adx_min"]:
        return None
    if p["require_volume_confirmation"] and vol_ratio < p["volume_surge_threshold"]:
        return None

    close = current["close"]
    signal = None

    # Long: RSI above threshold + positive ROC + not overbought
    if (rsi > p["rsi_long_threshold"] and rsi < p["rsi_overbought"]
            and roc > p["roc_min_magnitude"]):
        if not p["use_acceleration_filter"] or accel > 0:
            signal = Signal(
                timestamp=int(current["timestamp"]),
                side="long",
                strength=min((rsi - 50) / 30, 1.0),
                strategy="momentum",
                reason=f"RSI={rsi:.0f} ROC={roc:.2f}% ADX={adx:.0f}",
                stop_loss=close - atr * p["stop_loss_atr_mult"],
                take_profit=close + atr * p["take_profit_atr_mult"],
                time_stop_hours=p["time_stop_hours"],
            )

    # Short: RSI below threshold + negative ROC + not oversold
    elif (rsi < p["rsi_short_threshold"] and rsi > p["rsi_oversold"]
          and roc < -p["roc_min_magnitude"]):
        if not p["use_acceleration_filter"] or accel < 0:
            signal = Signal(
                timestamp=int(current["timestamp"]),
                side="short",
                strength=min((50 - rsi) / 30, 1.0),
                strategy="momentum",
                reason=f"RSI={rsi:.0f} ROC={roc:.2f}% ADX={adx:.0f}",
                stop_loss=close + atr * p["stop_loss_atr_mult"],
                take_profit=close - atr * p["take_profit_atr_mult"],
                time_stop_hours=p["time_stop_hours"],
            )

    return signal
