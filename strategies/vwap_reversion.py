"""
CryptoResearchLab — Strategy: VWAP Reversion
Mean reversion to VWAP with volume and momentum filters.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    "vwap_deviation_pct": 1.5,
    "rsi_oversold": 30.0,
    "rsi_overbought": 70.0,
    "adx_max": 30.0,
    "require_volume_spike": True,
    "volume_spike_threshold": 1.5,
    "require_wick_rejection": True,
    "wick_ratio_min": 0.4,
    "stop_loss_atr_mult": 2.0,
    "take_profit_atr_mult": 2.5,
    "time_stop_hours": 24,
    "require_regime": True,
    "allowed_regimes": ["lateral", "mean_reversion"],
}

def vwap_reversion_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    if bar_idx < 50:
        return None
    if position is not None:
        return None
    if p["require_regime"] and regime not in p["allowed_regimes"]:
        return None

    current = df.iloc[bar_idx]
    vwap = current.get("vwap_20", None)
    rsi = current.get("rsi_14", 50)
    adx = current.get("adx_14", 0)
    atr = current.get("atr_14", 0)
    vol_ratio = current.get("volume_ratio", 1.0)
    body_wick = current.get("body_wick_ratio", 0.5)
    buying_pressure = current.get("buying_pressure", 0.5)

    if vwap is None or pd.isna(vwap) or vwap <= 0:
        return None
    if pd.isna(atr) or atr <= 0 or pd.isna(rsi):
        return None
    if not pd.isna(adx) and adx > p["adx_max"]:
        return None

    close = current["close"]
    deviation_pct = (close - vwap) / vwap * 100
    signal = None

    # Long: price well below VWAP + oversold RSI
    if (deviation_pct < -p["vwap_deviation_pct"]
            and rsi < p["rsi_oversold"]):
        if p["require_volume_spike"] and vol_ratio < p["volume_spike_threshold"]:
            return None
        if p["require_wick_rejection"] and buying_pressure < p["wick_ratio_min"]:
            return None

        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="long",
            strength=min(abs(deviation_pct) / 3, 1.0),
            strategy="vwap_reversion",
            reason=f"VWAP dev={deviation_pct:.1f}% RSI={rsi:.0f}",
            stop_loss=close - atr * p["stop_loss_atr_mult"],
            take_profit=vwap,  # Target = revert to VWAP
            time_stop_hours=p["time_stop_hours"],
        )

    # Short: price well above VWAP + overbought RSI
    elif (deviation_pct > p["vwap_deviation_pct"]
          and rsi > p["rsi_overbought"]):
        if p["require_volume_spike"] and vol_ratio < p["volume_spike_threshold"]:
            return None
        selling_pressure = current.get("selling_pressure", 0.5)
        if p["require_wick_rejection"] and selling_pressure < p["wick_ratio_min"]:
            return None

        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="short",
            strength=min(abs(deviation_pct) / 3, 1.0),
            strategy="vwap_reversion",
            reason=f"VWAP dev=+{deviation_pct:.1f}% RSI={rsi:.0f}",
            stop_loss=close + atr * p["stop_loss_atr_mult"],
            take_profit=vwap,  # Target = revert to VWAP
            time_stop_hours=p["time_stop_hours"],
        )

    return signal
