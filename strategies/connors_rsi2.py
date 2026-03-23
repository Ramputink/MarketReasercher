"""
CryptoResearchLab — Strategy: Connors RSI-2 (Aggressive Mean Reversion)
Uses ultra-short RSI(2) to detect extreme pullbacks within established trends.
Larry Connors' classic: buy RSI(2)<10 in uptrend, sell RSI(2)>90 in downtrend.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    "rsi_period": 2,
    "entry_threshold_long": 10.0,
    "entry_threshold_short": 90.0,
    "exit_threshold_long": 65.0,
    "exit_threshold_short": 35.0,
    "trend_ema_period": 200,
    "require_adx": True,
    "adx_min": 15.0,
    "require_volume": False,
    "volume_threshold": 0.8,
    "stop_loss_atr_mult": 2.0,
    "take_profit_atr_mult": 3.0,
    "time_stop_hours": 24,
    "require_trend_regime": True,
    "allowed_regimes": ["trend", "lateral", "mean_reversion"],
}


def _rsi_n(close_vals, period):
    """Compute RSI for a specific period on an array of close prices."""
    if len(close_vals) < period + 1:
        return np.nan
    deltas = np.diff(close_vals[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def connors_rsi2_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    min_lookback = max(p["trend_ema_period"], 60)
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

    # Compute RSI(2)
    close_arr = df["close"].iloc[max(0, bar_idx - p["rsi_period"] - 5):bar_idx + 1].values
    rsi2 = _rsi_n(close_arr, p["rsi_period"])
    if np.isnan(rsi2):
        return None

    # Trend filter: EMA of trend_ema_period
    ema_period = p["trend_ema_period"]
    ema_slice = df["close"].iloc[max(0, bar_idx - ema_period):bar_idx + 1]
    if len(ema_slice) < ema_period:
        return None
    trend_ema = ema_slice.ewm(span=ema_period, adjust=False).mean().iloc[-1]

    is_uptrend = close > trend_ema
    is_downtrend = close < trend_ema

    # ADX filter
    if p["require_adx"] and (pd.isna(adx) or adx < p["adx_min"]):
        return None

    # Volume filter
    if p["require_volume"] and (pd.isna(vol_ratio) or vol_ratio < p["volume_threshold"]):
        return None

    # Signal logic
    side = None
    if is_uptrend and rsi2 < p["entry_threshold_long"]:
        side = "long"
    elif is_downtrend and rsi2 > p["entry_threshold_short"]:
        side = "short"

    if side is None:
        return None

    # Strength based on how extreme the RSI is
    if side == "long":
        strength = min(1.0, 0.5 + (p["entry_threshold_long"] - rsi2) / 20.0)
    else:
        strength = min(1.0, 0.5 + (rsi2 - p["entry_threshold_short"]) / 20.0)

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
        strategy="connors_rsi2",
        reason=f"RSI2={rsi2:.1f}_trend={'up' if is_uptrend else 'down'}",
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_stop_hours=p["time_stop_hours"],
        metadata={"rsi2": rsi2, "trend_ema": trend_ema, "regime": regime},
    )
