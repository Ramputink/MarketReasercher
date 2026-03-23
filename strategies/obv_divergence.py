"""
CryptoResearchLab — Strategy: OBV Divergence
On-Balance Volume divergence from price — smart money detection.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    "lookback": 20,
    "min_price_move_pct": 0.5,
    "obv_divergence_threshold": 0.3,
    "rsi_confirm": True,
    "rsi_oversold": 35.0,
    "rsi_overbought": 65.0,
    "adx_max": 40.0,
    "stop_loss_atr_mult": 2.0,
    "take_profit_atr_mult": 3.0,
    "time_stop_hours": 30,
    "require_regime": False,
    "allowed_regimes": ["lateral", "mean_reversion", "trend", "breakout"],
}

def obv_divergence_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    if bar_idx < p["lookback"] + 10:
        return None
    if position is not None:
        return None
    if p["require_regime"] and regime not in p["allowed_regimes"]:
        return None

    current = df.iloc[bar_idx]
    atr = current.get("atr_14", 0)
    rsi = current.get("rsi_14", 50)
    obv_div = current.get("obv_divergence", 0)

    if pd.isna(atr) or atr <= 0:
        return None

    # Use precomputed OBV divergence if available
    if not pd.isna(obv_div) and abs(obv_div) > p["obv_divergence_threshold"]:
        pass  # Use the feature
    else:
        # Compute manually
        window = df.iloc[bar_idx - p["lookback"]:bar_idx + 1]
        prices = window["close"].values
        obv_vals = window.get("obv")
        if obv_vals is None or obv_vals.isna().any():
            return None
        obv_vals = obv_vals.values

        # Normalize both to [0, 1] range for comparison
        p_range = prices.max() - prices.min()
        o_range = obv_vals.max() - obv_vals.min()
        if p_range == 0 or o_range == 0:
            return None

        price_norm = (prices - prices.min()) / p_range
        obv_norm = (obv_vals - obv_vals.min()) / o_range

        # Price trend vs OBV trend (linear regression slope)
        x = np.arange(len(prices))
        price_slope = np.polyfit(x, price_norm, 1)[0]
        obv_slope = np.polyfit(x, obv_norm, 1)[0]

        obv_div = obv_slope - price_slope

        if abs(obv_div) < p["obv_divergence_threshold"]:
            return None

    close = current["close"]
    signal = None

    # Bullish: OBV rising while price falling (accumulation)
    if obv_div > p["obv_divergence_threshold"]:
        if p["rsi_confirm"] and (pd.isna(rsi) or rsi > p["rsi_overbought"]):
            return None

        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="long",
            strength=min(abs(obv_div) / 1.0, 1.0),
            strategy="obv_divergence",
            reason=f"Bullish OBV div={obv_div:.2f} (accumulation)",
            stop_loss=close - atr * p["stop_loss_atr_mult"],
            take_profit=close + atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    # Bearish: OBV falling while price rising (distribution)
    elif obv_div < -p["obv_divergence_threshold"]:
        if p["rsi_confirm"] and (pd.isna(rsi) or rsi < p["rsi_oversold"]):
            return None

        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="short",
            strength=min(abs(obv_div) / 1.0, 1.0),
            strategy="obv_divergence",
            reason=f"Bearish OBV div={obv_div:.2f} (distribution)",
            stop_loss=close + atr * p["stop_loss_atr_mult"],
            take_profit=close - atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    return signal
