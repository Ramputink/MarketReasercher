"""
CryptoResearchLab — Strategy: RSI Divergence
Detects bullish/bearish divergence between price and RSI.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    "rsi_period": 14,
    "divergence_lookback": 14,
    "min_price_delta_pct": 0.5,
    "min_rsi_delta": 5.0,
    "rsi_extreme_low": 35.0,
    "rsi_extreme_high": 65.0,
    "require_volume_decline": False,
    "volume_decline_ratio": 0.8,
    "adx_max": 35.0,
    "stop_loss_atr_mult": 2.5,
    "take_profit_atr_mult": 3.5,
    "time_stop_hours": 36,
    "require_regime": True,
    "allowed_regimes": ["lateral", "mean_reversion", "trend"],
}

def rsi_divergence_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    lookback = p["divergence_lookback"] + p["rsi_period"] + 5
    if bar_idx < lookback:
        return None
    if position is not None:
        return None
    if p["require_regime"] and regime not in p["allowed_regimes"]:
        return None

    current = df.iloc[bar_idx]
    atr = current.get("atr_14", 0)
    adx = current.get("adx_14", 0)
    rsi_now = current.get("rsi_14", 50)

    if pd.isna(atr) or atr <= 0 or pd.isna(rsi_now):
        return None
    if not pd.isna(adx) and adx > p["adx_max"]:
        return None

    window = df.iloc[bar_idx - p["divergence_lookback"]:bar_idx + 1]
    prices = window["close"].values
    rsi_col = "rsi_14" if "rsi_14" in window.columns else "rsi_7"
    rsi_vals = window[rsi_col].values

    if np.any(np.isnan(rsi_vals)):
        return None

    close = current["close"]
    signal = None

    # Bullish divergence: price making lower lows, RSI making higher lows
    price_start_low = np.min(prices[:len(prices)//2])
    price_end_low = np.min(prices[len(prices)//2:])
    rsi_start_low = np.min(rsi_vals[:len(rsi_vals)//2])
    rsi_end_low = np.min(rsi_vals[len(rsi_vals)//2:])

    if price_start_low == 0:
        return None
    price_delta_pct = (price_end_low - price_start_low) / price_start_low * 100
    rsi_delta = rsi_end_low - rsi_start_low

    if (price_delta_pct < -p["min_price_delta_pct"]
            and rsi_delta > p["min_rsi_delta"]
            and rsi_now < p["rsi_extreme_low"]):
        if p["require_volume_decline"]:
            vol_now = current.get("volume_ratio", 1.0)
            if pd.isna(vol_now) or vol_now >= p["volume_decline_ratio"]:
                return None  # Volume NOT declining → skip

        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="long",
            strength=min(abs(rsi_delta) / 20, 1.0),
            strategy="rsi_divergence",
            reason=f"Bullish div: price Δ={price_delta_pct:.1f}% RSI Δ=+{rsi_delta:.0f}",
            stop_loss=close - atr * p["stop_loss_atr_mult"],
            take_profit=close + atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    # Bearish divergence: price making higher highs, RSI making lower highs
    price_start_high = np.max(prices[:len(prices)//2])
    price_end_high = np.max(prices[len(prices)//2:])
    rsi_start_high = np.max(rsi_vals[:len(rsi_vals)//2])
    rsi_end_high = np.max(rsi_vals[len(rsi_vals)//2:])

    if price_start_high == 0:
        return None
    price_up_pct = (price_end_high - price_start_high) / price_start_high * 100
    rsi_down = rsi_start_high - rsi_end_high

    if (signal is None
            and price_up_pct > p["min_price_delta_pct"]
            and rsi_down > p["min_rsi_delta"]
            and rsi_now > p["rsi_extreme_high"]):
        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="short",
            strength=min(abs(rsi_down) / 20, 1.0),
            strategy="rsi_divergence",
            reason=f"Bearish div: price Δ=+{price_up_pct:.1f}% RSI Δ=-{rsi_down:.0f}",
            stop_loss=close + atr * p["stop_loss_atr_mult"],
            take_profit=close - atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    return signal
