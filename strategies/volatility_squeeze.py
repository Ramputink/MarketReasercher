"""
CryptoResearchLab — Strategy: Volatility Squeeze (TTM Squeeze variant)
Keltner inside Bollinger = squeeze. Entry on release with momentum direction.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Evolved via genetic algorithm (2026-03-22, gen 472, 5375 evals)
    # Walk-forward ROBUST: Sharpe=3.31, WF_Sharpe=3.94, PF=3.72, 49 trades, $536 PnL
    "bb_period": 20,
    "bb_std": 2.0,
    "keltner_period": 20,
    "keltner_atr_mult": 1.1946,          # Evolved: tighter Keltner = more squeeze detection
    "squeeze_min_bars": 9,                # Evolved: require longer compression period
    "momentum_lookback": 13,              # Evolved: slightly longer momentum window
    "require_volume_surge": True,
    "volume_surge_threshold": 1.9133,     # Evolved: high bar for volume confirmation
    "adx_min": 0.0,
    "stop_loss_atr_mult": 3.7981,         # Evolved: wide stops for breakout moves
    "take_profit_atr_mult": 5.2517,       # Evolved: let winners run
    "time_stop_hours": 36,
    "require_regime": True,
    "allowed_regimes": ["breakout", "lateral"],
}

def volatility_squeeze_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    lookback = max(p["bb_period"], p["keltner_period"], p["momentum_lookback"]) + p["squeeze_min_bars"] + 5
    if bar_idx < lookback:
        return None
    if position is not None:
        return None
    if p["require_regime"] and regime not in p["allowed_regimes"]:
        return None

    current = df.iloc[bar_idx]
    atr = current.get("atr_14", 0)
    vol_ratio = current.get("volume_ratio", 1.0)
    adx = current.get("adx_14", 0)

    if pd.isna(atr) or atr <= 0:
        return None
    if not pd.isna(adx) and adx < p["adx_min"]:
        return None

    # Compute Bollinger Bands
    closes = df["close"].iloc[bar_idx - p["bb_period"]:bar_idx + 1]
    bb_mid = closes.rolling(p["bb_period"]).mean().iloc[-1]
    bb_std = closes.rolling(p["bb_period"]).std().iloc[-1]
    if pd.isna(bb_mid) or pd.isna(bb_std):
        return None
    bb_upper = bb_mid + p["bb_std"] * bb_std
    bb_lower = bb_mid - p["bb_std"] * bb_std

    # Keltner Channel
    ema_center = closes.ewm(span=p["keltner_period"], adjust=False).mean().iloc[-1]
    kc_upper = ema_center + atr * p["keltner_atr_mult"]
    kc_lower = ema_center - atr * p["keltner_atr_mult"]

    # Squeeze detection: BB inside Keltner
    is_squeezed_now = bb_lower > kc_lower and bb_upper < kc_upper

    # Check if squeeze just released
    squeeze_count = 0
    for j in range(1, p["squeeze_min_bars"] + 2):
        if bar_idx - j < 0:
            break
        past_bar = df.iloc[bar_idx - j]
        past_bb_bw_pct = past_bar.get("bb_bw_percentile", 50)
        if not pd.isna(past_bb_bw_pct) and past_bb_bw_pct < 20:
            squeeze_count += 1

    if squeeze_count < p["squeeze_min_bars"]:
        return None

    # Squeeze must be releasing (current bar NOT squeezed, or BB expanding)
    if is_squeezed_now:
        return None  # Still in squeeze, wait for release

    if p["require_volume_surge"] and vol_ratio < p["volume_surge_threshold"]:
        return None

    # Momentum direction using linear regression of last N closes
    mom_closes = df["close"].iloc[bar_idx - p["momentum_lookback"]:bar_idx + 1].values
    x = np.arange(len(mom_closes))
    slope = np.polyfit(x, mom_closes, 1)[0]

    close = current["close"]
    signal = None

    if slope > 0:
        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="long",
            strength=min(abs(slope) / atr * 10, 1.0),
            strategy="volatility_squeeze",
            reason=f"Squeeze release UP, mom_slope={slope:.5f}",
            stop_loss=close - atr * p["stop_loss_atr_mult"],
            take_profit=close + atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )
    elif slope < 0:
        signal = Signal(
            timestamp=int(current["timestamp"]),
            side="short",
            strength=min(abs(slope) / atr * 10, 1.0),
            strategy="volatility_squeeze",
            reason=f"Squeeze release DOWN, mom_slope={slope:.5f}",
            stop_loss=close + atr * p["stop_loss_atr_mult"],
            take_profit=close - atr * p["take_profit_atr_mult"],
            time_stop_hours=p["time_stop_hours"],
        )

    return signal
