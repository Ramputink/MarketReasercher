"""
CryptoResearchLab — Strategy: Williams %R + CCI Combo
Williams %R detects instant overbought/oversold, CCI confirms momentum.
Backtests show adding volume + BB Width filters turns -2% into +5% net.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Evolved via genetic algorithm (2026-03-22, gen ~500, 1061 evals)
    # Walk-forward ROBUST: Sharpe=1.59, WF_Sharpe=1.50, 19 trades, $109 PnL
    "wr_period": 10,                      # Evolved: shorter WR window
    "wr_oversold": -89.5301,              # Evolved: deep oversold
    "wr_overbought": -21.6745,            # Evolved: shallow overbought
    "cci_period": 15,                     # Evolved: shorter CCI
    "cci_long_threshold": -92.6157,       # Evolved: deep CCI long
    "cci_short_threshold": 148.9217,      # Evolved: very high CCI short
    "require_bb_squeeze": True,           # Evolved: BB squeeze required
    "bb_squeeze_percentile": 30.0,
    "adx_max": 35.0,
    "require_volume_spike": True,         # Evolved: volume spike required
    "volume_spike_threshold": 1.7638,     # Evolved: high volume bar
    "stop_loss_atr_mult": 1.3198,         # Evolved: tight SL
    "take_profit_atr_mult": 2.6337,       # Evolved: moderate TP
    "time_stop_hours": 24,
    "require_trend_regime": True,
    "allowed_regimes": ["lateral", "mean_reversion", "trend"],
}


def _williams_r(high_arr, low_arr, close_arr, period):
    """Compute Williams %R: (highest_high - close) / (highest_high - lowest_low) * -100."""
    n = len(close_arr)
    if n < period:
        return np.nan
    hh = np.max(high_arr[-period:])
    ll = np.min(low_arr[-period:])
    if hh == ll:
        return -50.0
    return ((hh - close_arr[-1]) / (hh - ll)) * -100


def _cci(high_arr, low_arr, close_arr, period):
    """Compute Commodity Channel Index."""
    n = len(close_arr)
    if n < period:
        return np.nan
    tp = (high_arr[-period:] + low_arr[-period:] + close_arr[-period:]) / 3
    tp_mean = np.mean(tp)
    mean_dev = np.mean(np.abs(tp - tp_mean))
    if mean_dev == 0:
        return 0.0
    return (tp[-1] - tp_mean) / (0.015 * mean_dev)


def williams_cci_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    min_lookback = max(p["wr_period"], p["cci_period"]) + 10
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
    bb_pct = current.get("bb_bw_percentile", 50)

    if pd.isna(atr_val) or atr_val <= 0:
        return None

    # ADX filter — mean reversion works better in low-trend conditions
    if not pd.isna(adx) and adx > p["adx_max"]:
        return None

    # Volume spike
    if p["require_volume_spike"]:
        if pd.isna(vol_ratio) or vol_ratio < p["volume_spike_threshold"]:
            return None

    # BB squeeze filter
    if p["require_bb_squeeze"]:
        if pd.isna(bb_pct) or bb_pct > p["bb_squeeze_percentile"]:
            return None

    # Compute Williams %R
    start = max(0, bar_idx - p["wr_period"] - 2)
    h = df["high"].iloc[start:bar_idx + 1].values
    l = df["low"].iloc[start:bar_idx + 1].values
    c = df["close"].iloc[start:bar_idx + 1].values
    wr = _williams_r(h, l, c, p["wr_period"])
    if np.isnan(wr):
        return None

    # Compute CCI
    start_cci = max(0, bar_idx - p["cci_period"] - 2)
    h_cci = df["high"].iloc[start_cci:bar_idx + 1].values
    l_cci = df["low"].iloc[start_cci:bar_idx + 1].values
    c_cci = df["close"].iloc[start_cci:bar_idx + 1].values
    cci = _cci(h_cci, l_cci, c_cci, p["cci_period"])
    if np.isnan(cci):
        return None

    # Signal logic: WR extreme + CCI confirmation
    side = None
    if wr <= p["wr_oversold"] and cci <= p["cci_long_threshold"]:
        side = "long"  # Both say oversold → buy
    elif wr >= p["wr_overbought"] and cci >= p["cci_short_threshold"]:
        side = "short"  # Both say overbought → sell

    if side is None:
        return None

    # Strength
    strength = 0.5
    if side == "long":
        strength += min(0.25, (abs(wr - p["wr_oversold"])) / 30)
    else:
        strength += min(0.25, (abs(wr - p["wr_overbought"])) / 30)
    if vol_ratio and vol_ratio > 1.5:
        strength += 0.1
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
        strategy="williams_cci",
        reason=f"WR={wr:.0f}_CCI={cci:.0f}",
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_stop_hours=p["time_stop_hours"],
        metadata={"williams_r": wr, "cci": cci, "regime": regime},
    )
