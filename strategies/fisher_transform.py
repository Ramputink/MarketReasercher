"""
CryptoResearchLab — Strategy: Fisher Transform (Ehlers)
Converts prices to Gaussian distribution for sharp reversal signals.
Detects trend exhaustion — complementary to trend-following strategies.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Evolved via genetic algorithm (2026-03-22, gen 190, 1104 evals)
    # Walk-forward ROBUST: Sharpe=1.06, WF_Sharpe=2.19, PF=1.67, 34 trades, $152 PnL
    "period": 18,                         # Evolved: longer period for stability
    "signal_threshold": 1.982,            # Evolved: higher threshold = fewer but better signals
    "require_divergence": False,          # Evolved: divergence not needed
    "divergence_lookback": 14,
    "adx_filter": True,                   # Evolved: no ADX filter
    "adx_max": 35.3155,
    "require_volume": False,
    "volume_threshold": 0.9,
    "stop_loss_atr_mult": 2.1824,         # Evolved: tight SL for reversal trades
    "take_profit_atr_mult": 3.9579,       # Evolved: wide TP
    "time_stop_hours": 24,
    "require_trend_regime": True,
    "allowed_regimes": ["lateral", "mean_reversion", "trend"],
}


def _compute_fisher(high_arr, low_arr, period):
    """Compute Fisher Transform values."""
    n = len(high_arr)
    if n < period + 1:
        return None, None

    fisher = np.full(n, 0.0)
    fisher_signal = np.full(n, 0.0)
    value = 0.0

    for i in range(period - 1, n):
        hh = np.max(high_arr[i - period + 1:i + 1])
        ll = np.min(low_arr[i - period + 1:i + 1])

        if hh == ll:
            raw = 0.0
        else:
            hl2 = (high_arr[i] + low_arr[i]) / 2
            raw = 2.0 * ((hl2 - ll) / (hh - ll) - 0.5)

        # Clamp to avoid log(0)
        raw = max(-0.999, min(0.999, raw))

        # Smooth
        value = 0.5 * raw + 0.5 * value

        # Clamp again
        value = max(-0.999, min(0.999, value))

        # Fisher Transform
        fisher_signal[i] = fisher[i - 1] if i > 0 else 0
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value)) + 0.5 * (fisher[i - 1] if i > 0 else 0)

    return fisher, fisher_signal


def fisher_transform_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    min_lookback = max(p["period"] + 10, 60)
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

    # ADX filter — fisher works in various conditions, cap extreme trends
    if p["adx_filter"] and not pd.isna(adx) and adx > p["adx_max"]:
        return None

    if p["require_volume"] and (pd.isna(vol_ratio) or vol_ratio < p["volume_threshold"]):
        return None

    # Compute Fisher Transform
    lookback = min(bar_idx + 1, 150)
    start = bar_idx + 1 - lookback
    h = df["high"].iloc[start:bar_idx + 1].values
    l = df["low"].iloc[start:bar_idx + 1].values

    fisher, fisher_sig = _compute_fisher(h, l, p["period"])
    if fisher is None:
        return None

    last = len(fisher) - 1
    prev = last - 1
    if prev < p["period"]:
        return None

    ft = fisher[last]
    ft_prev = fisher[prev]
    ft_sig = fisher_sig[last]

    # Signal: Fisher crosses signal line at extreme levels
    threshold = p["signal_threshold"]
    side = None

    # Bearish reversal: Fisher was above threshold, now crossing down
    if ft_prev > threshold and ft < ft_prev and ft < ft_sig:
        side = "short"
    # Bullish reversal: Fisher was below -threshold, now crossing up
    elif ft_prev < -threshold and ft > ft_prev and ft > ft_sig:
        side = "long"

    if side is None:
        return None

    # Divergence confirmation (optional)
    if p["require_divergence"]:
        div_lb = p["divergence_lookback"]
        if last - div_lb < 0:
            return None
        c_arr = df["close"].iloc[start:bar_idx + 1].values
        if side == "long":
            # Bullish div: price makes lower low, Fisher makes higher low
            price_lower = c_arr[last] < np.min(c_arr[last - div_lb:last])
            fisher_higher = fisher[last] > np.min(fisher[last - div_lb:last])
            if not (price_lower and fisher_higher):
                return None
        else:
            price_higher = c_arr[last] > np.max(c_arr[last - div_lb:last])
            fisher_lower = fisher[last] < np.max(fisher[last - div_lb:last])
            if not (price_higher and fisher_lower):
                return None

    strength = 0.5
    strength += min(0.3, abs(ft) / 4.0)
    if abs(ft - ft_sig) > 0.5:
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
        strategy="fisher_transform",
        reason=f"Fisher={ft:.2f}_sig={ft_sig:.2f}_{'reversal' if side == 'long' else 'reversal'}",
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_stop_hours=p["time_stop_hours"],
        metadata={"fisher": ft, "fisher_signal": ft_sig, "regime": regime},
    )
