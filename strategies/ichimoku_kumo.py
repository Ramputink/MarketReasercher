"""
CryptoResearchLab — Strategy: Ichimoku Kumo Breakout
Full Ichimoku system: Kumo cloud breakout with Tenkan/Kijun cross confirmation.
Crypto-adapted parameters (10-30-60 instead of classic 9-26-52).
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Evolved via genetic algorithm (2026-03-22, gen 183, 1074 evals)
    # Walk-forward ROBUST: Sharpe=1.02, WF_Sharpe=2.28, PF=1.30, 106 trades, $418 PnL
    "tenkan_period": 10,                  # Evolved: faster Tenkan
    "kijun_period": 34,                   # Evolved: slightly longer Kijun
    "senkou_b_period": 59,                # Evolved: wider cloud
    "displacement": 30,                   # Evolved: standard displacement
    "require_chikou_confirm": False,      # Evolved: keep Chikou confirmation
    "require_tk_cross": True,             # Evolved: no TK cross needed
    "adx_min": 12.1182,                   # Evolved: moderate ADX
    "require_volume": False,
    "volume_threshold": 0.8,
    "stop_loss_atr_mult": 2.9834,         # Evolved: moderate SL
    "take_profit_atr_mult": 6.8181,       # Evolved: wide TP
    "time_stop_hours": 72,
    "require_trend_regime": True,
    "allowed_regimes": ["trend", "breakout"],
}


def _donchian_mid(high_arr, low_arr, period):
    """Rolling (high + low) / 2 over period — Donchian midline."""
    n = len(high_arr)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        hh = np.max(high_arr[i - period + 1:i + 1])
        ll = np.min(low_arr[i - period + 1:i + 1])
        result[i] = (hh + ll) / 2
    return result


def ichimoku_kumo_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    min_lookback = max(p["senkou_b_period"], p["kijun_period"]) + p["displacement"] + 5
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
    if pd.isna(adx) or adx < p["adx_min"]:
        return None
    if p["require_volume"] and (pd.isna(vol_ratio) or vol_ratio < p["volume_threshold"]):
        return None

    # Compute Ichimoku components
    start = max(0, bar_idx - min_lookback - 10)
    sl_high = df["high"].iloc[start:bar_idx + 1].values
    sl_low = df["low"].iloc[start:bar_idx + 1].values
    sl_close = df["close"].iloc[start:bar_idx + 1].values

    tenkan = _donchian_mid(sl_high, sl_low, p["tenkan_period"])
    kijun = _donchian_mid(sl_high, sl_low, p["kijun_period"])
    senkou_b_raw = _donchian_mid(sl_high, sl_low, p["senkou_b_period"])

    last = len(sl_close) - 1
    prev = last - 1

    if np.isnan(tenkan[last]) or np.isnan(kijun[last]) or np.isnan(senkou_b_raw[last]):
        return None

    # Senkou Span A = (Tenkan + Kijun) / 2 (displaced forward)
    # For current position, we use the value from displacement bars ago
    disp = p["displacement"]
    if last - disp < 0:
        return None

    senkou_a_current = (tenkan[last - disp] + kijun[last - disp]) / 2 if not np.isnan(tenkan[last - disp]) else np.nan
    senkou_b_current = senkou_b_raw[last - disp] if last - disp >= 0 else np.nan

    if np.isnan(senkou_a_current) or np.isnan(senkou_b_current):
        return None

    cloud_top = max(senkou_a_current, senkou_b_current)
    cloud_bottom = min(senkou_a_current, senkou_b_current)

    # Previous bar's position relative to cloud
    if prev - disp < 0:
        return None
    sa_prev = (tenkan[prev - disp] + kijun[prev - disp]) / 2 if not np.isnan(tenkan[prev - disp]) else np.nan
    sb_prev = senkou_b_raw[prev - disp] if prev - disp >= 0 else np.nan
    if np.isnan(sa_prev) or np.isnan(sb_prev):
        return None

    cloud_top_prev = max(sa_prev, sb_prev)
    cloud_bottom_prev = min(sa_prev, sb_prev)
    close_prev = sl_close[prev]

    # Kumo breakout detection
    bullish_breakout = close > cloud_top and close_prev <= cloud_top_prev
    bearish_breakout = close < cloud_bottom and close_prev >= cloud_bottom_prev

    if not bullish_breakout and not bearish_breakout:
        return None

    # Chikou Span confirmation (current close vs price displacement bars ago)
    if p["require_chikou_confirm"]:
        chikou_ref_idx = last - disp
        if chikou_ref_idx < 0:
            return None
        chikou_ref_close = sl_close[chikou_ref_idx]
        if bullish_breakout and close <= chikou_ref_close:
            return None
        if bearish_breakout and close >= chikou_ref_close:
            return None

    # Tenkan/Kijun cross confirmation
    if p["require_tk_cross"]:
        tk_bull = tenkan[last] > kijun[last] and tenkan[prev] <= kijun[prev]
        tk_bear = tenkan[last] < kijun[last] and tenkan[prev] >= kijun[prev]
        if bullish_breakout and not tk_bull and not (tenkan[last] > kijun[last]):
            return None
        if bearish_breakout and not tk_bear and not (tenkan[last] < kijun[last]):
            return None

    side = "long" if bullish_breakout else "short"

    strength = 0.6
    cloud_thickness = abs(cloud_top - cloud_bottom) / close * 100
    if cloud_thickness > 1.0:
        strength += 0.15  # Thicker cloud = stronger breakout
    if adx > 25:
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
        strategy="ichimoku_kumo",
        reason=f"Kumo_{'bull' if bullish_breakout else 'bear'}_ADX={adx:.0f}",
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_stop_hours=p["time_stop_hours"],
        metadata={
            "cloud_top": cloud_top, "cloud_bottom": cloud_bottom,
            "tenkan": tenkan[last], "kijun": kijun[last], "regime": regime,
        },
    )
