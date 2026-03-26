"""
CryptoResearchLab — Strategy: Trend Following (1-2 day horizon)
Uses BMSB as directional filter, Fibonacci pullbacks as entries.

AGENT-EDITABLE FILE: autoresearch modifies parameters and rules here.
"""
import numpy as np
import pandas as pd
from typing import Optional

from engine.backtester import Signal


# ═══════════════════════════════════════════════════════════════
# PARAMETERS — autoresearch modifies these
# ═══════════════════════════════════════════════════════════════

PARAMS = {
    # BMSB (Bull Market Support Band)
    "bmsb_sma_period": 20,
    "bmsb_ema_period": 21,
    "require_bmsb_bullish": True,
    # Trend confirmation — evolved (2026-03-22, gen ~400, 1112 evals)
    # Walk-forward: Sharpe=1.62, WF_Sharpe=1.96, 181 trades, $735 PnL
    "adx_threshold": 30.168,              # Evolved: strict ADX
    "trend_ema_period": 50,
    # Fibonacci pullback entry — evolved
    "fib_levels": [0.382, 0.5, 0.618],
    "fib_zone_tolerance_pct": 2.0956,     # Evolved: moderate tolerance
    "fib_lookback": 50,
    # Volume filter
    "min_volume_ratio": 0.9814,           # Evolved: lower volume floor
    "volume_ma_period": 20,
    # Momentum filter — evolved
    "rsi_lower_bound": 34,                # Evolved: standard oversold
    "rsi_upper_bound": 70,                # Evolved: wider upper band
    # Structure confirmation
    "require_higher_lows": False,
    "structure_lookback": 5,
    # Risk — evolved
    "stop_loss_atr_mult": 4.5456,         # Evolved: wide SL
    "take_profit_atr_mult": 5.5786,       # Evolved: very wide TP
    "time_stop_hours": 48,
    # Regime filter
    "require_trend_regime": True,
    "allowed_regimes": ["trend", "breakout"],
}


# ═══════════════════════════════════════════════════════════════
# STRATEGY LOGIC — autoresearch modifies this function
# ═══════════════════════════════════════════════════════════════

def trend_following_strategy(
    df: pd.DataFrame,
    bar_idx: int,
    position: Optional[object] = None,
    regime: str = "unknown",
) -> Optional[Signal]:
    """
    Trend Following Strategy (1-2 day horizon).

    Entry logic:
    1. BMSB confirms bullish/bearish trend
    2. ADX confirms trend strength
    3. Price pulls back to Fibonacci zone
    4. Volume supports the move
    5. Structure confirms (higher lows for long, lower highs for short)

    Exit: TP/SL/time stop.
    """
    p = PARAMS

    min_lookback = max(
        p["bmsb_sma_period"], p["bmsb_ema_period"],
        p["trend_ema_period"], p["fib_lookback"], 60
    )
    if bar_idx < min_lookback:
        return None

    current = df.iloc[bar_idx]

    if position is not None:
        return None

    # ─── Regime filter ───
    if p["require_trend_regime"] and regime not in p["allowed_regimes"]:
        return None

    close = float(current["close"])

    # ─── BMSB trend filter ───
    bmsb_bullish = current.get("bmsb_bullish", False)
    if pd.isna(bmsb_bullish):
        bmsb_bullish = False

    bmsb_sma = current.get("bmsb_sma", None)
    bmsb_ema = current.get("bmsb_ema", None)

    if bmsb_sma is None or bmsb_ema is None or pd.isna(bmsb_sma) or pd.isna(bmsb_ema):
        return None

    # Determine trend direction
    trend_ema = df["close"].iloc[max(0, bar_idx - p["trend_ema_period"]):bar_idx + 1].ewm(
        span=p["trend_ema_period"], adjust=False).mean().iloc[-1]
    is_uptrend = close > trend_ema and bmsb_bullish
    is_downtrend = close < trend_ema and not bmsb_bullish

    if not is_uptrend and not is_downtrend:
        return None

    # ─── ADX trend strength ───
    adx_val = current.get("adx_14", 0)
    if pd.isna(adx_val) or adx_val < p["adx_threshold"]:
        return None

    # ─── Fibonacci pullback zone ───
    fib_window = df.iloc[max(0, bar_idx - p["fib_lookback"]):bar_idx + 1]
    swing_high = float(fib_window["high"].max())
    swing_low = float(fib_window["low"].min())
    fib_range = swing_high - swing_low

    if fib_range <= 0:
        return None

    in_fib_zone = False
    matched_fib = None

    for fib_level in p["fib_levels"]:
        if is_uptrend:
            # Price retraces from high: fib level = swing_high - fib * range
            fib_price = swing_high - fib_level * fib_range
            tolerance = close * p["fib_zone_tolerance_pct"] / 100
            if abs(close - fib_price) <= tolerance:
                in_fib_zone = True
                matched_fib = fib_level
                break
        elif is_downtrend:
            # Price retraces from low: fib level = swing_low + fib * range
            fib_price = swing_low + fib_level * fib_range
            tolerance = close * p["fib_zone_tolerance_pct"] / 100
            if abs(close - fib_price) <= tolerance:
                in_fib_zone = True
                matched_fib = fib_level
                break

    if not in_fib_zone:
        return None

    # ─── Volume filter ───
    vol_ratio = current.get("volume_ratio", 1.0)
    if pd.isna(vol_ratio):
        vol_ratio = 1.0
    if vol_ratio < p["min_volume_ratio"]:
        return None

    # ─── RSI filter ───
    rsi_val = current.get("rsi_14", 50)
    if not pd.isna(rsi_val):
        if is_uptrend and rsi_val < p["rsi_lower_bound"]:
            return None  # Too oversold for trend following
        if is_downtrend and rsi_val > p["rsi_upper_bound"]:
            return None  # Too overbought for short trend following

    # ─── Structure confirmation ───
    if p["require_higher_lows"]:
        struct_window = df.iloc[max(0, bar_idx - p["structure_lookback"]):bar_idx + 1]
        if is_uptrend:
            # Check for higher lows
            lows = struct_window["low"].values
            if len(lows) >= 3:
                # At least last 3 lows should be ascending
                recent_lows = [lows[i] for i in range(len(lows)) if i == 0 or lows[i] <= lows[i-1] * 1.01]
                if len(recent_lows) >= 2 and recent_lows[-1] < recent_lows[0]:
                    return None  # Lower lows — not a healthy uptrend
        elif is_downtrend:
            highs = struct_window["high"].values
            if len(highs) >= 3:
                recent_highs = [highs[i] for i in range(len(highs)) if i == 0 or highs[i] >= highs[i-1] * 0.99]
                if len(recent_highs) >= 2 and recent_highs[-1] > recent_highs[0]:
                    return None  # Higher highs — not a healthy downtrend

    # ─── Generate signal ───
    side = "long" if is_uptrend else "short"
    atr_val = float(current.get("atr_14", close * 0.02))

    # Strength: stronger at deeper fib levels + stronger ADX
    strength = 0.5
    if matched_fib and matched_fib >= 0.5:
        strength += 0.15  # Deeper pullback = more conviction
    if adx_val > 35:
        strength += 0.15
    if vol_ratio > 1.5:
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
        strategy="trend_following",
        reason=f"BMSB_{'bull' if is_uptrend else 'bear'}_fib={matched_fib}_ADX={adx_val:.0f}",
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_stop_hours=p["time_stop_hours"],
        metadata={
            "fib_level": matched_fib,
            "adx": adx_val,
            "bmsb_bullish": bool(bmsb_bullish),
            "regime": regime,
        },
    )
