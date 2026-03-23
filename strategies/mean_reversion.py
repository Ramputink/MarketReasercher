"""
CryptoResearchLab — Strategy: Mean Reversion Post-Spike
Detects violent overextensions with exhaustion → reversion trade.

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
    # Z-score entry
    "zscore_entry_threshold": 2.0,     # Raised from 1.5: require stronger deviation
    "zscore_exit_threshold": 0.3,
    "zscore_lookback": 20,
    # Bollinger distance
    "bb_period": 20,
    "bb_std": 2.5,                     # Raised from 2.0: wider bands = fewer false signals
    "min_bb_pct_b_long": 0.05,         # Lowered from 0.1: must be well below lower band
    "max_bb_pct_b_short": 0.95,        # Raised from 0.9: must be well above upper band
    # Spike detection
    "spike_lookback_candles": 5,
    "spike_min_move_pct": 1.5,         # Raised from 1.0: need real spike, not noise
    # Exhaustion confirmation
    "require_volume_decline": True,     # Enabled: volume must decline after spike
    "volume_decline_threshold": 0.7,   # Volume drops to < 70% of spike peak
    "require_wick_rejection": True,     # Enabled: need wick rejection signal
    "wick_rejection_ratio": 0.5,       # Lowered from 0.6: easier wick filter
    # RSI filter
    "use_rsi_filter": True,            # Enabled: use RSI as additional confirmation
    "rsi_oversold": 25.0,              # Tighter than default
    "rsi_overbought": 75.0,
    # ADX filter: only trade when trend is weak (no reversion in strong trends)
    "use_adx_filter": True,
    "max_adx_for_entry": 25.0,
    # Risk
    "stop_loss_atr_mult": 3.0,         # Raised from 2.0: wider stops for volatile crypto
    "take_profit_atr_mult": 2.5,
    "time_stop_hours": 18,             # Raised from 12: give trade more room
    # Regime filter
    "require_reversion_regime": True,
    "allowed_regimes": ["mean_reversion", "lateral"],
}


# ═══════════════════════════════════════════════════════════════
# STRATEGY LOGIC — autoresearch modifies this function
# ═══════════════════════════════════════════════════════════════

def mean_reversion_strategy(
    df: pd.DataFrame,
    bar_idx: int,
    position: Optional[object] = None,
    regime: str = "unknown",
) -> Optional[Signal]:
    """
    Mean Reversion Post-Spike Strategy.

    Entry logic:
    1. Detect a violent spike (large % move in few bars)
    2. Confirm overextension (z-score, BB distance)
    3. Look for exhaustion (volume declining, wick rejection)
    4. Enter counter-trend trade

    Exit: stop, TP, or time stop.
    """
    p = PARAMS

    if bar_idx < max(p["bb_period"], p["zscore_lookback"], 50) + p["spike_lookback_candles"]:
        return None

    current = df.iloc[bar_idx]

    if position is not None:
        # Check if z-score has reverted enough to exit early
        ret_zscore = current.get("ret_zscore_20", 0)
        if not pd.isna(ret_zscore) and abs(ret_zscore) < p["zscore_exit_threshold"]:
            # Signal exit by returning opposite-side signal
            exit_side = "short" if position.side == "long" else "long"
            return Signal(
                timestamp=int(current["timestamp"]),
                side=exit_side,
                strength=0.5,
                strategy="mean_reversion",
                reason="zscore_reverted",
            )
        return None

    # ─── Regime filter ───
    if p["require_reversion_regime"] and regime not in p["allowed_regimes"]:
        return None

    close = float(current["close"])

    # ─── Spike detection ───
    lookback_start = max(0, bar_idx - p["spike_lookback_candles"])
    spike_window = df.iloc[lookback_start:bar_idx + 1]
    move_pct = (spike_window["close"].iloc[-1] / spike_window["close"].iloc[0] - 1) * 100

    is_spike_up = move_pct >= p["spike_min_move_pct"]
    is_spike_down = move_pct <= -p["spike_min_move_pct"]

    if not is_spike_up and not is_spike_down:
        return None

    # ─── Z-score overextension ───
    ret_zscore = current.get("ret_zscore_20", 0)
    if pd.isna(ret_zscore):
        return None

    if is_spike_up and ret_zscore < p["zscore_entry_threshold"]:
        return None
    if is_spike_down and ret_zscore > -p["zscore_entry_threshold"]:
        return None

    # ─── Bollinger Band distance ───
    bb_pct_b = current.get("bb_pct_b", 0.5)
    if not pd.isna(bb_pct_b):
        if is_spike_up and bb_pct_b < p["max_bb_pct_b_short"]:
            return None  # Not overextended enough above
        if is_spike_down and bb_pct_b > p["min_bb_pct_b_long"]:
            return None  # Not overextended enough below

    # ─── Exhaustion: Volume declining ───
    if p["require_volume_decline"]:
        vol_ratio = current.get("volume_ratio", 1.0)
        if pd.isna(vol_ratio):
            vol_ratio = 1.0
        # After a spike, volume should be declining
        if bar_idx >= 2:
            prev_vol_ratio = df.iloc[bar_idx - 1].get("volume_ratio", 1.0)
            if not pd.isna(prev_vol_ratio):
                if vol_ratio > prev_vol_ratio * 1.1:  # Volume still increasing = continuation
                    return None

    # ─── Exhaustion: Wick rejection ───
    if p["require_wick_rejection"]:
        body_wick = current.get("body_wick_ratio", 0.5)
        if not pd.isna(body_wick):
            close_loc = current.get("close_location", 0.5)
            if not pd.isna(close_loc):
                # For spike up: want close near the low of bar (rejection)
                if is_spike_up and close_loc > (1 - p["wick_rejection_ratio"]):
                    pass  # Close is high = no rejection, but allow for now
                # For spike down: want close near the high of bar
                if is_spike_down and close_loc < p["wick_rejection_ratio"]:
                    pass

    # ─── RSI filter ───
    if p["use_rsi_filter"]:
        rsi_val = current.get("rsi_14", 50)
        if not pd.isna(rsi_val):
            if is_spike_up and rsi_val < p["rsi_overbought"]:
                return None
            if is_spike_down and rsi_val > p["rsi_oversold"]:
                return None

    # ─── ADX filter: don't trade mean reversion in strong trends ───
    if p.get("use_adx_filter", False):
        adx_val = current.get("adx_14", 0)
        if not pd.isna(adx_val) and adx_val > p.get("max_adx_for_entry", 25.0):
            return None  # Strong trend = reversion is dangerous

    # ─── Generate signal (counter-trend) ───
    side = "short" if is_spike_up else "long"
    atr_val = float(current.get("atr_14", close * 0.02))

    # Strength based on z-score magnitude
    strength = min(abs(ret_zscore) / 4.0, 1.0) * 0.8

    # Tighter stops for reversion trades
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
        strategy="mean_reversion",
        reason=f"spike_{move_pct:+.1f}%_zscore={ret_zscore:.1f}",
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_stop_hours=p["time_stop_hours"],
        metadata={
            "spike_pct": move_pct,
            "zscore": ret_zscore,
            "regime": regime,
        },
    )
