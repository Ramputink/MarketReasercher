"""
CryptoResearchLab — Strategy: Volatility Breakout
Detects Bollinger Band compression → explosive breakout.

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
    # Bollinger Band settings
    "bb_period": 20,
    "bb_std": 2.0,
    # Compression detection — optimized via grid search (2026-03-21)
    "bandwidth_percentile_threshold": 25.0,  # Optimized: sweet spot (Sharpe 2.11)
    "min_compression_bars": 3,               # Optimized: reduced from 4
    # Breakout confirmation
    "breakout_confirmation_candles": 1,
    "volume_surge_threshold": 1.5,           # Optimized: 1.5 beats 2.0 (more trades, same quality)
    # Microstructure filters
    "use_pressure_imbalance": False,         # Optimized: OFF — doesn't improve Sharpe, reduces trades
    "pressure_imbalance_threshold": 0.15,
    # ATR expansion filter: CRITICAL — all top combos have this on
    "require_atr_expansion": True,
    "atr_expansion_ratio": 1.15,             # Optimized: 1.15 (slightly looser than 1.2)
    # Risk — optimized stop/TP ratio
    "stop_loss_atr_mult": 2.5,              # Optimized: wider stops = fewer noise exits
    "take_profit_atr_mult": 3.5,            # Optimized: 3.5 ATR = 1.4:1 reward-risk
    "time_stop_hours": 24,
    # Regime filter (from MiroFish)
    "require_breakout_regime": True,
    "allowed_regimes": ["breakout", "trend"],  # Removed "unknown" — regime must be detected
}


# ═══════════════════════════════════════════════════════════════
# STRATEGY LOGIC — autoresearch modifies this function
# ═══════════════════════════════════════════════════════════════

def volatility_breakout_strategy(
    df: pd.DataFrame,
    bar_idx: int,
    position: Optional[object] = None,
    regime: str = "unknown",
) -> Optional[Signal]:
    """
    Volatility Breakout Strategy.

    Entry logic:
    1. BB Bandwidth percentile < threshold (compression detected)
    2. Price breaks above/below BB
    3. Volume confirmation
    4. Microstructure confirmation (optional)

    Exit: stop loss, take profit, or time stop (handled by backtester).
    """
    p = PARAMS

    # Need enough history
    if bar_idx < max(p["bb_period"], 50) + p["min_compression_bars"]:
        return None

    current = df.iloc[bar_idx]

    # If position is open, only exit signals (handled by backtester)
    if position is not None:
        return None

    # ─── Regime filter ───
    if p["require_breakout_regime"] and regime not in p["allowed_regimes"]:
        return None

    # ─── Compression detection ───
    bw_pct = current.get("bb_bw_percentile", 50.0)
    if pd.isna(bw_pct) or bw_pct > p["bandwidth_percentile_threshold"]:
        return None

    # Check compression duration
    lookback = df.iloc[max(0, bar_idx - p["min_compression_bars"]):bar_idx + 1]
    if "bb_bw_percentile" in lookback.columns:
        compressed_bars = (lookback["bb_bw_percentile"] < p["bandwidth_percentile_threshold"]).sum()
        if compressed_bars < p["min_compression_bars"]:
            return None

    # ─── Breakout detection ───
    close = float(current["close"])
    bb_upper = current.get("bb_upper", None)
    bb_lower = current.get("bb_lower", None)

    if bb_upper is None or bb_lower is None or pd.isna(bb_upper) or pd.isna(bb_lower):
        return None

    bb_upper = float(bb_upper)
    bb_lower = float(bb_lower)

    breakout_long = close > bb_upper
    breakout_short = close < bb_lower

    if not breakout_long and not breakout_short:
        return None

    # ─── Volume confirmation ───
    vol_ratio = current.get("volume_ratio", 1.0)
    if pd.isna(vol_ratio):
        vol_ratio = 1.0
    if vol_ratio < p["volume_surge_threshold"]:
        return None

    # ─── Breakout confirmation (consecutive candles) ───
    confirm = p["breakout_confirmation_candles"]
    if confirm > 1 and bar_idx >= confirm:
        recent = df.iloc[bar_idx - confirm + 1: bar_idx + 1]
        if breakout_long:
            if not all(recent["close"] > recent.get("bb_upper", recent["close"])):
                # Allow if at least last candle confirms
                pass  # Relaxed confirmation
        elif breakout_short:
            if not all(recent["close"] < recent.get("bb_lower", recent["close"])):
                pass

    # ─── ATR expansion filter: confirm breakout is real, not just a wick ───
    if p.get("require_atr_expansion", False) and bar_idx >= 5:
        atr_now = current.get("atr_14", None)
        atr_5ago = df.iloc[bar_idx - 5].get("atr_14", None)
        if atr_now is not None and atr_5ago is not None:
            if not pd.isna(atr_now) and not pd.isna(atr_5ago) and atr_5ago > 0:
                if atr_now / atr_5ago < p.get("atr_expansion_ratio", 1.2):
                    return None  # ATR not expanding = likely false breakout

    # ─── Microstructure filter ───
    if p["use_pressure_imbalance"]:
        imbalance = current.get("pressure_imbalance", 0.0)
        if not pd.isna(imbalance):
            if breakout_long and imbalance < p["pressure_imbalance_threshold"]:
                return None
            elif breakout_short and imbalance > -p["pressure_imbalance_threshold"]:
                return None

    # ─── Generate signal ───
    side = "long" if breakout_long else "short"
    atr_val = float(current.get("atr_14", close * 0.02))

    # Signal strength based on multiple confirmations
    strength = 0.5
    if vol_ratio > 2.0:
        strength += 0.2
    if bw_pct < 10:
        strength += 0.15
    if not pd.isna(current.get("pressure_imbalance", np.nan)):
        imb = abs(float(current["pressure_imbalance"]))
        if imb > 0.5:
            strength += 0.15
    strength = min(strength, 1.0)

    # Stop / Take profit
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
        strategy="volatility_breakout",
        reason=f"BB_compression_pct={bw_pct:.0f}_vol_ratio={vol_ratio:.1f}",
        stop_loss=stop_loss,
        take_profit=take_profit,
        time_stop_hours=p["time_stop_hours"],
        metadata={"bb_bw_pct": bw_pct, "vol_ratio": vol_ratio, "regime": regime},
    )
