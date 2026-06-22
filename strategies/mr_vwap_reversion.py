"""
CryptoResearchLab — Strategy: Intraday Micro-Volatility Mean Reversion (LONG-ONLY)

Thesis (Shleifer & Vishny 1997, "Limits to Arbitrage"): short-horizon price
dislocations persist because arbitrage capital is constrained. On 1h crypto bars,
sharp dips BELOW the volume-weighted average price (VWAP) during NON-trending phases
tend to revert. We buy the dislocation and target reversion to the mean.

Research grounding (web, 2026): "When price extends 2+ std from VWAP and the market
is not trending hard, it often reverts. With filters (ADX, regime) 55-65% win rate;
without filters reversion fails badly on trend days."

LONG-ONLY by construction. Micro-volatility is captured by sizing the entry band with
realized volatility (Garman-Klass), so the trigger adapts to current vol.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    # Entry: how many ATRs below VWAP before we call it a dislocation. (gk_vol is
    # ANNUALIZED so it is unusable as a per-bar scale; ATR is the correct price-unit
    # measure of micro-volatility distance.)
    "dislocation_atr": 0.8,          # (vwap - close)/atr_14 >= this
    "bb_pct_b_max": 0.20,            # must also be near/under lower Bollinger band
    "rsi_oversold": 40.0,            # rsi_7 oversold confirmation
    "adx_max": 30.0,                 # only when NOT trending hard (ranging market)
    "require_recovery": True,        # close in upper half of bar = buyers stepping in
    "close_location_min": 0.40,
    "min_realized_vol": 0.0005,      # skip dead-flat bars (no micro-vol to exploit)
    "stop_loss_atr_mult": 1.6,       # tight stop below the dislocation
    "take_profit_atr_mult": 2.2,     # target reversion (modest, mean-reversion)
    "time_stop_hours": 12,           # intraday: exit if no reversion within 12h
    "allowed_regimes": ["mean_reversion", "lateral", "unknown"],
    "require_regime": True,
}


def mr_vwap_reversion_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    if bar_idx < 50 or position is not None:
        return None
    if p["require_regime"] and regime not in p["allowed_regimes"]:
        return None

    cur = df.iloc[bar_idx]
    close = float(cur["close"])
    vwap = float(cur.get("vwap_20", np.nan))
    gkv = float(cur.get("gk_vol", np.nan))
    atr = float(cur.get("atr_14", 0.0))
    adx = float(cur.get("adx_14", 0.0))
    rsi = float(cur.get("rsi_7", 50.0))
    bb_b = float(cur.get("bb_pct_b", 0.5))
    cloc = float(cur.get("close_location", 0.5))
    rvol = float(cur.get("realized_vol_20", 0.0))

    if not np.isfinite(vwap) or vwap <= 0 or atr <= 0 or not np.isfinite(gkv) or gkv <= 0:
        return None
    if rvol < p["min_realized_vol"]:
        return None
    if adx > p["adx_max"]:          # trending too hard -> reversion unsafe
        return None

    # dislocation below VWAP in ATR units (this is the "micro-volatility" trigger)
    z = (vwap - close) / atr
    if z < p["dislocation_atr"]:
        return None
    if bb_b > p["bb_pct_b_max"]:    # must be at/under lower band
        return None
    if rsi > p["rsi_oversold"]:
        return None
    if p["require_recovery"] and cloc < p["close_location_min"]:
        return None

    strength = float(min(z / (p["dislocation_atr"] * 2.0), 1.0))
    return Signal(
        timestamp=int(cur["timestamp"]),
        side="long",
        strength=max(strength, 0.2),
        strategy="mr_vwap_reversion",
        reason=f"Dislocation z={z:.2f} below VWAP, rsi7={rsi:.0f}, adx={adx:.0f} (ranging)",
        stop_loss=close - atr * p["stop_loss_atr_mult"],
        take_profit=close + atr * p["take_profit_atr_mult"],
        time_stop_hours=p["time_stop_hours"],
    )
