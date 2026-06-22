"""
CryptoResearchLab — Strategy: Volatility-Expansion Momentum (LONG-ONLY)

Thesis (time-series momentum, Moskowitz/Ooi/Pedersen + crypto factor literature):
trends are born when volatility expands out of a contraction. We go long when:
  (1) realized volatility is EXPANDING (short-window vol > long-window vol), AND
  (2) price breaks above its recent range (momentum), AND
  (3) short-horizon return is positive, AND
  (4) volume confirms.

Research grounding (web, 2026): crypto time-series momentum (28d lookback / 5d hold)
reached Sharpe ~1.51 vs market 0.84; "momentum with volatility filters improved to
~1.2 Sharpe". Volatility targeting (Barroso & Santa-Clara 2015) is applied at the
portfolio/sizing layer, not here.

LONG-ONLY. Captures intraday micro-volatility breakouts; wide TP lets winners run.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    "vol_ratio_min": 1.15,          # short vol / long vol -> expansion regime
    "breakout_lookback": 24,        # bars; break above this rolling high = momentum
    "breakout_buffer_atr": 0.05,    # require close a touch above the high
    "mom_lookback_col": "ret_12",   # positive 12h return required
    "mom_min": 0.0,
    "volume_ratio_min": 1.2,        # volume confirmation
    "adx_min": 18.0,                # some directional strength
    "stop_loss_atr_mult": 2.0,
    "take_profit_atr_mult": 6.0,    # let momentum run
    "time_stop_hours": 36,
    "allowed_regimes": ["breakout", "trend", "unknown"],
    "require_regime": True,
}


def vol_expansion_long_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    lookback = p["breakout_lookback"] + 5
    if bar_idx < max(lookback, 50) or position is not None:
        return None
    if p["require_regime"] and regime not in p["allowed_regimes"]:
        return None

    cur = df.iloc[bar_idx]
    close = float(cur["close"])
    atr = float(cur.get("atr_14", 0.0))
    adx = float(cur.get("adx_14", 0.0))
    vol_ratio = float(cur.get("vol_ratio", 1.0))
    vol_conf = float(cur.get("volume_ratio", 1.0))
    mom = float(cur.get(p["mom_lookback_col"], 0.0))

    if atr <= 0 or not np.isfinite(vol_ratio):
        return None
    if vol_ratio < p["vol_ratio_min"]:        # need volatility EXPANSION
        return None
    if adx < p["adx_min"]:
        return None
    if mom <= p["mom_min"]:                    # positive momentum only
        return None
    if vol_conf < p["volume_ratio_min"]:
        return None

    window = df.iloc[bar_idx - p["breakout_lookback"]:bar_idx]
    roll_high = float(window["high"].max())
    if close <= roll_high + p["breakout_buffer_atr"] * atr:
        return None                            # no breakout

    strength = float(min((close - roll_high) / atr + (vol_ratio - 1.0), 1.0))
    return Signal(
        timestamp=int(cur["timestamp"]),
        side="long",
        strength=max(strength, 0.2),
        strategy="vol_expansion_long",
        reason=f"Vol expansion {vol_ratio:.2f}x + breakout>{roll_high:.4f}, mom={mom:.3f}",
        stop_loss=close - atr * p["stop_loss_atr_mult"],
        take_profit=close + atr * p["take_profit_atr_mult"],
        time_stop_hours=p["time_stop_hours"],
    )
