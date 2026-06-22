"""
CryptoResearchLab — Strategy: Bull Market Support Band (LONG-ONLY, long-term)

The REAL Bull Market Support Band: 20-WEEK SMA + 21-WEEK EMA (not the 20/21-BAR
version that the legacy feature builder mislabels as "BMSB"). On daily data the
weekly band is approximated with 140-day SMA (20w x 7) and 147-day EMA (21w x 7);
a daily harness can instead pass a true weekly-resampled band via the same columns.

Thesis: in crypto bull cycles price rides ABOVE the band and bounces off it; the band
is dynamic support. Sitting in cash when price is below the band avoids the deep bear
drawdowns that wreck buy-and-hold. Research (web, 2026): an EMA Bitcoin/cash strategy
returned ~126% annualized at Sharpe ~1.9 (2012-2023) — long-or-cash trend following is
the strongest documented crypto edge for the long horizon.

Reads precomputed columns `bmsb_sma_real` and `bmsb_ema_real` (built by the daily
harness). Falls back to the legacy `bmsb_sma`/`bmsb_ema` if the real ones are absent.
LONG-ONLY: emits long entries; exits are signalled by flipping side to flat via stop.
"""
import numpy as np
import pandas as pd
from typing import Optional
from engine.backtester import Signal

PARAMS = {
    "entry_buffer": 0.00,        # close must be this fraction above band top to enter
    "exit_buffer": 0.03,         # exit when close falls this fraction below band bottom
    "require_band_bullish": False,  # optionally require EMA>SMA (band turning up)
    "confirm_bars": 2,           # bars price must hold above band before entry
    "stop_loss_pct": 0.18,       # wide disaster stop (band exit is the real exit)
    "take_profit_pct": 5.0,      # effectively off; we ride trends, exit on band break
    "time_stop_hours": 24 * 400, # long horizon: no time stop in practice
    # Volatility targeting (Barroso & Santa-Clara 2015): size the ENTRY inversely to
    # recent realized vol so high-vol regimes get a smaller position (lower drawdown).
    # Applied via Signal.strength, which the engine multiplies into position size.
    "enable_vol_target": False,  # off by default (preserves flat-sizing behavior)
    "vol_target_annual": 0.60,   # target ~60% annualized vol (crypto is high-vol)
    "vol_lookback": 30,          # bars (days on daily data) for realized-vol estimate
    "strength_floor": 0.20,      # never size below 20% of full
}


def _vol_target_strength(df, bar_idx, p):
    """clip(vol_target / realized_vol_annualized, floor, 1.0). Causal: uses only
    closes up to bar_idx. Returns 1.0 if disabled or vol cannot be estimated."""
    if not p.get("enable_vol_target", False):
        return 1.0
    lb = int(p.get("vol_lookback", 30))
    if bar_idx < lb + 1:
        return 1.0
    closes = df["close"].iloc[bar_idx - lb:bar_idx + 1].to_numpy(dtype=float)
    rets = closes[1:] / closes[:-1] - 1.0
    sd = float(np.std(rets))
    if sd <= 0:
        return 1.0
    ann = sd * np.sqrt(365.0)
    s = p["vol_target_annual"] / ann
    return float(min(max(s, p["strength_floor"]), 1.0))


def _band(cur):
    sma = cur.get("bmsb_sma_real", cur.get("bmsb_sma", np.nan))
    ema = cur.get("bmsb_ema_real", cur.get("bmsb_ema", np.nan))
    return float(sma), float(ema)


def bmsb_long_strategy(
    df: pd.DataFrame, bar_idx: int,
    position: Optional[object] = None, regime: str = "unknown",
) -> Optional[Signal]:
    p = PARAMS
    if bar_idx < 5:
        return None

    cur = df.iloc[bar_idx]
    close = float(cur["close"])
    sma, ema = _band(cur)
    if not (np.isfinite(sma) and np.isfinite(ema)) or sma <= 0 or ema <= 0:
        return None
    band_top = max(sma, ema)
    band_bot = min(sma, ema)

    # ---- exit logic (when in a position): decisive close below the band ----
    if position is not None:
        if close < band_bot * (1.0 - p["exit_buffer"]):
            # flip to flat by emitting opposite side -> engine closes on signal_exit
            return Signal(timestamp=int(cur["timestamp"]), side="short",
                          strength=0.0, strategy="bmsb_long",
                          reason=f"Close {close:.2f} below band {band_bot:.2f} -> to cash")
        return None

    # ---- entry logic: price holding above the band ----
    if p["require_band_bullish"] and ema <= sma:
        return None
    if close <= band_top * (1.0 + p["entry_buffer"]):
        return None

    # confirmation: price held above band_top for confirm_bars
    cb = p["confirm_bars"]
    if cb > 0 and bar_idx >= cb:
        recent = df.iloc[bar_idx - cb:bar_idx + 1]
        st = recent.get("bmsb_sma_real", recent.get("bmsb_sma"))
        et = recent.get("bmsb_ema_real", recent.get("bmsb_ema"))
        tops = np.maximum(st.to_numpy(dtype=float), et.to_numpy(dtype=float))
        if not np.all(recent["close"].to_numpy(dtype=float) > tops):
            return None

    strength = _vol_target_strength(df, bar_idx, p)
    return Signal(
        timestamp=int(cur["timestamp"]),
        side="long",
        strength=strength,
        strategy="bmsb_long",
        reason=f"Close {close:.2f} above BMSB top {band_top:.2f} (bull regime, strength={strength:.2f})",
        stop_loss=close * (1.0 - p["stop_loss_pct"]),
        take_profit=close * (1.0 + p["take_profit_pct"]),
        time_stop_hours=p["time_stop_hours"],
    )
