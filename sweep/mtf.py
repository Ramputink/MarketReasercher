"""
sweep.mtf — multi-timeframe confirmation (no look-ahead)
========================================================

A directional signal on the BASE timeframe is only honoured if the HIGHER
timeframe(s) do not contradict its direction. The hard part is doing this without
leaking the future: at base bar i (which a strategy decides on at bar i's CLOSE),
we may only consult higher-TF bars that have THEMSELVES fully closed by that
moment — never the still-forming higher-TF bar.

Mechanics
---------
- A bar's `timestamp` is its OPEN time; it closes at `timestamp + tf_ms`.
- For base bar i, its decision time is `close_i = ts_i + base_tf_ms`.
- The confirmation bar is the most recent higher-TF bar j with
  `close_j = ts_j + higher_tf_ms <= close_i`. Strictly `<=`, so a higher-TF bar
  is usable only once it has closed.
- The higher-TF DIRECTION at bar j is computed purely from data up to bar j's
  close (EMA trend), so the whole chain is backward-looking.

The result is a per-base-bar integer direction in {-1, 0, +1} that can be merged
into the base DataFrame as a plain column and read like any other feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_TF_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}


def tf_ms(tf: str) -> int:
    if tf not in _TF_MS:
        raise KeyError(f"unknown timeframe {tf!r}")
    return _TF_MS[tf]


def higher_tf_direction(higher_df: pd.DataFrame, ema_period: int = 50) -> np.ndarray:
    """
    Backward-looking trend direction for each higher-TF bar, knowable at that
    bar's close: +1 if close>EMA and EMA rising, -1 if close<EMA and EMA falling,
    else 0 (no clear trend). Uses only past/current bars.
    """
    close = higher_df["close"].astype(float)
    ema = close.ewm(span=ema_period, adjust=False).mean()
    rising = ema.diff() > 0
    falling = ema.diff() < 0
    direction = np.zeros(len(higher_df), dtype=int)
    direction[(close.values > ema.values) & rising.values] = 1
    direction[(close.values < ema.values) & falling.values] = -1
    return direction


def align_confirmation(
    base_df: pd.DataFrame,
    higher_df: pd.DataFrame,
    base_tf: str,
    higher_tf: str,
    ema_period: int = 50,
) -> np.ndarray:
    """
    Per-base-bar higher-TF direction in {-1,0,+1}, strictly no look-ahead.

    For each base bar, picks the most recent higher-TF bar that has already CLOSED
    by the base bar's close. Base bars before the first closed higher-TF bar get 0.
    """
    base_close = base_df["timestamp"].to_numpy(dtype=np.int64) + tf_ms(base_tf)
    higher_close = higher_df["timestamp"].to_numpy(dtype=np.int64) + tf_ms(higher_tf)
    hdir = higher_tf_direction(higher_df, ema_period)

    # For each base bar, index of the last higher bar with higher_close <= base_close.
    # searchsorted(..., 'right') gives count of higher_close <= value; minus 1 = index.
    idx = np.searchsorted(higher_close, base_close, side="right") - 1
    out = np.zeros(len(base_df), dtype=int)
    valid = idx >= 0
    out[valid] = hdir[idx[valid]]
    return out


def add_confirmation_columns(
    base_df: pd.DataFrame,
    base_tf: str,
    confirmations: dict,          # {higher_tf: higher_df}
    ema_period: int = 50,
) -> pd.DataFrame:
    """
    Returns a copy of base_df with one `_conf_<tf>` column per confirmation TF,
    each a no-look-ahead direction in {-1,0,+1}. Leaves base_df untouched.
    """
    out = base_df.copy()
    for higher_tf, higher_df in confirmations.items():
        out[f"_conf_{higher_tf}"] = align_confirmation(
            base_df, higher_df, base_tf, higher_tf, ema_period
        )
    return out


def confirmation_ok(row, side: str, confirmation_tfs: list, mode: str = "no_oppose") -> bool:
    """
    Decide whether a base-TF signal of `side` ("long"/"short") is confirmed.

    mode:
      - "no_oppose": take the signal unless any higher TF points the other way
        (neutral higher-TF is allowed through). Permissive.
      - "all_agree": every confirmation TF must point the SAME way as the signal.
        Strict (fewer, higher-quality trades).
    """
    want = 1 if side == "long" else -1
    dirs = [int(row.get(f"_conf_{tf}", 0)) for tf in confirmation_tfs]
    if not dirs:
        return True
    if mode == "all_agree":
        return all(d == want for d in dirs)
    # default "no_oppose"
    return all(d != -want for d in dirs)
