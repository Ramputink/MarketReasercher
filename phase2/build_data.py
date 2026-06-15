"""
phase2.build_data — freeze the cross-sectional panel (OHLCV + funding)
======================================================================

Fetches full-history daily OHLCV and 8h funding-rate history for every perp in
the universe from Binance USDⓈ-M, aligns them onto one common daily UTC date
index, and freezes:

    snapshots/close.parquet     (T × N)  daily close price per asset
    snapshots/open.parquet      (T × N)  daily open  (next-bar execution price)
    snapshots/ret.parquet       (T × N)  daily simple close-to-close return
    snapshots/funding.parquet   (T × N)  daily summed funding rate (long pays +)
    snapshots/manifest.json     SHA-256 of every file + lockbox cutoff timestamp

Missing data before an asset lists is left as NaN — the portfolio engine treats
NaN as "not in the tradeable universe yet", which is the honest, survivorship-
free behaviour. The lockbox cutoff is stamped here ONCE so every later run seals
the identical final `LOCKBOX_DAYS` of history.

Run:
    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m phase2.build_data
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import ccxt
import numpy as np
import pandas as pd

from phase2 import (
    UNIVERSE, perp, BASE_TF, FUNDING_INTERVAL_HOURS,
    LOCKBOX_DAYS, SNAP_DIR, MANIFEST, EXCHANGE_ID,
)

DAY_MS = 86_400_000


def _exchange() -> ccxt.Exchange:
    ex = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True, "timeout": 30000})
    ex.load_markets()
    return ex


def _now_ms() -> int:
    # avoid Date.now-style nondeterminism concerns; this is build-time only
    return int(pd.Timestamp.utcnow().value // 1_000_000)


def _fetch_ohlcv_full(ex: ccxt.Exchange, sym: str) -> pd.DataFrame:
    """Paginate daily OHLCV from listing to now (page cap = 1000)."""
    out, since, limit = [], 0, 1000
    now = _now_ms()
    while True:
        candles = ex.fetch_ohlcv(sym, BASE_TF, since=since, limit=limit)
        if not candles:
            break
        out.extend(candles)
        last_ts = candles[-1][0]
        nxt = last_ts + DAY_MS
        if nxt <= since or nxt > now:      # no progress or reached present
            break
        if len(candles) < limit:           # partial page ⇒ caught up to present
            break
        since = nxt
        time.sleep(ex.rateLimit / 1000)
    if not out:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df


def _fetch_funding_full(ex: ccxt.Exchange, sym: str, start_ms: int) -> pd.DataFrame:
    """
    Paginate 8h funding-rate history forward from `start_ms` (the asset's listing
    date) to now. NOTE: Binance ignores since=0 and returns only the latest page,
    so a real start timestamp is mandatory.
    """
    out, since, limit = [], int(start_ms), 1000
    now = _now_ms()
    interval = FUNDING_INTERVAL_HOURS * 3_600_000
    while True:
        try:
            rows = ex.fetch_funding_rate_history(sym, since=since, limit=limit)
        except Exception as e:
            print(f"      funding fetch err {sym}: {str(e)[:80]}")
            break
        if not rows:
            break
        out.extend(rows)
        last_ts = rows[-1]["timestamp"]
        nxt = last_ts + interval
        if nxt <= since or nxt > now:
            break
        if len(rows) < limit:
            break
        since = nxt
        time.sleep(ex.rateLimit / 1000)
    if not out:
        return pd.DataFrame(columns=["ts", "funding_rate"])
    df = pd.DataFrame([{"ts": r["timestamp"], "funding_rate": float(r["fundingRate"])}
                       for r in out])
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df


def _to_daily_index(ts_ms: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(ts_ms, unit="ms", utc=True).dt.floor("D")


def build() -> dict:
    os.makedirs(SNAP_DIR, exist_ok=True)
    ex = _exchange()

    closes, opens, fundings = {}, {}, {}
    for i, base in enumerate(UNIVERSE, 1):
        sym = perp(base)
        print(f"[{i:2}/{len(UNIVERSE)}] {base:5} OHLCV …", flush=True)
        o = _fetch_ohlcv_full(ex, sym)
        if o.empty:
            print(f"      no OHLCV for {base}, skipping")
            continue
        listing_ms = int(o["ts"].iloc[0])
        o["date"] = _to_daily_index(o["ts"])
        o = o.drop_duplicates("date", keep="last").set_index("date")
        closes[base] = o["close"]
        opens[base] = o["open"]

        print(f"          funding …", flush=True)
        f = _fetch_funding_full(ex, sym, listing_ms)
        if not f.empty:
            f["date"] = _to_daily_index(f["ts"])
            # daily funding = sum of the (typically 3) 8h payments that day
            daily_f = f.groupby("date")["funding_rate"].sum()
            fundings[base] = daily_f
        else:
            fundings[base] = pd.Series(dtype=float)
        print(f"          {len(o)} bars, funding rows {len(f)}", flush=True)

    close_df = pd.DataFrame(closes).sort_index()
    open_df = pd.DataFrame(opens).reindex(close_df.index)
    fund_df = pd.DataFrame(fundings).reindex(close_df.index)
    ret_df = close_df.pct_change(fill_method=None)

    # trim to the common usable span (drop all-NaN leading rows)
    valid = close_df.notna().any(axis=1)
    close_df, open_df, fund_df, ret_df = (
        close_df[valid], open_df[valid], fund_df[valid], ret_df[valid]
    )

    # lockbox cutoff: seal final LOCKBOX_DAYS by timestamp
    last_date = close_df.index.max()
    cutoff = last_date - pd.Timedelta(days=LOCKBOX_DAYS)
    cutoff_ts = int(cutoff.value // 1_000_000)  # ns -> ms

    files = {
        "close": close_df, "open": open_df, "ret": ret_df, "funding": fund_df,
    }
    manifest = {
        "exchange": EXCHANGE_ID,
        "universe": UNIVERSE,
        "base_tf": BASE_TF,
        "n_dates": int(len(close_df)),
        "start": str(close_df.index.min().date()),
        "end": str(close_df.index.max().date()),
        "lockbox_days": LOCKBOX_DAYS,
        "lockbox_cutoff_ts": cutoff_ts,
        "lockbox_cutoff_date": str(cutoff.date()),
        "files": {},
    }
    for name, df in files.items():
        path = os.path.join(SNAP_DIR, f"{name}.parquet")
        df.to_parquet(path)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        manifest["files"][name] = {"path": f"{name}.parquet", "sha256": h,
                                   "shape": list(df.shape)}

    with open(MANIFEST, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n── frozen ───────────────────────────────────────────")
    print(f"dates : {manifest['start']} → {manifest['end']}  ({manifest['n_dates']} days)")
    print(f"assets: {list(close_df.columns)}")
    print(f"lockbox sealed from {manifest['lockbox_cutoff_date']} "
          f"(last {LOCKBOX_DAYS} days)")
    print(f"manifest: {MANIFEST}")
    return manifest


if __name__ == "__main__":
    build()
