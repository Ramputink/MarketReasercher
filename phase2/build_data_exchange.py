"""
phase2.build_data_exchange — freeze the SAME cross-sectional panel on a SECOND
exchange, for the cheapest possible out-of-sample generalisation test.
==============================================================================

The funding-carry edge was discovered on Binance USDⓈ-M. The single most
informative robustness check is whether the SAME pre-registered carry config
replicates on an INDEPENDENT exchange's data — if it does, the edge is a market
structure fact, not a Binance-specific artifact.

This builder is `build_data.py` generalised over the exchange. It fetches the
same UNIVERSE from any ccxt perp venue and freezes a hash-verified snapshot into
its own directory, stamping the SAME lockbox cutoff date as the Binance panel so
the in-sample / lockbox windows line up exactly for an apples-to-apples compare.

Run:
    python3 -m phase2.build_data_exchange bybit
    python3 -m phase2.build_data_exchange okx
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import time

import ccxt
import pandas as pd

from phase2 import UNIVERSE, perp, BASE_TF, FUNDING_INTERVAL_HOURS, LOCKBOX_DAYS, PKG_DIR
from phase2.build_data import _to_daily_index, _now_ms

# Match the Binance panel's sealed cutoff so the OOS windows align exactly.
FIXED_CUTOFF_DATE = "2024-06-07"
# Page forward from a real early date: ccxt `since=0` makes some venues (Bybit)
# return only the most-recent page, truncating history.
START_MS = int(pd.Timestamp("2019-01-01").value // 1_000_000)
DAY_MS = 86_400_000


def _page_forward(fetch, since, page, parse_ts, step_ms):
    """
    Generic forward paginator robust across venues. Stops on no-progress instead
    of on a partial page — Bybit caps pages well below the requested `limit`, so
    the `len < limit` heuristic used for Binance would quit after one page.
    """
    out, now, last_seen = [], _now_ms(), -1
    while True:
        try:
            rows = fetch(since, page)
        except Exception as e:
            print(f"      fetch err: {str(e)[:80]}"); break
        if not rows:
            break
        out.extend(rows)
        last_ts = parse_ts(rows[-1])
        if last_ts <= last_seen:          # no forward progress -> done
            break
        last_seen = last_ts
        nxt = last_ts + step_ms
        if nxt > now:
            break
        since = nxt
        time.sleep(0.05)
    return out


def _fetch_ohlcv_full(ex, sym):
    rows = _page_forward(
        lambda s, p: ex.fetch_ohlcv(sym, BASE_TF, since=s, limit=p),
        START_MS, 1000, lambda r: r[0], DAY_MS)
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def _fetch_funding_full(ex, sym, start_ms):
    interval = FUNDING_INTERVAL_HOURS * 3_600_000
    rows = _page_forward(
        lambda s, p: ex.fetch_funding_rate_history(sym, since=s, limit=p),
        int(start_ms), 200, lambda r: r["timestamp"], interval)
    if not rows:
        return pd.DataFrame(columns=["ts", "funding_rate"])
    df = pd.DataFrame([{"ts": r["timestamp"], "funding_rate": float(r["fundingRate"])}
                       for r in rows])
    return df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)


def _exchange(exchange_id: str) -> ccxt.Exchange:
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True, "timeout": 30000,
                                     "options": {"defaultType": "swap"}})
    ex.load_markets()
    return ex


def build_exchange(exchange_id: str) -> dict:
    out_dir = os.path.join(PKG_DIR, f"snapshots_{exchange_id}")
    os.makedirs(out_dir, exist_ok=True)
    ex = _exchange(exchange_id)

    closes, opens, fundings = {}, {}, {}
    for i, base in enumerate(UNIVERSE, 1):
        sym = perp(base)
        if sym not in ex.markets:
            print(f"[{i:2}/{len(UNIVERSE)}] {base:5} not listed on {exchange_id}, skip")
            continue
        print(f"[{i:2}/{len(UNIVERSE)}] {base:5} OHLCV …", flush=True)
        try:
            o = _fetch_ohlcv_full(ex, sym)
        except Exception as e:
            print(f"      OHLCV err {base}: {str(e)[:80]}"); continue
        if o.empty:
            print(f"      no OHLCV for {base}, skipping"); continue
        listing_ms = int(o["ts"].iloc[0])
        o["date"] = _to_daily_index(o["ts"])
        o = o.drop_duplicates("date", keep="last").set_index("date")
        closes[base] = o["close"]; opens[base] = o["open"]

        print(f"          funding …", flush=True)
        f = _fetch_funding_full(ex, sym, listing_ms)
        if not f.empty:
            f["date"] = _to_daily_index(f["ts"])
            fundings[base] = f.groupby("date")["funding_rate"].sum()
        else:
            fundings[base] = pd.Series(dtype=float)
        print(f"          {len(o)} bars, funding rows {len(f)}", flush=True)

    if not closes:
        print("no assets fetched — aborting"); return {}

    close_df = pd.DataFrame(closes).sort_index()
    open_df = pd.DataFrame(opens).reindex(close_df.index)
    fund_df = pd.DataFrame(fundings).reindex(close_df.index)
    ret_df = close_df.pct_change(fill_method=None)
    valid = close_df.notna().any(axis=1)
    close_df, open_df, fund_df, ret_df = (
        close_df[valid], open_df[valid], fund_df[valid], ret_df[valid])

    cutoff = pd.Timestamp(FIXED_CUTOFF_DATE)
    cutoff_ts = int(cutoff.value // 1_000_000)

    files = {"close": close_df, "open": open_df, "ret": ret_df, "funding": fund_df}
    manifest = {
        "exchange": exchange_id, "universe": list(close_df.columns), "base_tf": BASE_TF,
        "n_dates": int(len(close_df)),
        "start": str(close_df.index.min().date()), "end": str(close_df.index.max().date()),
        "lockbox_days": LOCKBOX_DAYS, "lockbox_cutoff_ts": cutoff_ts,
        "lockbox_cutoff_date": str(cutoff.date()),
        "cutoff_matches_binance_panel": True, "files": {},
    }
    for name, df in files.items():
        path = os.path.join(out_dir, f"{name}.parquet")
        df.to_parquet(path)
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        manifest["files"][name] = {"path": f"{name}.parquet", "sha256": h,
                                   "shape": list(df.shape)}
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print("\n── frozen ───────────────────────────────────────────")
    print(f"exchange: {exchange_id}")
    print(f"dates : {manifest['start']} → {manifest['end']}  ({manifest['n_dates']} days)")
    print(f"assets: {len(close_df.columns)} → {list(close_df.columns)}")
    print(f"lockbox sealed from {manifest['lockbox_cutoff_date']}")
    print(f"snapshot: {out_dir}")
    return manifest


if __name__ == "__main__":
    ex_id = sys.argv[1] if len(sys.argv) > 1 else "bybit"
    build_exchange(ex_id)
