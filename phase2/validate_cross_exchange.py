"""
phase2.validate_cross_exchange — does the carry edge replicate on a 2nd venue?
=============================================================================

The cheapest, strongest robustness test for the funding-contrarian carry edge
(discovered on Binance USDⓈ-M): run the EXACT pre-registered canonical config —
no re-fitting, no search — on an INDEPENDENT exchange's frozen panel and read its
sealed lockbox once. Because exactly ONE config is evaluated, the Deflated Sharpe
is deflated by a single trial (the most favourable, and most honest, deflation).

If the same config is positive and significant on a different exchange's funding
and prices, the edge is a market-structure fact, not a Binance data artifact. If
it collapses, that is the honest verdict.

Run:
    python3 -m phase2.validate_cross_exchange bybit
    python3 -m phase2.validate_cross_exchange bybit okx   # several venues
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from phase2 import PKG_DIR, RESULTS_DIR, MIN_UNIVERSE
from phase2 import signals as sg
from phase2.portfolio import backtest, PortfolioCosts
from benchmark import statistics as st

# ── THE pre-registered canonical carry config (frozen; identical across venues) ─
CANONICAL_CARRY = dict(
    family="carry", k=4, rebalance_every=7, long_only=False, dollar_neutral=True,
    weight_mode="equal", gross=1.0, vol_lookback=20, carry_lookback=7,
)


def _load_panel(snap_dir: str) -> dict:
    man = json.load(open(os.path.join(snap_dir, "manifest.json")))
    panel = {}
    for name in ("close", "open", "ret", "funding"):
        df = pd.read_parquet(os.path.join(snap_dir, f"{name}.parquet"))
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        panel[name] = df
    panel["cutoff"] = pd.Timestamp(man["lockbox_cutoff_ts"], unit="ms").floor("D")
    panel["manifest"] = man
    return panel


def _run(panel: dict, window: str, c=CANONICAL_CARRY) -> dict:
    scores = sg.carry_score(panel["funding"], c["carry_lookback"])
    idx = scores.index
    cutoff = panel["cutoff"]
    mask = (idx >= cutoff) if window == "lockbox" else (idx < cutoff)
    res = backtest(
        scores[mask], panel["open"][mask], panel["close"][mask], panel["funding"][mask],
        rebalance_every=c["rebalance_every"], k=c["k"], long_only=c["long_only"],
        dollar_neutral=c["dollar_neutral"], weight_mode=c["weight_mode"],
        gross=c["gross"], vol_lookback=c["vol_lookback"], min_universe=MIN_UNIVERSE,
        costs=PortfolioCosts(),
    )
    per = res.period_returns
    out = {"window": window, "n": int(len(per)), "ret_pct": round(float(res.total_return_pct), 2),
           "turnover": round(float(res.turnover_mean), 3),
           "funding_share_pct": round(float(res.funding_pnl_pct), 2),
           "ruined": bool(res.ruined)}
    if len(per) >= 2:
        out["sharpe"] = round(float(st.sharpe_per_obs(per)), 4)
        out["tstat"] = round(float(st.tstat_pvalue(per)["tstat"]), 3)
        if window == "lockbox":
            dsr = st.deflated_sharpe_ratio(per, n_trials=1)   # single pre-registered config
            out["dsr"] = round(float(dsr["dsr"]), 4)
            out["certified"] = bool(dsr["dsr"] >= 0.95 and out["sharpe"] > 0
                                    and out["tstat"] >= 2.0 and len(per) >= 30)
    else:
        out["sharpe"] = None
    return out


def validate(exchange_id: str) -> dict:
    snap = os.path.join(PKG_DIR, f"snapshots_{exchange_id}")
    if not os.path.isdir(snap):
        raise SystemExit(f"no snapshot for {exchange_id}: run "
                         f"`python3 -m phase2.build_data_exchange {exchange_id}` first")
    panel = _load_panel(snap)
    man = panel["manifest"]
    res = {"exchange": exchange_id, "config": CANONICAL_CARRY,
           "panel": {"start": man["start"], "end": man["end"], "n_dates": man["n_dates"],
                     "n_assets": len(man["universe"]), "lockbox_cutoff": man["lockbox_cutoff_date"]},
           "in_sample": _run(panel, "is"), "lockbox": _run(panel, "lockbox")}
    return res


def main():
    venues = sys.argv[1:] or ["bybit"]
    # Binance reference (the original discovery panel)
    binance = _load_panel(os.path.join(PKG_DIR, "snapshots"))
    ref = {"exchange": "binanceusdm",
           "panel": {"start": binance["manifest"]["start"], "end": binance["manifest"]["end"]},
           "in_sample": _run(binance, "is"), "lockbox": _run(binance, "lockbox")}

    rows = [ref] + [validate(v) for v in venues]

    print("=" * 78)
    print("CROSS-EXCHANGE REPLICATION — canonical funding-carry (clb7/k4/weekly/MN)")
    print("=" * 78)
    hdr = f"{'exchange':14} {'window':8} {'n':>4} {'Sharpe':>8} {'tstat':>7} {'ret%':>9} {'DSR':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        for w in ("in_sample", "lockbox"):
            m = r[w]
            print(f"{r['exchange']:14} {w:8} {m['n']:>4} "
                  f"{(m.get('sharpe') if m.get('sharpe') is not None else float('nan')):>8.4f} "
                  f"{m.get('tstat', float('nan')):>7.3f} {m['ret_pct']:>9.2f} "
                  f"{m.get('dsr', float('nan')):>6.3f}")
    print()
    for r in rows[1:]:
        lb = r["lockbox"]
        verdict = ("REPLICATES (same sign, significant)" if lb["sharpe"] and lb["sharpe"] > 0
                   and lb.get("tstat", 0) >= 1.5 else "does NOT replicate")
        print(f"  {r['exchange']}: lockbox Sharpe {lb['sharpe']}, tstat {lb.get('tstat')}, "
              f"ret {lb['ret_pct']}% → {verdict}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "cross_exchange_carry.json")
    json.dump({"reference_and_venues": rows}, open(out_path, "w"), indent=2, default=str)
    print(f"\nresults → {out_path}")
    return rows


if __name__ == "__main__":
    main()
