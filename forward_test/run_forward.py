"""
forward_test.run_forward — score the pre-registered configs on TRUE-OOS data.
=============================================================================

On each run it (1) fetches a fresh full panel from the registered exchange,
(2) scores every immutable pre-registered config on ONLY the data strictly after
`forward_start`, and (3) appends one record per config to forward_log.jsonl. No
fitting; the configs never change. Over weeks this builds a real forward track
record that the historical lockbox cannot provide.

The live snapshot is regenerable and gitignored; only the append-only log and the
immutable pre-registration are version-controlled.

Run:  python3 -m forward_test.run_forward            # fetch fresh + score
      python3 -m forward_test.run_forward --no-fetch  # reuse last live snapshot
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from forward_test import PREREG_PATH, LOG_PATH, LIVE_SNAP, FORWARD_START
from phase2 import signals as sg, MIN_UNIVERSE
from phase2.portfolio import backtest as xs_backtest, PortfolioCosts
from benchmark import statistics as st


def _load_live_panel():
    man = json.load(open(os.path.join(LIVE_SNAP, "manifest.json")))
    panel = {}
    for name in ("close", "open", "funding"):
        df = pd.read_parquet(os.path.join(LIVE_SNAP, f"{name}.parquet"))
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        panel[name] = df
    panel["manifest"] = man
    return panel


def _score_carry(panel, cfg, fwd_start):
    """Carry MN over the forward window (warmup uses pre-forward data, legal)."""
    scores = sg.carry_score(panel["funding"], cfg["carry_lookback"])
    res = xs_backtest(
        scores, panel["open"], panel["close"], panel["funding"],
        rebalance_every=cfg["rebalance_every"], k=cfg["k"], long_only=cfg["long_only"],
        dollar_neutral=cfg["dollar_neutral"], weight_mode=cfg["weight_mode"],
        gross=cfg["gross"], vol_lookback=cfg["vol_lookback"], min_universe=MIN_UNIVERSE,
        costs=PortfolioCosts())
    eq = res.equity_curve
    fwd = eq.index > fwd_start
    per = res.period_returns[-int(fwd.sum()):] if fwd.sum() else np.array([])
    return per


def _score_phase3(panel, cfg, fwd_start):
    """Phase-3 3-sleeve cash-gating ensemble + overlay, daily forward returns."""
    from phase3_five_strategies import s1_trend, s4_capitulation, s5_breakout
    from phase3_portfolio_overlay import apply_overlay
    close, funding = panel["close"], panel["funding"]
    from phase3_capitulation import backtest as lo_backtest
    fns = {"trend_ts": s1_trend, "breakout": s5_breakout, "capitulation": s4_capitulation}
    ports = []
    for fn in fns.values():
        W = fn(close, funding)
        port, _, _ = lo_backtest(close, W)
        ports.append(port.reindex(close.index).fillna(0.0))
    ens = sum(ports) / len(ports)
    capped, _ = apply_overlay(ens, breaker=True)
    fwd = capped.index > fwd_start
    return capped[fwd].values


def _metrics(per: np.ndarray) -> dict:
    per = np.asarray(per, dtype=float)
    n = int(len(per))
    if n == 0:
        return {"n": 0, "status": "no forward data yet"}
    total = float(np.prod(1.0 + per) - 1.0) * 100.0
    out = {"n": n, "total_ret_pct": round(total, 3)}
    if n >= 2 and per.std() > 0:
        out["sharpe_per_obs"] = round(float(st.sharpe_per_obs(per)), 4)
        out["tstat"] = round(float(st.tstat_pvalue(per)["tstat"]), 3)
    out["status"] = "scored" if n >= 20 else f"accumulating ({n} obs; >=20 for a believable read)"
    return out


def main():
    if not os.path.exists(PREREG_PATH):
        raise SystemExit("no pre-registration. Run `python3 -m forward_test.preregister` first.")
    prereg = json.load(open(PREREG_PATH))
    fwd_start = pd.Timestamp(prereg["forward_start"])

    if "--no-fetch" not in sys.argv:
        from phase2.build_data_exchange import build_exchange
        print(f"fetching fresh {prereg['exchange']} panel → {LIVE_SNAP} …")
        build_exchange(prereg["exchange"], out_dir=LIVE_SNAP)

    panel = _load_live_panel()
    data_end = str(panel["close"].index.max().date())
    run_stamp = pd.Timestamp(panel["manifest"].get("end", data_end))

    records = []
    for name, cfg in prereg["configs"].items():
        if cfg["type"] == "carry_market_neutral":
            per = _score_carry(panel, cfg, fwd_start)
        elif cfg["type"] == "spot_longonly_ensemble_overlay":
            per = _score_phase3(panel, cfg, fwd_start)
        else:
            continue
        m = _metrics(per)
        rec = {"run_date": str(run_stamp.date()), "config": name,
               "prereg_hash": prereg["hash"], "forward_start": prereg["forward_start"],
               "data_end": data_end, **m}
        records.append(rec)

    with open(LOG_PATH, "a") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    print("\n" + "=" * 70)
    print(f"FORWARD TEST — true OOS after {prereg['forward_start']}  (data → {data_end})")
    print("=" * 70)
    for r in records:
        extra = (f"Sharpe/obs {r.get('sharpe_per_obs')}, tstat {r.get('tstat')}, "
                 f"ret {r.get('total_ret_pct')}%" if r["n"] else "")
        print(f"  {r['config']:26} n={r['n']:>3}  {r['status']}  {extra}")
    print(f"\nappended {len(records)} records → {LOG_PATH}")
    return records


if __name__ == "__main__":
    main()
