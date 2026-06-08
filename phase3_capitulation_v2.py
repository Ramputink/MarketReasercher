"""
phase3_capitulation_v2.py — Path 1: aggressive market-regime CASH gate
=====================================================================

Same spot long-only buy-capitulation signal as v1, but with a HARD market-regime
gate layered on top: total portfolio exposure is forced to 0 (all cash) whenever
the broad market is in a sustained downtrend. The thesis under test: a long-only
system can only survive a -46% bear by sitting OUT of it.

Discipline: the capitulation signal is FIXED at v1's in-sample-best config (no
re-search). Only the few NEW regime-gate params are calibrated, on IN-SAMPLE
ONLY. This is the SECOND read of the sealed lockbox — selection risk is therefore
higher than a virgin read; reported as such.

Regime ON (allowed to be long) requires BOTH:
  - BTC above its long MA (btc_ma days), AND
  - breadth = fraction of assets above their own 50d MA  >=  breadth_thresh
Otherwise -> 100% cash.

Run:  python phase3_capitulation_v2.py
"""
from __future__ import annotations

import json, itertools
import numpy as np
import pandas as pd

from phase3_capitulation import (load_panel, signal_weights, backtest, metrics, per_year)

# FIXED v1 in-sample-best capitulation signal (NOT re-searched here)
SIGNAL = {"funding_lookback": 7, "z_thresh": 1.0, "confirm_ma": 5,
          "risk_ma": 50, "hold_days": 10, "max_positions": 10}

# Only these regime-gate params are calibrated (in-sample only)
REGIME_GRID = {"btc_ma": [100, 150, 200], "breadth_thresh": [0.30, 0.40, 0.50]}
BREADTH_MA = 50


def regime_on(close, btc_ma, breadth_thresh):
    """Daily boolean: is the broad market risk-on enough to hold longs?"""
    btc_ok = close["BTC"] > close["BTC"].rolling(btc_ma).mean() if "BTC" in close else True
    above = (close > close.rolling(BREADTH_MA).mean())
    breadth = above.sum(axis=1) / close.notna().sum(axis=1).replace(0, np.nan)
    return (btc_ok & (breadth >= breadth_thresh)).fillna(False)


def gated_weights(close, funding, sig, reg):
    W = signal_weights(close, funding, sig)
    on = regime_on(close, reg["btc_ma"], reg["breadth_thresh"])
    return W.mul(on.astype(float), axis=0)   # force 0 (cash) when regime off


def main():
    close, funding, cutoff, man = load_panel()
    ins = close.index < cutoff
    print(f"panel {man['start']}→{man['end']}, lockbox sealed from {man['lockbox_cutoff_date']}")
    print(f"FIXED signal: {SIGNAL}")
    print(f"calibrating ONLY regime gate on in-sample "
          f"({len(list(itertools.product(*REGIME_GRID.values())))} configs)…\n")

    keys = list(REGIME_GRID)
    best = None
    for vals in itertools.product(*[REGIME_GRID[k] for k in keys]):
        reg = dict(zip(keys, vals))
        W = gated_weights(close, funding, SIGNAL, reg)
        Wi = W[ins]
        if Wi.abs().sum().sum() == 0:
            continue
        port, eq, wx = backtest(close[ins], Wi)
        if (wx.sum(axis=1) > 0).sum() < 40:
            continue
        m = metrics(port, eq, wx)
        score = m["calmar"] if m["cagr_pct"] > 0 else -9
        print(f"  btc_ma={reg['btc_ma']} breadth>={reg['breadth_thresh']}: "
              f"IS CAGR {m['cagr_pct']:6.1f}% Sharpe {m['sharpe']:.2f} "
              f"maxDD {m['max_dd_pct']:6.1f}% expo {m['avg_exposure_pct']:.0f}%")
        if best is None or score > best["score"]:
            best = {"score": score, "reg": reg, "is": m}

    print(f"\nBEST regime gate (in-sample): {best['reg']}")
    print(f"in-sample: CAGR {best['is']['cagr_pct']}% Sharpe {best['is']['sharpe']} "
          f"maxDD {best['is']['max_dd_pct']}% expo {best['is']['avg_exposure_pct']}%")

    # ── SECOND sealed-lockbox read ──
    reg = best["reg"]
    W = gated_weights(close, funding, SIGNAL, reg)
    Wl = W[~ins]
    port, eq, wx = backtest(close[~ins], Wl)
    lb = metrics(port, eq, wx); lby = per_year(port)
    bh_ret = close[~ins].pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    bh_eq = (1 + bh_ret).cumprod()
    bh = {"total_ret_pct": round((bh_eq.iloc[-1] - 1) * 100, 2),
          "max_dd_pct": round((bh_eq / bh_eq.cummax() - 1).min() * 100, 2)}

    out = {"strategy": "spot_longonly_buy_capitulation_v2_regime_gate",
           "signal": SIGNAL, "regime_gate": reg, "lockbox_read_number": 2,
           "in_sample": best["is"], "lockbox_oos": lb, "lockbox_per_year": lby,
           "lockbox_buy_hold_basket": bh}
    json.dump(out, open("reports/phase3_capitulation_v2_lockbox_results.json", "w"),
              indent=2, default=str)

    print("\n" + "=" * 70)
    print("SEALED LOCKBOX (2-yr OOS) — v2 with HARD regime cash-gate  [READ #2]")
    print("=" * 70)
    for k, v in lb.items():
        print(f"  {k:22} {v}")
    print(f"  per-year Sharpe: {lby}")
    print(f"\n  vs v1 (no hard gate): -15.3% / Sharpe 0.075 / maxDD -43.5%")
    print(f"  vs buy&hold basket : {bh['total_ret_pct']}% / maxDD {bh['max_dd_pct']}%")
    print("\nresults → reports/phase3_capitulation_v2_lockbox_results.json")
    return out


if __name__ == "__main__":
    main()
