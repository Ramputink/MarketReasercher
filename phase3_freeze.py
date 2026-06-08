"""
phase3_freeze.py — pre-screen and FREEZE the long-only candidate set
====================================================================

Reads the per-strategy best genomes from the evolution checkpoint, re-evaluates
each on the IN-SAMPLE window (spot long-only + macro risk-off gate), and applies
an honest pre-screen before the sealed 2-year lockbox is ever touched:

  trades >= MIN_TRADES, in-sample Sharpe > 0, walk-forward-ish per-year stability
  (not >1 deeply-negative year), positive median cross-year, survives 2x costs.

Survivors are frozen to reports/phase3_frozen_candidates.json with their params
and in-sample diagnostics + selection rationale. NOTHING here reads the lockbox.

Run:  PHASE3_LONG_ONLY=1 PHASE3_MACRO_GATE=1 python phase3_freeze.py
"""
from __future__ import annotations

import copy
import importlib
import json
import os
import numpy as np
import pandas as pd

from benchmark.spec import BENCHMARK_SPEC as SPEC
from benchmark.data_lockbox import DataLockbox
from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester
from benchmark import statistics as st
from auto_evolve import STRATEGY_REGISTRY

MIN_TRADES = 40
CKPT = "reports/evolution_checkpoint.json"
OUT = "reports/phase3_frozen_candidates.json"


def _gated_fn(mod, fn, macro_gate=True):
    def f(d, i, p):
        sig = fn(d, i, p, regime=d.iloc[i].get("_regime", "unknown") if i < len(d) else "unknown")
        if sig is not None and getattr(sig, "side", None) != "long" and p is None:
            return None
        if (macro_gate and sig is not None and p is None
                and getattr(sig, "side", None) == "long"
                and float(d.iloc[i].get("macro_bullish", 1.0)) < 0.5):
            return None
        return sig
    return f


def _run(df, strat, params, costs_mult=1.0):
    reg = STRATEGY_REGISTRY[strat]
    mod = importlib.import_module(reg["module"])
    fn = getattr(mod, reg["function"])
    pref = copy.deepcopy(getattr(mod, reg["params_dict"]))
    pref.update(params)
    setattr(mod, reg["params_dict"], pref)
    cfg = BacktestConfig()
    if costs_mult != 1.0:
        cfg = BacktestConfig(commission_rate=cfg.commission_rate * costs_mult,
                             slippage_bps=cfg.slippage_bps * 4.0)
    bt = Backtester(cfg, RiskConfig(), funding_bps_per_8h=0.0, long_only=True)
    tr, eq, m = bt.run(df, _gated_fn(mod, fn), strat)
    return tr, np.asarray(eq, float), m


def _yearly(df, tr):
    if len(tr) == 0:
        return {}
    tr = tr.copy()
    tr["year"] = pd.to_datetime(tr["exit_time"], unit="ms").dt.year
    out = {}
    for y, g in tr.groupby("year"):
        r = g["pnl_net"].values
        out[int(y)] = round(st.sharpe_per_obs(r), 3) if len(r) > 1 else 0.0
    return out


def main():
    df = DataLockbox(SPEC).in_sample()
    assert int(df["timestamp"].max()) < DataLockbox(SPEC).lockbox_cutoff_ts(), "LOCKBOX LEAK"
    ck = json.load(open(CKPT))
    ps = ck.get("hall_of_fame_per_strategy", {})
    best = {s: max(gl, key=lambda g: g.get("fitness", -99)) for s, gl in ps.items() if gl}

    results = []
    for strat, g in best.items():
        if strat not in STRATEGY_REGISTRY:
            continue
        params = g.get("params", {})
        try:
            tr, eq, m = _run(df, strat, params, 1.0)
            tr2, _, _ = _run(df, strat, params, 2.0)  # cost-stress (2x comm, 4x slip)
        except Exception as e:
            print(f"  {strat}: eval error {e}")
            continue
        rets = tr["pnl_net"].values if len(tr) else np.array([])
        sh = st.sharpe_per_obs(rets) if len(rets) > 1 else 0.0
        sh_stress = st.sharpe_per_obs(tr2["pnl_net"].values) if len(tr2) > 1 else 0.0
        dd = ((eq / np.maximum.accumulate(eq)) - 1).min() * 100 if len(eq) > 1 else 0.0
        yearly = _yearly(df, tr)
        neg_years = sum(1 for v in yearly.values() if v < 0)
        n = len(tr)
        passed = (n >= MIN_TRADES and sh > 0 and neg_years <= 1
                  and (np.median(list(yearly.values())) if yearly else -1) > 0
                  and sh_stress > 0 and abs(dd) < 25
                  and float(g.get("cross_asset_median_sharpe", -1)) > 0)
        results.append({
            "strategy": strat, "params": params,
            "in_sample": {"trades": n, "sharpe": round(sh, 3),
                          "sharpe_cost_stress_2x": round(sh_stress, 3),
                          "max_dd_pct": round(dd, 2),
                          "wf_sharpe": g.get("wf_sharpe"),
                          "cross_asset_median_sharpe": g.get("cross_asset_median_sharpe"),
                          "cross_asset_profitable": g.get("cross_asset_profitable"),
                          "yearly_sharpe": yearly, "neg_years": neg_years},
            "passed_prescreen": bool(passed),
        })

    results.sort(key=lambda r: (r["passed_prescreen"], r["in_sample"]["sharpe"]), reverse=True)
    frozen = [r for r in results if r["passed_prescreen"]]
    out = {
        "spec_version": SPEC.version, "spec_fingerprint": SPEC.fingerprint(),
        "lockbox_cutoff_ts": DataLockbox(SPEC).lockbox_cutoff_ts(),
        "min_trades": MIN_TRADES, "macro_gate": True, "long_only": True,
        "n_passed": len(frozen),
        "frozen_candidates": frozen,
        "all_evaluated": results,
    }
    json.dump(out, open(OUT, "w"), indent=2, default=str)

    print(f"\n{'strategy':18} {'trades':>6} {'IS_Sh':>6} {'stress':>6} {'maxDD%':>7} {'negYr':>5} {'xasset':>6} {'PASS':>5}")
    for r in results:
        i = r["in_sample"]
        print(f"{r['strategy']:18} {i['trades']:6} {i['sharpe']:6.2f} {i['sharpe_cost_stress_2x']:6.2f} "
              f"{i['max_dd_pct']:7.1f} {i['neg_years']:5} {float(i['cross_asset_median_sharpe'] or 0):6.2f} "
              f"{'YES' if r['passed_prescreen'] else 'no':>5}")
    print(f"\nFROZEN: {len(frozen)} candidates -> {OUT}")
    return out


if __name__ == "__main__":
    main()
