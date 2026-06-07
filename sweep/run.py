"""
sweep.run — honest timeframe sweep on BTC/ETH with MTF confirmation
===================================================================

For each base timeframe {15m,1h,4h,1d} and each asset {BTC,ETH}, search the
directional strategy families (with a no-look-ahead higher-TF confirmation gate),
score honestly on the SEALED lockbox, and report — per (tf,asset):

  - best lockbox Sharpe / profit factor / trades   (true OOS)
  - Deflated Sharpe Ratio, deflated by the real number of search trials
  - cross-asset check (best genome run on the OTHER asset's lockbox)

The whole thing reuses the benchmark's own leak-free ConservativeBacktester and
DSR statistics — the same honesty bar that rejected XRP. Nothing here can certify
an edge the referee wouldn't.

Run (background-friendly):
    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m sweep.run [n_samples]
"""
from __future__ import annotations

import copy
import importlib
import json
import os
import random
import sys

import numpy as np
import pandas as pd

from sweep import SWEEP_SYMBOLS, SWEEP_TIMEFRAMES
from sweep.build_data import SNAP_DIR, MANIFEST
from sweep import mtf
from benchmark.harness import ConservativeBacktester, ExecModel
from benchmark import statistics as st

# Directional families only (lstm_pattern excluded — needs TF/sklearn).
from auto_evolve import STRATEGY_REGISTRY, random_genome

DIRECTIONAL = [s for s in STRATEGY_REGISTRY if s != "lstm_pattern"]
BAR_HOURS = {"15m": 0.25, "1h": 1.0, "4h": 4.0, "1d": 24.0}
CONF_MODE = "no_oppose"          # higher TF may not contradict the signal
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading + split
# ─────────────────────────────────────────────────────────────────────────────

def _load(symbol: str, tf: str) -> pd.DataFrame:
    with open(MANIFEST) as f:
        man = json.load(f)
    meta = man["entries"][f"{symbol}|{tf}"]
    df = pd.read_parquet(os.path.join(SNAP_DIR, meta["path"]))
    df.attrs["lockbox_cutoff_ts"] = meta["lockbox_cutoff_ts"]
    return df


def _prepare(symbol: str, base_tf: str):
    """Base df with MTF confirmation columns + in-sample/lockbox split."""
    base = _load(symbol, base_tf)
    confs = {h: _load(symbol, h) for h in SWEEP_TIMEFRAMES[base_tf]}
    base = mtf.add_confirmation_columns(base, base_tf, confs)
    cut = base.attrs["lockbox_cutoff_ts"]
    in_s = base[base["timestamp"] < cut].reset_index(drop=True)
    lock = base[base["timestamp"] >= cut].reset_index(drop=True)
    return in_s, lock


# ─────────────────────────────────────────────────────────────────────────────
# Strategy + MTF gate
# ─────────────────────────────────────────────────────────────────────────────

def _build_fn(strategy: str, params: dict, confirmation_tfs: list):
    reg = STRATEGY_REGISTRY[strategy]
    mod = importlib.import_module(reg["module"])
    fn = getattr(mod, reg["function"])
    pattr = reg["params_dict"]
    merged = copy.deepcopy(getattr(mod, pattr))
    merged.update(params)

    def strategy_fn(df, i, pos):
        original = getattr(mod, pattr)
        setattr(mod, pattr, merged)
        try:
            regime = str(df.iloc[i].get("_regime", "unknown")) if i < len(df) else "unknown"
            sig = fn(df, i, pos, regime=regime)
        finally:
            setattr(mod, pattr, original)
        if sig is None:
            return None
        # MTF confirmation gate (no look-ahead — columns precomputed leak-free).
        if confirmation_tfs and not mtf.confirmation_ok(df.iloc[i], sig.side, confirmation_tfs, CONF_MODE):
            return None
        return sig

    return strategy_fn


def _exec(bar_hours: float) -> ExecModel:
    return ExecModel(
        commission_rate=0.001, slippage_bps=5.0, funding_bps_per_8h=1.0,
        entry_lag_bars=1, position_pct=0.10, initial_capital=10_000.0,
        warmup_bars=60, bar_hours=bar_hours,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep(n_samples: int = 150, seed: int = 42):
    os.makedirs(RESULTS, exist_ok=True)
    rng = random.Random(seed)
    results = []

    for base_tf in SWEEP_TIMEFRAMES:
        conf_tfs = SWEEP_TIMEFRAMES[base_tf]
        bh = BAR_HOURS[base_tf]
        bt = ConservativeBacktester(_exec(bh))

        for sym in SWEEP_SYMBOLS:
            in_s, lock = _prepare(sym, base_tf)
            other = [s for s in SWEEP_SYMBOLS if s != sym][0]
            _, other_lock = _prepare(other, base_tf)

            best = None
            for t in range(n_samples):
                strat = rng.choice(DIRECTIONAL)
                g = random_genome(strat, generation=0)
                fn = _build_fn(strat, g.params, conf_tfs)
                r_is = bt.run(in_s, fn, strat)
                # Admission floor for the in-sample search. 1d on ~245 in-sample
                # bars is inherently trade-poor — results there are underpowered.
                if r_is.n_trades < 15 or r_is.ruined:
                    continue
                score = r_is.sharpe_annualised
                if best is None or score > best["is_sharpe"]:
                    best = {"strategy": strat, "params": g.params,
                            "is_sharpe": score, "is_trades": r_is.n_trades}

            row = {"tf": base_tf, "symbol": sym, "confirm": conf_tfs,
                   "n_samples": n_samples, "best": None}
            if best is not None:
                fn = _build_fn(best["strategy"], best["params"], conf_tfs)
                r_lock = bt.run(lock, fn, best["strategy"])
                r_other = bt.run(other_lock, fn, best["strategy"])
                dsr = st.deflated_sharpe_ratio(r_lock.trade_returns, n_samples, None) \
                    if r_lock.n_trades >= 5 else {"dsr": 0.0, "sr_star": 0.0}
                row["best"] = {
                    "strategy": best["strategy"],
                    "is_sharpe": round(best["is_sharpe"], 3),
                    "is_trades": best["is_trades"],
                    "lockbox_sharpe": round(r_lock.sharpe_annualised, 3),
                    "lockbox_pf": round(r_lock.profit_factor, 3) if np.isfinite(r_lock.profit_factor) else None,
                    "lockbox_trades": r_lock.n_trades,
                    "lockbox_net_pnl": round(r_lock.net_pnl, 2),
                    "lockbox_maxdd": round(r_lock.max_drawdown_pct, 2),
                    "lockbox_ruined": r_lock.ruined,
                    "dsr": round(dsr["dsr"], 4),
                    "dsr_sr_star": round(dsr.get("sr_star", 0), 4),
                    f"cross_{other.split('/')[0]}_sharpe": round(r_other.sharpe_annualised, 3),
                    f"cross_{other.split('/')[0]}_pnl": round(r_other.net_pnl, 2),
                    "params": best["params"],
                }
            results.append(row)
            b = row["best"]
            if b:
                print(f"[{base_tf:>3} {sym:8} conf={conf_tfs}] best={b['strategy']:18} "
                      f"IS_Sh={b['is_sharpe']:+.2f}({b['is_trades']}t) -> "
                      f"LOCK_Sh={b['lockbox_sharpe']:+.2f}({b['lockbox_trades']}t) "
                      f"DSR={b['dsr']:.3f} cross={b[f'cross_{other.split(chr(47))[0]}_sharpe']:+.2f}",
                      flush=True)
            else:
                print(f"[{base_tf:>3} {sym:8}] no genome cleared the 20-trade floor", flush=True)

    out = os.path.join(RESULTS, "sweep_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)

    _summary(results)
    print(f"\nFull results -> {out}")
    return results


def _summary(results):
    print("\n" + "=" * 100)
    print("TIMEFRAME SWEEP — honest lockbox results (BTC/ETH, MTF confirmation)")
    print("=" * 100)
    print(f"{'TF':>4} {'asset':8} {'best strat':18} {'IS_Sh':>6} {'LOCK_Sh':>8} "
          f"{'trades':>6} {'PF':>5} {'DSR':>6} {'cross_Sh':>8} {'maxDD%':>7}")
    print("-" * 100)
    for r in results:
        b = r["best"]
        if not b:
            print(f"{r['tf']:>4} {r['symbol']:8} {'(none)':18}")
            continue
        cross = next((v for k, v in b.items() if k.startswith("cross_") and k.endswith("_sharpe")), 0)
        print(f"{r['tf']:>4} {r['symbol']:8} {b['strategy']:18} {b['is_sharpe']:+6.2f} "
              f"{b['lockbox_sharpe']:+8.2f} {b['lockbox_trades']:>6} "
              f"{(b['lockbox_pf'] or 0):>5.2f} {b['dsr']:>6.3f} {cross:+8.2f} {b['lockbox_maxdd']:>7.1f}")
    print("-" * 100)
    cert = [r for r in results if r["best"] and r["best"]["dsr"] >= 0.95]
    print(f"Certifiable (DSR>=0.95): {len(cert)} of {len(results)}")
    print("Note: DSR deflated by per-(tf,asset) search trials. Lockbox is sealed (never searched).")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    sweep(n_samples=n)
