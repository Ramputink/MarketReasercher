"""
sweep.run — honest timeframe sweep on BTC/ETH (USDC) with MTF confirmation
==========================================================================

For each base timeframe {15m,1h,4h,1d} and each asset {BTC/USDC,ETH/USDC},
search the directional strategy families (with a no-look-ahead higher-TF
confirmation gate), score honestly on the SEALED lockbox, and report per
(tf,asset):

  - best lockbox Sharpe / profit factor / trades   (true OOS)
  - Deflated Sharpe Ratio, deflated by the real number of search trials
  - generalisation check (best genome run on the OTHER asset's lockbox — NOT a
    pairs trade; BTC and ETH are each traded independently vs USDC)

Reuses the benchmark's own leak-free ConservativeBacktester + DSR statistics —
the same honesty bar that rejected XRP. The in-sample search is parallelised
across cores so full-history 15m (~260k bars) is tractable.

Run (background-friendly):
    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m sweep.run [n_samples] [cores]
"""
from __future__ import annotations

import copy
import importlib
import json
import multiprocessing as mp
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
from auto_evolve import STRATEGY_REGISTRY, random_genome

DIRECTIONAL = [s for s in STRATEGY_REGISTRY if s != "lstm_pattern"]
BAR_HOURS = {"15m": 0.25, "1h": 1.0, "4h": 4.0, "1d": 24.0}
CONF_MODE = "no_oppose"
MIN_TRADES = 15
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def _load(symbol: str, tf: str) -> pd.DataFrame:
    with open(MANIFEST) as f:
        man = json.load(f)
    meta = man["entries"][f"{symbol}|{tf}"]
    df = pd.read_parquet(os.path.join(SNAP_DIR, meta["path"]))
    df.attrs["lockbox_cutoff_ts"] = meta["lockbox_cutoff_ts"]
    return df


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Halve memory: float64 feature columns -> float32. Negligible effect on
    Sharpe/DSR (~1e-7), but it keeps full-history 15m (~246k bars) within RAM so
    parallel workers don't OOM."""
    cols = df.select_dtypes(include=["float64"]).columns
    for c in cols:
        df[c] = df[c].astype("float32")
    return df


def _prepare(symbol: str, base_tf: str):
    base = _load(symbol, base_tf)
    confs = {h: _load(symbol, h) for h in SWEEP_TIMEFRAMES[base_tf]}
    base = mtf.add_confirmation_columns(base, base_tf, confs)
    cut = base.attrs["lockbox_cutoff_ts"]
    in_s = _downcast(base[base["timestamp"] < cut].reset_index(drop=True))
    lock = _downcast(base[base["timestamp"] >= cut].reset_index(drop=True))
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
# Parallel worker: load the in-sample df once per process, evaluate genomes
# ─────────────────────────────────────────────────────────────────────────────

# The in-sample df is set in the PARENT before each Pool is forked. With the fork
# start method the workers inherit it copy-on-write — one shared physical copy for
# ALL cores, not one per worker. That removes the per-worker memory blowup that
# caused the OOM, so we can run at full core count.
_WDF = None

def _weval(args):
    strategy, params, conf_tfs, bar_hours = args
    try:
        fn = _build_fn(strategy, params, conf_tfs)
        r = ConservativeBacktester(_exec(bar_hours)).run(_WDF, fn, strategy)
        if r.n_trades < MIN_TRADES or r.ruined:
            return None
        return (float(r.sharpe_annualised), int(r.n_trades), strategy, params)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def sweep(n_samples: int = 250, cores: int = 0, seed: int = 42):
    # macOS default 'spawn' stalls here (workers re-import the heavy stack and
    # hang); 'fork' runs the pool correctly. Fork before any backtest work.
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    os.makedirs(RESULTS, exist_ok=True)
    rng = random.Random(seed)
    # Full multicore: copy-on-write sharing means more workers cost ~no extra RAM.
    cores = cores if cores > 0 else max(1, mp.cpu_count() - 1)
    print(f"sweep: {cores} cores (fork + copy-on-write shared in-sample df)", flush=True)
    results = []

    # Fastest TFs first (fewest bars) so higher-TF results — where a directional
    # edge is most plausible — land within minutes, not behind the slow 15m run.
    fast_first = sorted(SWEEP_TIMEFRAMES, key=lambda tf: BAR_HOURS[tf], reverse=True)
    for base_tf in fast_first:
        conf_tfs = SWEEP_TIMEFRAMES[base_tf]
        bh = BAR_HOURS[base_tf]

        for sym in SWEEP_SYMBOLS:
            in_s, lock = _prepare(sym, base_tf)
            other = [s for s in SWEEP_SYMBOLS if s != sym][0]
            _, other_lock = _prepare(other, base_tf)

            # Share the in-sample df with workers via fork copy-on-write (set in
            # the parent, inherited read-only by every worker — no per-worker copy).
            global _WDF
            _WDF = in_s
            genomes = [random_genome(rng.choice(DIRECTIONAL), generation=0) for _ in range(n_samples)]
            tasks = [(g.strategy, g.params, conf_tfs, bh) for g in genomes]
            with mp.Pool(processes=cores) as pool:
                evald = [x for x in pool.map(_weval, tasks) if x is not None]
            _WDF = None

            best = max(evald, key=lambda x: x[0]) if evald else None
            row = {"tf": base_tf, "symbol": sym, "confirm": conf_tfs,
                   "n_samples": n_samples, "in_sample_bars": len(in_s),
                   "lockbox_bars": len(lock), "best": None}

            if best is not None:
                is_sh, is_tr, strat, params = best
                fn = _build_fn(strat, params, conf_tfs)
                r_lock = ConservativeBacktester(_exec(bh)).run(lock, fn, strat)
                r_other = ConservativeBacktester(_exec(bh)).run(other_lock, fn, strat)
                dsr = st.deflated_sharpe_ratio(r_lock.trade_returns, n_samples, None) \
                    if r_lock.n_trades >= 5 else {"dsr": 0.0, "sr_star": 0.0}
                ocoin = other.split("/")[0]
                row["best"] = {
                    "strategy": strat, "is_sharpe": round(is_sh, 3), "is_trades": is_tr,
                    "lockbox_sharpe": round(r_lock.sharpe_annualised, 3),
                    "lockbox_pf": round(r_lock.profit_factor, 3) if np.isfinite(r_lock.profit_factor) else None,
                    "lockbox_trades": r_lock.n_trades,
                    "lockbox_net_pnl": round(r_lock.net_pnl, 2),
                    "lockbox_maxdd": round(r_lock.max_drawdown_pct, 2),
                    "lockbox_ruined": r_lock.ruined,
                    "dsr": round(dsr["dsr"], 4), "dsr_sr_star": round(dsr.get("sr_star", 0), 4),
                    "cross_coin": ocoin,
                    "cross_sharpe": round(r_other.sharpe_annualised, 3),
                    "cross_net_pnl": round(r_other.net_pnl, 2),
                    "params": params,
                }
                b = row["best"]
                print(f"[{base_tf:>3} {sym:8} conf={conf_tfs}] {strat:18} "
                      f"IS_Sh={is_sh:+.2f}({is_tr}t) -> LOCK_Sh={b['lockbox_sharpe']:+.2f}"
                      f"({b['lockbox_trades']}t) DSR={b['dsr']:.3f} cross_{ocoin}={b['cross_sharpe']:+.2f}",
                      flush=True)
            else:
                print(f"[{base_tf:>3} {sym:8}] no genome cleared the {MIN_TRADES}-trade floor", flush=True)
            results.append(row)

    out = os.path.join(RESULTS, "sweep_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    _summary(results)
    print(f"\nFull results -> {out}")
    return results


def _summary(results):
    print("\n" + "=" * 104)
    print("TIMEFRAME SWEEP — honest lockbox results (BTC/ETH vs USDC, full history, MTF confirmation)")
    print("=" * 104)
    print(f"{'TF':>4} {'asset':9} {'best strat':18} {'IS_Sh':>6} {'LOCK_Sh':>8} "
          f"{'trades':>6} {'PF':>5} {'DSR':>6} {'cross_Sh':>8} {'maxDD%':>7}")
    print("-" * 104)
    for r in results:
        b = r["best"]
        if not b:
            print(f"{r['tf']:>4} {r['symbol']:9} {'(none)':18}")
            continue
        print(f"{r['tf']:>4} {r['symbol']:9} {b['strategy']:18} {b['is_sharpe']:+6.2f} "
              f"{b['lockbox_sharpe']:+8.2f} {b['lockbox_trades']:>6} "
              f"{(b['lockbox_pf'] or 0):>5.2f} {b['dsr']:>6.3f} {b['cross_sharpe']:+8.2f} {b['lockbox_maxdd']:>7.1f}")
    print("-" * 104)
    cert = [r for r in results if r["best"] and r["best"]["dsr"] >= 0.95]
    print(f"Certifiable (DSR>=0.95): {len(cert)} of {len(results)}")
    print("DSR deflated by per-(tf,asset) search trials. Lockbox = final 365d, sealed (never searched).")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    c = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    sweep(n_samples=n, cores=c)
