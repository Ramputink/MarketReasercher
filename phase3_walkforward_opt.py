"""
phase3_walkforward_opt.py — HONEST walk-forward hyperparameter search (long-only)
=================================================================================

Improves the spot long-only 3-sleeve strategy (trend_ts + breakout +
capitulation) by tuning its INDICATOR hyperparameters — but under the project's
anti-self-deception discipline, not by maximising a backtest number:

  • Search happens ONLY on pre-lockbox data. The sealed 2-year lockbox is NEVER
    read here (no peeking — that read is reserved for a single, deliberate, or
    pre-registered forward evaluation elsewhere).
  • Each candidate is scored by WALK-FORWARD ROBUSTNESS: pre-lockbox is split into
    K contiguous time folds and the candidate's fitness is its WORST-fold Sharpe.
    A config that only works in one sub-period is rejected — we reward edges that
    persist across regimes, exactly like the GA's robust fitness.
  • The winner's Sharpe is DEFLATED (DSR) by the CUMULATIVE number of trials ever
    run (persisted in the checkpoint). So looping this to "search more" RAISES the
    certification bar instead of cherry-picking the luckiest config — more trials,
    harder to beat the deflation benchmark. This is the honest loop.

The risk overlay (vol target + DD breaker) is left FIXED — it is risk control,
not alpha, and must not be fit to the data. Only the sleeve indicators are tuned.

Run:
    python3 phase3_walkforward_opt.py [n_trials] [seed]
Loop it (cloud or local) and the checkpoint accumulates; the deflation bar rises.
"""
from __future__ import annotations

import json
import os
import random
import sys

import numpy as np
import pandas as pd

from phase3_capitulation import load_panel, backtest, metrics, per_year
from phase3_five_strategies import _weights_from, _ema
from phase3_portfolio_overlay import apply_overlay
from benchmark import statistics as st

CKPT = "reports/phase3_wfopt_checkpoint.json"
K_FOLDS = 4
WARMUP_DAYS = 260            # drop leading days so the 200d EMA is defined in fold 1
ANN = 365

# ── search space over INDICATOR hyperparameters (textbook values are inside) ──
SPACE = {
    "trend_ema":   [100, 150, 200, 250],
    "trend_mom":   [30, 45, 60, 90],
    "bo_win":      [30, 40, 50, 60, 80],
    "bo_ema":      [100, 150, 200],
    "bo_hold":     [10, 15, 20],
    "cap_L":       [5, 7, 14],
    "cap_z":       [0.75, 1.0, 1.5],
    "cap_confirm": [3, 5, 8],
    "cap_hold":    [5, 10, 15],
    "cap_regime":  [100, 150, 200],
    "cap_breadth": [0.4, 0.5, 0.6],
    "max_pos":     [8, 10, 12],
}
# the current textbook config (phase3_five_strategies defaults) — the baseline
BASELINE = {"trend_ema": 200, "trend_mom": 60, "bo_win": 50, "bo_ema": 200,
            "bo_hold": 15, "cap_L": 7, "cap_z": 1.0, "cap_confirm": 5,
            "cap_hold": 10, "cap_regime": 150, "cap_breadth": 0.5, "max_pos": 10}


def _sticky(trig, hold):
    held = trig.copy()
    for k in range(1, hold):
        held = held | trig.shift(k, fill_value=False)
    return held


# ── parametrized sleeves (return long-only weight matrices) ──────────────────
def w_trend(close, funding, p):
    mom = close / close.shift(p["trend_mom"]) - 1
    mask = (close > _ema(close, p["trend_ema"])) & close.notna()
    return _weights_from(mask, mom, p["max_pos"])


def w_breakout(close, funding, p):
    hi = close.rolling(p["bo_win"]).max().shift(1)
    trig = (close > hi) & (close > _ema(close, p["bo_ema"])) & close.notna()
    held = _sticky(trig, p["bo_hold"]) & close.notna()
    mom = close / close.shift(60) - 1
    return _weights_from(held, mom, p["max_pos"])


def w_capit(close, funding, p):
    L = p["cap_L"]
    fz = (funding - funding.rolling(L).mean()) / funding.rolling(L).std(ddof=0)
    capit = (fz < -p["cap_z"]) & (close > _ema(close, p["cap_confirm"]))
    held = _sticky(capit, p["cap_hold"])
    btc_on = close["BTC"] > _ema(close["BTC"], p["cap_regime"]) if "BTC" in close else True
    breadth = (close > close.rolling(50).mean()).sum(axis=1) / close.notna().sum(axis=1).replace(0, np.nan)
    reg = (btc_on & (breadth >= p["cap_breadth"])).fillna(False)
    held = (held & close.notna()).mul(reg, axis=0).astype(bool)
    return _weights_from(held, -fz, p["max_pos"])


def ensemble_returns(close, funding, p, mask):
    """Equal-weight raw daily return of the 3 sleeves on `mask` (no overlay)."""
    cw = close[mask]
    ports = []
    for wf in (w_trend, w_breakout, w_capit):
        W = wf(close, funding, p)[mask]
        port, _, _ = backtest(cw, W)
        ports.append(port.reindex(cw.index).fillna(0.0))
    return sum(ports) / len(ports)


def _ann_sharpe(r):
    r = np.asarray(r, dtype=float)
    sd = r.std()
    return float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def walk_forward_fitness(ens_ret: pd.Series):
    """Split the (warmed-up) pre-lockbox series into K folds; return per-fold
    annualised Sharpe and the robust (worst-fold) fitness."""
    s = ens_ret.iloc[WARMUP_DAYS:]
    n = len(s)
    if n < K_FOLDS * 60:
        return None
    bounds = [int(n * i / K_FOLDS) for i in range(K_FOLDS + 1)]
    folds = [_ann_sharpe(s.iloc[bounds[i]:bounds[i + 1]].values) for i in range(K_FOLDS)]
    activity = float((s != 0).mean())
    return {"fold_sharpes": [round(x, 3) for x in folds],
            "worst_fold": round(min(folds), 3),
            "mean_fold": round(float(np.mean(folds)), 3),
            "n_negative_folds": int(sum(1 for x in folds if x < 0)),
            "activity": round(activity, 3)}


def random_config(rng):
    return {k: rng.choice(v) for k, v in SPACE.items()}


def evaluate(close, funding, ins_mask, p):
    ens = ensemble_returns(close, funding, p, ins_mask)
    wf = walk_forward_fitness(ens)
    return ens, wf


def _load_ckpt():
    if os.path.exists(CKPT):
        return json.load(open(CKPT))
    return {"cumulative_trials": 0, "best": None, "history": []}


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    seed_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None

    close, funding, cutoff, man = load_panel()
    ins = close.index < cutoff            # PRE-LOCKBOX ONLY — lockbox never touched
    print(f"panel {man['start']}→{man['end']}, lockbox sealed from {man['lockbox_cutoff_date']}")
    print(f"walk-forward search on PRE-LOCKBOX ({int(ins.sum())} days), {K_FOLDS} folds, "
          f"lockbox NOT read\n")

    ckpt = _load_ckpt()
    base_trials = ckpt["cumulative_trials"]
    # seed varies with cumulative trials so each loop run explores NEW configs
    seed = seed_arg if seed_arg is not None else (20260615 + base_trials)
    rng = random.Random(seed)

    # baseline (textbook) for reference
    _, base_wf = evaluate(close, funding, ins, BASELINE)
    print(f"baseline (textbook) worst-fold Sharpe {base_wf['worst_fold']} "
          f"folds {base_wf['fold_sharpes']}")

    best = ckpt["best"]
    evaluated = 0
    for i in range(n_trials):
        p = random_config(rng)
        ens, wf = evaluate(close, funding, ins, p)
        if wf is None or wf["activity"] < 0.05:
            continue
        evaluated += 1
        cand = {"config": p, "wf": wf, "is_daily_sharpe": round(_ann_sharpe(ens.values), 3)}
        if best is None or wf["worst_fold"] > best["wf"]["worst_fold"]:
            cand["ens_pre_returns"] = None      # not stored; recomputed for DSR below
            best = cand
            print(f"  [{i+1}/{n_trials}] NEW BEST worst-fold {wf['worst_fold']}  "
                  f"folds {wf['fold_sharpes']}  {p}")

    cumulative = base_trials + evaluated
    # DSR of the best config on full pre-lockbox daily returns, deflated by the
    # CUMULATIVE trials ever run (honest multiple-testing penalty).
    best_ens, _ = evaluate(close, funding, ins, best["config"])
    pre = best_ens.iloc[WARMUP_DAYS:].values
    dsr = st.deflated_sharpe_ratio(pre, n_trials=max(cumulative, 1))
    best["dsr_vs_cumulative_trials"] = round(float(dsr["dsr"]), 4)
    best["sr_star"] = round(float(dsr["sr_star"]), 4)
    best["cumulative_trials_at_eval"] = cumulative

    improved = (base_wf["worst_fold"] is not None and
                best["wf"]["worst_fold"] > base_wf["worst_fold"])
    run_summary = {
        "run_seed": seed, "evaluated_this_run": evaluated,
        "cumulative_trials": cumulative,
        "best_worst_fold": best["wf"]["worst_fold"],
        "baseline_worst_fold": base_wf["worst_fold"],
        "beats_baseline": bool(improved),
        "dsr": best["dsr_vs_cumulative_trials"],
    }
    ckpt["cumulative_trials"] = cumulative
    ckpt["best"] = best
    ckpt["baseline_wf"] = base_wf
    ckpt["history"].append(run_summary)
    os.makedirs("reports", exist_ok=True)
    json.dump(ckpt, open(CKPT, "w"), indent=2, default=str)

    print("\n" + "=" * 72)
    print("WALK-FORWARD SEARCH RESULT (pre-lockbox only; lockbox sealed)")
    print("=" * 72)
    print(f"  evaluated this run : {evaluated}   cumulative trials : {cumulative}")
    print(f"  baseline worst-fold: {base_wf['worst_fold']}")
    print(f"  BEST     worst-fold: {best['wf']['worst_fold']}  folds {best['wf']['fold_sharpes']}")
    print(f"  beats baseline     : {improved}")
    print(f"  DSR (deflated by {cumulative} cumulative trials): {best['dsr_vs_cumulative_trials']}  "
          f"(sr* {best['sr_star']})")
    print(f"  best config        : {best['config']}")
    print(f"\n  NOTE: the sealed lockbox was NOT read. To turn a robust winner into a")
    print(f"  deployable claim, pre-register it and forward-test (forward_test/), or")
    print(f"  spend ONE deliberate sealed-lockbox read — never optimise against it.")
    print(f"\ncheckpoint → {CKPT}")
    return ckpt


if __name__ == "__main__":
    main()
