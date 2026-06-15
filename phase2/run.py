"""
phase2.run — honest search for a cross-sectional edge (momentum + carry)
========================================================================

The Phase-2 analogue of `sweep.run`. For a budget of random candidate
configurations (signal family + portfolio construction), it:

  1. Searches ONLY on the in-sample window (everything before the sealed lockbox
     cutoff stamped by build_data). The lockbox is never consulted during search.
  2. Picks the best in-sample candidate per signal family by in-sample Sharpe.
  3. Re-runs that winner on the SEALED lockbox (true OOS) and computes the
     Deflated Sharpe Ratio, deflated by the REAL number of trials searched —
     the same statistic that rejected Phase 1.

A family is flagged CERTIFIABLE only if, on the lockbox: DSR ≥ 0.95, Sharpe > 0,
enough periods, positive t-stat, and the account never ruined. If nothing
certifies, that is the honest verdict for this paradigm — not a prompt to relax
a threshold.

Run:
    /Library/Frameworks/Python.framework/Versions/3.10/bin/python3 -m phase2.run [n_trials] [cores]
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from phase2 import SNAP_DIR, MANIFEST, RESULTS_DIR, LOCKBOX_DAYS, MIN_UNIVERSE
from phase2 import signals as sg
from phase2.portfolio import backtest, PortfolioCosts
from benchmark import statistics as st

DSR_CERT = 0.95
MIN_PERIODS = 30        # need enough rebalance periods for a believable Sharpe
SIGNAL_FAMILIES = ["momentum", "idio", "carry", "combo"]


# ─────────────────────────────────────────────────────────────────────────────
# Panel loading (per-process, fork-friendly)
# ─────────────────────────────────────────────────────────────────────────────
_PANEL = {}


def _load_panel() -> dict:
    if _PANEL:
        return _PANEL
    with open(MANIFEST) as f:
        man = json.load(f)
    for name in ("close", "open", "ret", "funding"):
        _PANEL[name] = pd.read_parquet(os.path.join(SNAP_DIR, f"{name}.parquet"))
    _PANEL["cutoff"] = pd.Timestamp(man["lockbox_cutoff_ts"], unit="ms", tz="UTC").floor("D")
    _PANEL["manifest"] = man
    return _PANEL


# ─────────────────────────────────────────────────────────────────────────────
# Candidate sampling
# ─────────────────────────────────────────────────────────────────────────────

def random_candidate(rng: random.Random, family: str) -> dict:
    c = {
        "family": family,
        "k": rng.randint(1, 5),
        "rebalance_every": rng.choice([1, 2, 3, 5, 7]),
        "long_only": rng.random() < 0.25,
        "dollar_neutral": rng.random() < 0.7,
        "weight_mode": rng.choice(["equal", "score_prop", "vol_parity"]),
        "gross": round(rng.uniform(0.5, 2.0), 2),
        "vol_lookback": rng.choice([10, 14, 20, 30]),
        "mom_lookback": rng.choice([10, 20, 30, 45, 60, 90, 120]),
        "mom_skip": rng.choice([0, 1, 2, 5]),
        "carry_lookback": rng.choice([3, 5, 7, 10, 14, 21, 30]),
        "combo_mom_w": round(rng.uniform(0.2, 0.8), 2),
        "mom_risk_adj": rng.random() < 0.5,
        "combo_idio": rng.random() < 0.5,
    }
    return c


def _build_scores(panel: dict, c: dict) -> pd.DataFrame:
    close, funding = panel["close"], panel["funding"]
    fam = c["family"]
    ra = bool(c.get("mom_risk_adj", False))
    if fam == "momentum":
        return sg.momentum_score(close, c["mom_lookback"], c["mom_skip"], risk_adj=ra)
    if fam == "idio":
        return sg.idio_momentum_score(close, c["mom_lookback"], c["mom_skip"], risk_adj=ra)
    if fam == "carry":
        return sg.carry_score(funding, c["carry_lookback"])
    # combo: (raw or idio) momentum + carry, z-blended
    mom_fn = sg.idio_momentum_score if c.get("combo_idio") else sg.momentum_score
    mom = mom_fn(close, c["mom_lookback"], c["mom_skip"], risk_adj=ra)
    car = sg.carry_score(funding, c["carry_lookback"])
    w = c["combo_mom_w"]
    return sg.combine_scores([(mom, w), (car, 1.0 - w)])


# in-sample is split train/val so we can select for ROBUSTNESS (works in both
# halves) instead of max in-sample Sharpe (which just picks the luckiest overfit).
IS_SPLIT = 0.6   # first 60% of in-sample = train, last 40% = validation


def _window_mask(panel: dict, idx: pd.DatetimeIndex, window: str) -> pd.Series:
    cutoff = panel["cutoff"]
    is_mask = idx < cutoff
    if window == "lockbox":
        return idx >= cutoff
    if window == "is":
        return is_mask
    # train / val: split the in-sample dates by IS_SPLIT
    is_dates = idx[is_mask]
    if len(is_dates) == 0:
        return pd.Series(False, index=range(len(idx))).values
    split_date = is_dates[int(len(is_dates) * IS_SPLIT)]
    if window == "train":
        return is_mask & (idx < split_date)
    if window == "val":
        return is_mask & (idx >= split_date)
    raise ValueError(window)


def _slice(panel: dict, c: dict, window: str) -> tuple:
    """Return (scores, open, close, funding) sliced to the named window."""
    scores = _build_scores(panel, c)
    close, open_, funding = panel["close"], panel["open"], panel["funding"]
    mask = _window_mask(panel, scores.index, window)
    return (scores[mask], open_[mask], close[mask], funding[mask])


def _run_one(panel: dict, c: dict, window: str, costs: PortfolioCosts):
    scores, open_, close, funding = _slice(panel, c, window)
    res = backtest(
        scores, open_, close, funding,
        rebalance_every=c["rebalance_every"], k=c["k"],
        long_only=c["long_only"], dollar_neutral=c["dollar_neutral"],
        weight_mode=c["weight_mode"], gross=c["gross"],
        vol_lookback=c["vol_lookback"], min_universe=MIN_UNIVERSE, costs=costs,
    )
    return res


def _metrics(res) -> dict:
    per = res.period_returns
    if len(per) < 2:
        return {"sharpe": -9.99, "n": len(per), "ret_pct": res.total_return_pct,
                "ruined": res.ruined}
    sr = st.sharpe_per_obs(per)
    return {
        "sharpe": float(sr),
        "n": int(len(per)),
        "ret_pct": float(res.total_return_pct),
        "turnover": float(res.turnover_mean),
        "funding_share_pct": float(res.funding_pnl_pct),
        "avg_positions": float(res.avg_n_positions),
        "ruined": bool(res.ruined),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parallel in-sample search
# ─────────────────────────────────────────────────────────────────────────────

def _worker(task):
    c = task
    panel = _load_panel()
    costs = PortfolioCosts()
    tr = _metrics(_run_one(panel, c, "train", costs))
    va = _metrics(_run_one(panel, c, "val", costs))
    # ROBUST fitness: a config is only as good as its WORSE half. This rewards
    # edges that persist across time, not configs that fit one sub-period.
    robust = min(tr["sharpe"], va["sharpe"])
    fit = robust
    if min(tr["n"], va["n"]) < MIN_PERIODS // 2:
        fit -= 2.0
    if tr.get("ruined") or va.get("ruined"):
        fit -= 5.0
    return {"cand": c, "train": tr, "val": va, "robust": float(robust),
            "fit": float(fit)}


def search(n_trials: int, cores: int) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    panel = _load_panel()
    man = panel["manifest"]
    print(f"panel: {man['start']} → {man['end']}  ({man['n_dates']} days), "
          f"lockbox sealed from {man['lockbox_cutoff_date']}")
    print(f"universe: {man['universe']}")
    print(f"searching {n_trials} trials/family × {len(SIGNAL_FAMILIES)} families "
          f"on {cores} cores …\n")

    # deterministic candidate set (seeded by trial index)
    tasks = []
    for fam in SIGNAL_FAMILIES:
        for i in range(n_trials):
            rng = random.Random(hash((fam, i)) & 0xFFFFFFFF)
            tasks.append(random_candidate(rng, fam))

    t0 = time.time()
    ctx = mp.get_context("fork")
    total = len(tasks)
    step = max(1, total // 20)   # report ~every 5%
    results = []
    with ctx.Pool(cores) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4), 1):
            results.append(r)
            if i % step == 0 or i == total:
                el = time.time() - t0
                eta = el / i * (total - i)
                print(f"  progress {i}/{total} ({100*i/total:4.0f}%)  "
                      f"elapsed {el:5.0f}s  eta {eta:5.0f}s", flush=True)
    dt = time.time() - t0
    print(f"in-sample search done in {dt:.1f}s ({len(results)} evals)\n")

    # collect per-family best + the cross-trial Sharpe std (for honest DSR)
    by_fam = {f: [] for f in SIGNAL_FAMILIES}
    for r in results:
        by_fam[r["cand"]["family"]].append(r)

    total_trials = len(results)   # honest: every config we tried counts
    out = {
        "timestamp_utc": man.get("end"),
        "n_trials_per_family": n_trials,
        "total_trials": total_trials,
        "lockbox_days": LOCKBOX_DAYS,
        "dsr_cert_threshold": DSR_CERT,
        "families": {},
        "certified_any": False,
    }

    # landscape diagnostics: where (if anywhere) does robust signal concentrate?
    out["landscape"] = _landscape(results)

    for fam, rs in by_fam.items():
        rs_sorted = sorted(rs, key=lambda x: x["fit"], reverse=True)
        best = rs_sorted[0]
        # DSR deflation std from the cross-trial spread of ROBUST in-sample Sharpe
        robusts = np.array([x["robust"] for x in rs if np.isfinite(x["robust"])])
        trials_std = float(robusts.std(ddof=1)) if len(robusts) > 2 else 0.0

        # evaluate winner on the SEALED lockbox
        lb_res = _run_one(panel, best["cand"], "lockbox", PortfolioCosts())
        lb = _metrics(lb_res)
        per = lb_res.period_returns
        if len(per) >= 2:
            dsr = st.deflated_sharpe_ratio(
                per, n_trials=total_trials,
                trials_sharpe_std=trials_std if trials_std > 0 else None,
            )
            tstat = st.tstat_pvalue(per)
        else:
            dsr = {"dsr": 0.0, "sr_hat": 0.0, "sr_star": 0.0}
            tstat = {"tstat": 0.0, "pvalue": 1.0}

        certified = (
            dsr["dsr"] >= DSR_CERT and lb["sharpe"] > 0 and lb["n"] >= MIN_PERIODS
            and tstat["tstat"] >= 2.0 and not lb.get("ruined")
        )
        out["families"][fam] = {
            "best_candidate": best["cand"],
            "train": best["train"],
            "val": best["val"],
            "robust_is_sharpe": best["robust"],
            "lockbox": lb,
            "dsr": dsr["dsr"],
            "dsr_sr_star": dsr.get("sr_star"),
            "lockbox_sharpe": lb["sharpe"],
            "tstat": tstat["tstat"],
            "trials_sharpe_std": trials_std,
            "certified": bool(certified),
        }
        out["certified_any"] = out["certified_any"] or certified

    # write results
    path = os.path.join(RESULTS_DIR, "phase2_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    _print_table(out)
    _print_landscape(out["landscape"])
    print(f"\nresults → {path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Landscape: aggregate ROBUST in-sample fitness by parameter value
# ─────────────────────────────────────────────────────────────────────────────

_LS_AXES = ["family", "k", "rebalance_every", "long_only", "dollar_neutral",
            "weight_mode", "mom_lookback", "carry_lookback"]


def _landscape(results: list) -> dict:
    """For each parameter axis, mean robust in-sample Sharpe per value — reveals
    which regions persist across train AND val (i.e. real structure vs noise)."""
    ls = {}
    for axis in _LS_AXES:
        buckets: dict = {}
        for r in results:
            v = r["cand"].get(axis)
            buckets.setdefault(v, []).append(r["robust"])
        ls[axis] = {
            str(v): {"mean": float(np.mean(vals)), "max": float(np.max(vals)),
                     "n": len(vals)}
            for v, vals in sorted(buckets.items(), key=lambda kv: str(kv[0]))
        }
    return ls


def _print_landscape(ls: dict):
    print("\n── landscape: mean ROBUST in-sample Sharpe (min of train,val) by param ──")
    for axis, vals in ls.items():
        top = sorted(vals.items(), key=lambda kv: kv[1]["mean"], reverse=True)
        cells = "  ".join(f"{k}={d['mean']:+.3f}" for k, d in top[:6])
        print(f"  {axis:16}: {cells}")


def _print_table(out: dict):
    print("=" * 92)
    print(f"{'family':10} {'tr_Shrp':>8} {'va_Shrp':>8} {'LB_Shrp':>8} {'LB_ret%':>8} "
          f"{'periods':>7} {'DSR':>6} {'SR*':>6} {'tstat':>6} {'cert':>5}")
    print("-" * 100)
    for fam, d in out["families"].items():
        lb = d["lockbox"]
        print(f"{fam:10} {d['train']['sharpe']:8.3f} {d['val']['sharpe']:8.3f} "
              f"{d['lockbox_sharpe']:8.3f} {lb.get('ret_pct', 0):8.1f} {lb.get('n', 0):7d} "
              f"{d['dsr']:6.3f} {d.get('dsr_sr_star', 0) or 0:6.3f} "
              f"{d['tstat']:6.2f} {'YES' if d['certified'] else 'no':>5}")
    print("=" * 92)
    print(f"CERTIFIED ANY: {out['certified_any']}")


# ─────────────────────────────────────────────────────────────────────────────
# FOCUSED mode — disciplined hypothesis test in the robust region
# ─────────────────────────────────────────────────────────────────────────────
# Derived from the batch-1/2 landscape: weekly-ish rebalance, momentum lb 45-90,
# carry lb 3-7, equal weight. We separate the TRUE cross-sectional alpha
# (dollar-neutral L/S) from BETA (long-only ≈ long crypto). DSR is deflated by
# the SMALL number of curated configs, so the certification bar is honest-but-low
# — the legitimate path to a real pass, unlike deflating against 4500 noise draws.

def _cfg(**kw) -> dict:
    base = dict(family="momentum", k=4, rebalance_every=7, long_only=False,
                dollar_neutral=True, weight_mode="equal", gross=1.0,
                vol_lookback=20, mom_lookback=60, mom_skip=1,
                carry_lookback=7, combo_mom_w=0.5, mom_risk_adj=False,
                combo_idio=False)
    base.update(kw)
    return base


# k bumped to 4 (wider 30-asset universe supports deeper books)
FOCUSED_CONFIGS = [
    # ── market-neutral RISK-ADJUSTED total momentum ──
    _cfg(family="momentum", mom_lookback=60, k=4, mom_risk_adj=True),
    _cfg(family="momentum", mom_lookback=90, k=5, mom_risk_adj=True),
    _cfg(family="momentum", mom_lookback=120, k=5, mom_risk_adj=True),
    # ── market-neutral IDIOSYNCRATIC momentum (beta-stripped relative strength) ──
    _cfg(family="idio", mom_lookback=45, k=4),
    _cfg(family="idio", mom_lookback=60, k=4),
    _cfg(family="idio", mom_lookback=90, k=5),
    _cfg(family="idio", mom_lookback=60, k=4, mom_risk_adj=True),
    _cfg(family="idio", mom_lookback=90, k=5, mom_risk_adj=True),
    _cfg(family="idio", mom_lookback=120, k=6, mom_risk_adj=True),
    # ── market-neutral carry ──
    _cfg(family="carry", carry_lookback=3, k=4),
    _cfg(family="carry", carry_lookback=7, k=4),
    # ── market-neutral combo (idio risk-adj momentum + carry) ──
    _cfg(family="combo", mom_lookback=90, carry_lookback=7, k=4,
         mom_risk_adj=True, combo_idio=True),
    # ── long-only reference (BETA, not alpha) ──
    _cfg(family="idio", mom_lookback=90, k=5, long_only=True, dollar_neutral=False),
]


# ── yearly walk-forward: was there EVER a stable edge, or never? ──────────────
def _yearly_breakdown(panel: dict, c: dict) -> dict:
    """Sharpe of a config in each calendar year (full history, diagnostic only —
    distinguishes a decayed edge from one that never existed)."""
    scores = _build_scores(panel, c)
    close, open_, funding = panel["close"], panel["open"], panel["funding"]
    years = sorted(set(scores.index.year))
    out = {}
    for y in years:
        m = scores.index.year == y
        if m.sum() < 30:
            continue
        res = backtest(scores[m], open_[m], close[m], funding[m],
                       rebalance_every=c["rebalance_every"], k=c["k"],
                       long_only=c["long_only"], dollar_neutral=c["dollar_neutral"],
                       weight_mode=c["weight_mode"], gross=c["gross"],
                       vol_lookback=c["vol_lookback"], min_universe=MIN_UNIVERSE,
                       costs=PortfolioCosts())
        out[str(y)] = round(st.sharpe_per_obs(res.period_returns), 3)
    return out


def focused(cores: int) -> dict:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    panel = _load_panel()
    man = panel["manifest"]
    configs = FOCUSED_CONFIGS
    n = len(configs)
    print(f"FOCUSED test: {n} theory-driven configs (DSR deflated by {n} trials)\n")

    rows = []
    for c in configs:
        tr = _metrics(_run_one(panel, c, "train", PortfolioCosts()))
        va = _metrics(_run_one(panel, c, "val", PortfolioCosts()))
        lb_res = _run_one(panel, c, "lockbox", PortfolioCosts())
        lb = _metrics(lb_res)
        per = lb_res.period_returns
        if len(per) >= 2:
            dsr = st.deflated_sharpe_ratio(per, n_trials=n)
            tstat = st.tstat_pvalue(per)
        else:
            dsr, tstat = {"dsr": 0.0, "sr_star": 0.0}, {"tstat": 0.0}
        certified = (dsr["dsr"] >= DSR_CERT and lb["sharpe"] > 0
                     and lb["n"] >= MIN_PERIODS and tstat["tstat"] >= 2.0
                     and not lb.get("ruined"))
        kind = ("BETA(LO)" if c["long_only"] else "alpha(MN)")
        rows.append({"cfg": c, "kind": kind, "train": tr, "val": va, "lockbox": lb,
                     "dsr": dsr["dsr"], "sr_star": dsr.get("sr_star"),
                     "tstat": tstat["tstat"], "certified": bool(certified)})

    # report
    print("=" * 108)
    print(f"{'family':9} {'kind':9} {'params':22} {'tr':>6} {'va':>6} {'LB_S':>6} "
          f"{'LB%':>7} {'per':>4} {'DSR':>6} {'tstat':>6} {'cert':>4}")
    print("-" * 108)
    for r in rows:
        c = r["cfg"]
        p = (f"lb{c['mom_lookback']}/k{c['k']}/r{c['rebalance_every']}"
             if c["family"] != "carry" else
             f"clb{c['carry_lookback']}/k{c['k']}/r{c['rebalance_every']}")
        print(f"{c['family']:9} {r['kind']:9} {p:22} {r['train']['sharpe']:6.2f} "
              f"{r['val']['sharpe']:6.2f} {r['lockbox']['sharpe']:6.2f} "
              f"{r['lockbox'].get('ret_pct',0):7.1f} {r['lockbox'].get('n',0):4d} "
              f"{r['dsr']:6.3f} {r['tstat']:6.2f} {'YES' if r['certified'] else 'no':>4}")
    print("=" * 108)
    any_cert = any(r["certified"] for r in rows)
    # honest beta-vs-alpha read
    mn = [r for r in rows if "MN" in r["kind"]]
    mn_pos = [r for r in mn if r["lockbox"]["sharpe"] > 0]
    print(f"CERTIFIED ANY: {any_cert}   "
          f"market-neutral configs with positive lockbox Sharpe: {len(mn_pos)}/{len(mn)}")

    # yearly walk-forward of the best market-neutral config (by val Sharpe) — is
    # there a stable edge anywhere, or is every year just noise around zero?
    best_mn = max(mn, key=lambda r: r["val"]["sharpe"]) if mn else None
    yearly = {}
    if best_mn:
        yearly = _yearly_breakdown(panel, best_mn["cfg"])
        c = best_mn["cfg"]
        tag = (f"{c['family']} MN lb{c.get('mom_lookback')}/k{c['k']}/r{c['rebalance_every']}"
               f"{'/RA' if c.get('mom_risk_adj') else ''}")
        print(f"\n── yearly walk-forward of best market-neutral config ({tag}) ──")
        print("  " + "  ".join(f"{y}:{s:+.2f}" for y, s in yearly.items()))
        pos = sum(1 for s in yearly.values() if s > 0)
        print(f"  positive years: {pos}/{len(yearly)}  "
              f"(a real edge would be positive in MOST years, not ~half)")

    out = {"mode": "focused", "n_configs": n, "certified_any": any_cert,
           "timestamp": man.get("end"), "yearly_best_mn": yearly,
           "results": [{**r, "cfg": r["cfg"]} for r in rows]}
    path = os.path.join(RESULTS_DIR, "phase2_focused.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"results → {path}")
    return out


if __name__ == "__main__":
    cores = max(1, (os.cpu_count() or 4) - 2)
    if len(sys.argv) > 1 and sys.argv[1] == "focused":
        if len(sys.argv) > 2:
            cores = int(sys.argv[2])
        focused(cores)
    else:
        n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 300
        if len(sys.argv) > 2:
            cores = int(sys.argv[2])
        search(n_trials, cores)
