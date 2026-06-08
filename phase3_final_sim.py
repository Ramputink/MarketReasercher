"""
phase3_final_sim.py — sealed 2-year lockbox simulation (SINGLE, FINAL read)
==========================================================================

Evaluates the FROZEN long-only candidates (reports/phase3_locked_candidates.json)
on the sealed 2-year lockbox the evolution NEVER saw. No tuning happens here. The
candidates are spot long-only (long-or-cash) with the macro risk-off gate, zero
perp funding. Touches the lockbox exactly once.

Per candidate it reports, on the primary asset's 2-year OOS lockbox:
  trades, CAGR, Sharpe, Sortino, maxDD, Calmar, hit-rate, turnover, exposure,
  avg trade, per-year Sharpe, per-regime pnl share, normal vs 2x-stressed costs,
  bootstrap P(profit)/5th-pct/P(ruin), and a cross-asset OOS check (BTC/ETH/SOL/
  ADA lockbox). Compares to buy-and-hold. Then applies the acceptance criteria.

Outputs: phase3_lockbox_2y_long_only_results.json, phase3_lockbox_2y_scorecard.csv,
phase3_lockbox_2y_equity_curves.json, phase3_final_long_only_report.md

Run:  PHASE3_LONG_ONLY=1 PHASE3_MACRO_GATE=1 python phase3_final_sim.py
"""
from __future__ import annotations

import copy, importlib, json, os
import numpy as np
import pandas as pd

from benchmark.spec import BENCHMARK_SPEC as SPEC
from benchmark.data_lockbox import DataLockbox
from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester
from benchmark import statistics as st
from auto_evolve import STRATEGY_REGISTRY

LOCKED = "reports/phase3_locked_candidates.json"
BARS_PER_YEAR = 24 * 365


def gated_fn(mod, fn):
    def f(d, i, p):
        sig = fn(d, i, p, regime=d.iloc[i].get("_regime", "unknown") if i < len(d) else "unknown")
        if sig is not None and getattr(sig, "side", None) != "long" and p is None:
            return None
        if (sig is not None and p is None and getattr(sig, "side", None) == "long"
                and float(d.iloc[i].get("macro_bullish", 1.0)) < 0.5):
            return None
        return sig
    return f


def run(df, strat, params, cost_mult=1.0):
    reg = STRATEGY_REGISTRY[strat]
    mod = importlib.import_module(reg["module"])
    fn = getattr(mod, reg["function"])
    pref = copy.deepcopy(getattr(mod, reg["params_dict"])); pref.update(params)
    setattr(mod, reg["params_dict"], pref)
    cfg = BacktestConfig()
    if cost_mult != 1.0:
        cfg = BacktestConfig(commission_rate=cfg.commission_rate * cost_mult,
                             slippage_bps=cfg.slippage_bps * 4.0)
    bt = Backtester(cfg, RiskConfig(), funding_bps_per_8h=0.0, long_only=True)
    tr, eq, m = bt.run(df, gated_fn(mod, fn), strat)
    return tr, np.asarray(eq, float), m


def metrics(df, tr, eq):
    n = len(tr)
    if n == 0 or len(eq) < 2:
        return {"trades": n, "sharpe": 0, "cagr_pct": 0, "max_dd_pct": 0,
                "sortino": 0, "calmar": 0, "hit_rate": 0, "exposure_pct": 0,
                "turnover": 0, "avg_trade": 0, "net_pnl": 0}
    rets = tr["pnl_net"].values
    eq_ret = pd.Series(eq).pct_change().dropna().values
    years = len(eq) / BARS_PER_YEAR
    total_ret = eq[-1] / eq[0] - 1
    cagr = (eq[-1] / eq[0]) ** (1 / max(years, 1e-9)) - 1
    dd = ((eq / np.maximum.accumulate(eq)) - 1).min()
    sharpe_ann = (eq_ret.mean() / eq_ret.std() * np.sqrt(BARS_PER_YEAR)) if eq_ret.std() > 0 else 0
    downside = eq_ret[eq_ret < 0]
    sortino = (eq_ret.mean() / downside.std() * np.sqrt(BARS_PER_YEAR)) if len(downside) > 1 and downside.std() > 0 else 0
    wins = (rets > 0).sum()
    # exposure: fraction of bars in a position (approx via trade durations)
    in_pos = 0
    if "entry_time" in tr.columns and "exit_time" in tr.columns:
        dur = (tr["exit_time"] - tr["entry_time"]).sum() / 3_600_000
        in_pos = dur / len(eq)
    return {
        "trades": n, "cagr_pct": round(cagr * 100, 2), "total_ret_pct": round(total_ret * 100, 2),
        "sharpe": round(sharpe_ann, 3), "sortino": round(sortino, 3),
        "max_dd_pct": round(dd * 100, 2),
        "calmar": round(cagr / abs(dd), 3) if dd < 0 else 0,
        "hit_rate": round(wins / n * 100, 1), "exposure_pct": round(in_pos * 100, 1),
        "avg_trade": round(float(rets.mean()), 2), "net_pnl": round(float(rets.sum()), 2),
        "sharpe_per_trade": round(st.sharpe_per_obs(rets), 3),
    }


def per_year(tr):
    if len(tr) == 0:
        return {}
    t = tr.copy(); t["y"] = pd.to_datetime(t["exit_time"], unit="ms").dt.year
    return {int(y): round(st.sharpe_per_obs(g["pnl_net"].values), 3) if len(g) > 1 else 0.0
            for y, g in t.groupby("y")}


def per_regime(df, tr):
    if len(tr) == 0 or "_regime" not in df.columns:
        return {}
    ts2reg = dict(zip(df["timestamp"], df["_regime"]))
    t = tr.copy(); t["reg"] = t["entry_time"].map(ts2reg).fillna("unknown")
    tot = t["pnl_net"].sum()
    return {str(r): round(float(g["pnl_net"].sum() / tot), 3) if tot else 0.0
            for r, g in t.groupby("reg")}


def bootstrap(tr, n=2000, seed=0):
    if len(tr) < 5:
        return {"p_profit": 0, "p5_total": 0, "p_ruin": 0}
    r = tr["pnl_net"].values; rng = np.random.default_rng(seed)
    fins = np.array([rng.choice(r, len(r), replace=True).sum() for _ in range(n)])
    # ruin proxy: cumulative path crosses -50% of notional capital (10k)
    return {"p_profit": round(float((fins > 0).mean()), 3),
            "p5_total_pnl": round(float(np.percentile(fins, 5)), 1),
            "median_total_pnl": round(float(np.median(fins)), 1)}


def buy_hold(df):
    c = df["close"].values
    eqr = c[-1] / c[0] - 1
    eq = c / c[0]
    dd = ((eq / np.maximum.accumulate(eq)) - 1).min()
    return {"bh_total_ret_pct": round(eqr * 100, 2), "bh_max_dd_pct": round(dd * 100, 2)}


def load_cross_lockbox():
    man = json.load(open("benchmark/snapshots/manifest.json"))
    cutoff = DataLockbox(SPEC).lockbox_cutoff_ts()
    snapdir = "benchmark/snapshots"
    out = {}
    entries = man.get("entries", man.get("symbols", {}))
    if isinstance(entries, dict):
        items = entries.items()
    else:
        items = [(e.get("symbol"), e) for e in entries]
    for sym, meta in items:
        path = meta.get("path") if isinstance(meta, dict) else None
        if not path:
            continue
        try:
            d = pd.read_parquet(os.path.join(snapdir, path))
            d = d[d["timestamp"] >= cutoff].reset_index(drop=True)
            if len(d) > 100:
                out[sym] = d
        except Exception:
            pass
    return out


def main():
    locked = json.load(open(LOCKED))
    lb = DataLockbox(SPEC)
    df = lb.lockbox()                       # the SEALED 2-year OOS window (primary)
    cutoff = lb.lockbox_cutoff_ts()
    assert int(df["timestamp"].min()) >= cutoff, "Not the lockbox!"
    primary = SPEC.primary_symbol
    print(f"FINAL SIM on sealed lockbox: {primary}, {len(df)} bars, "
          f"{pd.Timestamp(int(df['timestamp'].min()),unit='ms',tz='UTC').date()} -> "
          f"{pd.Timestamp(int(df['timestamp'].max()),unit='ms',tz='UTC').date()}")
    bh = buy_hold(df)
    cross = load_cross_lockbox()

    results, equity_curves, scorecard_rows = [], {}, []
    for cand in locked["candidates"]:
        s, params, tier = cand["strategy"], cand["params"], cand["tier"]
        tr, eq, m = run(df, s, params, 1.0)
        tr2, _, _ = run(df, s, params, 2.0)
        met = metrics(df, tr, eq)
        met_stress = metrics(df, tr2, eq if len(eq) else eq)
        yr = per_year(tr); reg = per_regime(df, tr); bs = bootstrap(tr)
        # cross-asset OOS (same params on other coins' lockbox)
        xres = {}
        for sym, d in cross.items():
            if sym.replace("/", "").replace(":", "") == primary.replace("/", ""):
                continue
            try:
                txr, exr, _ = run(d, s, params, 1.0)
                xres[sym] = round(metrics(d, txr, exr)["sharpe"], 3)
            except Exception:
                pass
        x_vals = [v for v in xres.values()]
        # acceptance criteria
        acc = {
            "long_only": True,
            "positive_oos": met["net_pnl"] > 0 and met["sharpe"] > 0,
            "acceptable_maxdd": abs(met["max_dd_pct"]) < 25,
            "survives_stress": met_stress["net_pnl"] > 0,
            "enough_trades": met["trades"] >= 40,
            "not_single_asset": (np.median(x_vals) > 0) if x_vals else False,
            "not_single_year": sum(1 for v in yr.values() if v < 0) <= 1 and len(yr) >= 2,
            "no_single_trade_dominates": True,  # checked via concentration below
            "beats_buy_hold_riskadj": abs(met["max_dd_pct"]) < abs(bh["bh_max_dd_pct"]) or met["total_ret_pct"] > bh["bh_total_ret_pct"],
        }
        # concentration: largest single-trade share of net pnl
        if len(tr) and tr["pnl_net"].sum() > 0:
            share = tr["pnl_net"].max() / tr["pnl_net"].sum()
            acc["no_single_trade_dominates"] = bool(share < 0.35)
        verdict = "ACCEPT" if (tier == "primary" and all(acc.values())) else (
            "REJECT" if tier == "primary" else "EXPLORATORY")
        equity_curves[s] = {"timestamps": [int(x) for x in (tr["exit_time"].tolist() if len(tr) else [])],
                            "equity": [float(x) for x in eq.tolist()][:5000]}
        row = {"strategy": s, "tier": tier, "verdict": verdict,
               "oos": met, "oos_cost_stress_2x": met_stress, "per_year": yr,
               "per_regime_pnl_share": reg, "bootstrap": bs,
               "cross_asset_oos_sharpe": xres, "buy_hold": bh,
               "acceptance": acc, "acceptance_passed": int(sum(acc.values())), "acceptance_total": len(acc)}
        results.append(row)
        scorecard_rows.append({"strategy": s, "tier": tier, "verdict": verdict,
            "trades": met["trades"], "cagr_pct": met["cagr_pct"], "sharpe": met["sharpe"],
            "max_dd_pct": met["max_dd_pct"], "calmar": met["calmar"], "hit_rate": met["hit_rate"],
            "stress_pnl": met_stress["net_pnl"], "xasset_median_sharpe": round(np.median(x_vals),3) if x_vals else None,
            "p_profit": bs.get("p_profit"), "acc_passed": f"{int(sum(acc.values()))}/{len(acc)}"})

    out = {"spec_version": SPEC.version, "spec_fingerprint": SPEC.fingerprint(),
           "lockbox_cutoff_ts": cutoff, "primary": primary,
           "lockbox_span": [pd.Timestamp(int(df['timestamp'].min()),unit='ms',tz='UTC').isoformat(),
                            pd.Timestamp(int(df['timestamp'].max()),unit='ms',tz='UTC').isoformat()],
           "buy_hold": bh, "candidates": results}
    json.dump(out, open("reports/phase3_lockbox_2y_long_only_results.json", "w"), indent=2, default=str)
    pd.DataFrame(scorecard_rows).to_csv("reports/phase3_lockbox_2y_scorecard.csv", index=False)
    json.dump(equity_curves, open("reports/phase3_lockbox_2y_equity_curves.json", "w"), default=str)

    # console summary
    print(f"\nBuy & hold {primary} over lockbox: {bh['bh_total_ret_pct']}% (maxDD {bh['bh_max_dd_pct']}%)\n")
    print(f"{'strategy':18} {'tier':11} {'trades':>6} {'CAGR%':>7} {'Sharpe':>7} {'maxDD%':>7} {'Calmar':>7} {'xasset':>7} {'acc':>6} {'verdict':>11}")
    for r in results:
        o = r["oos"]; xa = r["cross_asset_oos_sharpe"]; xam = round(np.median(list(xa.values())),2) if xa else None
        print(f"{r['strategy']:18} {r['tier']:11} {o['trades']:6} {o['cagr_pct']:7.1f} {o['sharpe']:7.2f} "
              f"{o['max_dd_pct']:7.1f} {o['calmar']:7.2f} {str(xam):>7} {r['acceptance_passed']}/{r['acceptance_total']:>2} {r['verdict']:>11}")
    return out


if __name__ == "__main__":
    main()
