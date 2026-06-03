"""
honest_eval.py — Liquidation-aware, out-of-sample honest return evaluator.

WHY THIS EXISTS
---------------
The core backtester (engine/backtester.py) is intentionally frozen and fills stop
losses EXACTLY at the stop price on any intrabar touch, with no gap-through or
liquidation modeling. That is fine at 1x, but it makes LEVERAGED returns dishonest:
it will happily report "32x return, no ruin" at 8x leverage even though XRP printed a
single -49.75% open->low bar that liquidates any long above ~2x. It also has no clean
out-of-sample hold-out: genomes are selected by walk-forward on the same series whose
return is then reported (in-sample selection bias).

This module adds the two honesty guards the adversarial review found missing:
  1. LIQUIDATION MODELING. Per trade we compute the Maximum Adverse Excursion (MAE)
     from the actual intrabar OHLC during the holding period. At leverage L (notional =
     L x equity, margin = equity), a position is LIQUIDATED when MAE >= (1 - maint)/L.
     A liquidated trade loses the full margin -> equity collapses -> ruin. This turns the
     engine's idealized "perfect stop fill" into a realistic ruin estimate.
  2. TRUE TRAIN/TEST HOLD-OUT. evaluate_holdout() lets you select params on a TRAIN
     slice and measure them on a never-seen TEST slice, so reported numbers are genuinely
     out-of-sample.

It does NOT modify the frozen engine, so the golden regression harness stays green.

USAGE
-----
    cd <repo> && ./venv/bin/python tools/honest_eval.py --strategy vol_regime_arb
"""
import sys, os, json, copy, argparse, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester
from engine.features import build_all_features

FUNDING_DAILY = 0.0003       # ~0.03%/day perp funding on notional (honest drag)
MAINT_MARGIN = 0.005         # 0.5% maintenance margin buffer before liquidation


def featured(asset="XRP"):
    """Build features + per-bar causal regime label for an asset."""
    raw = pd.read_parquet(f"data/binance_{asset}_USDT_1h.parquet")
    df = build_all_features(raw)
    from mirofish.scenario_engine import classify_regime_quantitative
    df["_regime"] = ["unknown" if i < 50 else classify_regime_quantitative(df, bar_idx=i).regime.value
                     for i in range(len(df))]
    return df.reset_index(drop=True)


def _apply_params(strat, override=None):
    from auto_evolve import STRATEGY_REGISTRY
    spec = STRATEGY_REGISTRY[strat]
    mod = importlib.import_module(spec["module"])
    fnref = getattr(mod, spec["function"])
    pdref = copy.deepcopy(getattr(mod, spec["params_dict"]))
    gp = f"reports/best_genome_{strat}.json"
    if os.path.exists(gp):
        pdref.update(json.load(open(gp)).get("params", {}))
    if override:
        pdref.update(override)
    setattr(mod, spec["params_dict"], pdref)
    return fnref


def _wrap(fnref, reg):
    def f(d, i, p):
        r = reg[i] if len(reg) == len(d) and 0 <= i < len(reg) else "unknown"
        return fnref(d, i, p, regime=r)
    return f


def trade_returns_and_mae(df, reg, strat, fnref):
    """Run the frozen engine at 1x, then derive per-trade fractional return and the
    Maximum Adverse Excursion (worst intrabar move against the position while held)."""
    bt = BacktestConfig(max_position_pct=1.0)
    rc = RiskConfig(); rc.sizing_method = "fixed_pct"
    trades, _eq, _m = Backtester(bt, rc).run(df, _wrap(fnref, reg), strat)
    ts = df["timestamp"].to_numpy()
    high = df["high"].to_numpy(); low = df["low"].to_numpy()
    t2i = {int(t): i for i, t in enumerate(ts)}
    out = []
    if len(trades) == 0:
        return out
    for _, t in trades.iterrows():
        sz = t.get("size", 0) or 0
        if sz <= 0:
            continue
        r = float(t["pnl_net"]) / sz
        e_i = t2i.get(int(t["entry_time"]))
        x_i = t2i.get(int(t["exit_time"]), e_i)
        ep = float(t["entry_price"])
        mae = 0.0
        if e_i is not None and x_i is not None and x_i >= e_i:
            seg_lo = low[e_i:x_i + 1]; seg_hi = high[e_i:x_i + 1]
            if len(seg_lo):
                if t["side"] == "long":
                    mae = max(0.0, (ep - float(seg_lo.min())) / ep)
                else:
                    mae = max(0.0, (float(seg_hi.max()) - ep) / ep)
        hrs = (t["exit_time"] - t["entry_time"]) / 3_600_000
        out.append({"r": r, "mae": mae, "hours": max(hrs, 0.0)})
    return out


def equity_path(trades, L, model_liquidation=True):
    """Compound trades at leverage L. Returns (final_mult, max_drawdown_pct, ruined)."""
    eq = 1.0; peak = 1.0; maxdd = 0.0; ruined = False
    liq_thresh = (1.0 - MAINT_MARGIN) / L
    for tr in trades:
        if model_liquidation and tr["mae"] >= liq_thresh:
            eq = 0.0; ruined = True; maxdd = 100.0
            break
        funding = L * FUNDING_DAILY * (tr["hours"] / 24.0)
        eq *= (1.0 + L * tr["r"] - funding)
        if eq <= 0:
            eq = 0.0; ruined = True; maxdd = 100.0
            break
        peak = max(peak, eq)
        maxdd = max(maxdd, (peak - eq) / peak * 100.0)
    return (0.0 if ruined else round(eq, 4)), round(maxdd, 1), ruined


def leverage_frontier(df, reg, strat, fnref, leverages=(1, 2, 3, 5, 8, 12, 20),
                      dd_cap=50.0, model_liquidation=True):
    trades = trade_returns_and_mae(df, reg, strat, fnref)
    rows = []
    for L in leverages:
        mult, dd, ruined = equity_path(trades, L, model_liquidation)
        rows.append({"L": L, "mult": mult, "maxDD_pct": dd, "ruined": ruined, "n_trades": len(trades)})
        if ruined:
            break
    ok = [r for r in rows if not r["ruined"] and r["maxDD_pct"] <= dd_cap]
    best = max(ok, key=lambda r: r["mult"]) if ok else None
    return rows, best


def _sample_params(strat, rng):
    """Sample one random param set from the registry's param_space."""
    from auto_evolve import STRATEGY_REGISTRY
    space = STRATEGY_REGISTRY[strat]["param_space"]
    p = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            p[name] = int(rng.integers(spec[1], spec[2] + 1))
        elif kind == "float":
            p[name] = float(rng.uniform(spec[1], spec[2]))
        elif kind == "bool":
            p[name] = bool(rng.integers(0, 2))
    return p


def _train_sharpe(df, reg, strat, override):
    """1x Sharpe on a segment for a given param override (selection metric, no leverage)."""
    fnref = _apply_params(strat, override)
    bt = BacktestConfig(max_position_pct=1.0); rc = RiskConfig(); rc.sizing_method = "fixed_pct"
    _t, _e, m = Backtester(bt, rc).run(df, _wrap(fnref, reg), strat)
    return (m.sharpe_ratio if m.total_trades >= 15 else -99.0), m.total_trades


def evaluate_holdout(strat, asset="XRP", train_frac=0.6, leverages=(1, 2, 3, 5),
                     model_liquidation=True, n_search=40, seed=42):
    """GENUINELY out-of-sample: random-search params on TRAIN only (selecting by TRAIN
    Sharpe), freeze them, then measure the leverage frontier on the never-seen TEST slice.
    The stored best_genome is NOT used for selection because it was evolved on the full
    series (which includes TEST) -> that would be a contaminated holdout."""
    df = featured(asset); reg = df["_regime"].to_numpy(); N = len(df)
    cut = int(N * train_frac)
    tr_df, tr_reg = df.iloc[:cut].reset_index(drop=True), reg[:cut]
    te_df, te_reg = df.iloc[cut:].reset_index(drop=True), reg[cut:]

    rng = np.random.default_rng(seed)
    best_override, best_sh = None, -1e9
    for _ in range(n_search):
        cand = _sample_params(strat, rng)
        sh, n = _train_sharpe(tr_df, tr_reg, strat, cand)
        if sh > best_sh:
            best_sh, best_override = sh, cand
    fnref = _apply_params(strat, best_override)
    res = {"selected_train_sharpe": round(best_sh, 3), "selected_params": best_override}
    for name, seg, sreg in [("TRAIN", tr_df, tr_reg), ("TEST(unseen)", te_df, te_reg)]:
        rows, best = leverage_frontier(seg, sreg, strat, fnref, leverages,
                                       model_liquidation=model_liquidation)
        res[name] = {"rows": rows, "best": best}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="vol_regime_arb")
    ap.add_argument("--asset", default="XRP")
    ap.add_argument("--no-liquidation", action="store_true",
                    help="disable liquidation modeling (reproduces the dishonest idealized engine)")
    args = ap.parse_args()

    df = featured(args.asset); reg = df["_regime"].to_numpy()
    fnref = _apply_params(args.strategy)
    model_liq = not args.no_liquidation

    print(f"\n=== {args.strategy} on {args.asset} — leverage frontier "
          f"({'LIQUIDATION-AWARE (honest)' if model_liq else 'IDEALIZED (dishonest)'}) ===")
    rows, best = leverage_frontier(df, reg, args.strategy, fnref, model_liquidation=model_liq)
    print(f"{'L':>4}{'mult':>10}{'maxDD%':>10}{'ruined':>9}")
    for r in rows:
        print(f"{r['L']:>4}{r['mult']:>10.3f}{r['maxDD_pct']:>10.1f}{str(r['ruined']):>9}")
    if best:
        print(f"-> honest max @ DD<=50%: L={best['L']}, {best['mult']:.3f}x full-period")
    else:
        print("-> no leverage survives DD<=50% without ruin")

    print(f"\n=== {args.strategy} — TRUE OUT-OF-SAMPLE HOLD-OUT (liquidation-aware) ===")
    ho = evaluate_holdout(args.strategy, args.asset, model_liquidation=model_liq)
    for seg in ("TRAIN", "TEST(unseen)"):
        b = ho[seg]["best"]
        bs = f"L={b['L']} {b['mult']:.3f}x maxDD {b['maxDD_pct']}%" if b else "no survivable leverage"
        print(f"  {seg:<14}: {bs}")
    print()


if __name__ == "__main__":
    main()
