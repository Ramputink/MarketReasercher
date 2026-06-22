"""
bmsb_search.py — honest train/test search for the real BMSB long-only strategy.

Picks ONE global parameter set (same rule for all coins) by mean risk-adjusted score
on the TRAIN segment (first 60% of each coin's daily history), then reports the frozen
rule's performance on the unseen TEST segment (last 40%). One global rule resists the
multiple-testing/per-coin overfit that plagues coin-specific tuning.

Run from repo root: ./venv/bin/python tools/bmsb_search.py
"""
import sys, os, json, copy, importlib, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester

TMP = "/Users/matveypro/.claude/jobs/1ca96668/tmp"
COINS = ["BTC", "ETH", "BNB", "SOL", "ADA", "LINK"]


def _load(coin):
    df = pd.read_pickle(f"{TMP}/{coin}_daily_feat.pkl").reset_index(drop=True)
    df = df[df["bmsb_sma_real"].notna()].reset_index(drop=True)
    return df, df["_regime"].to_numpy()


def _run(df, reg, override, L=1.0):
    mod = importlib.import_module("strategies.bmsb_long")
    base = copy.deepcopy(mod.PARAMS); base.update(override); mod.PARAMS = base
    fnref = mod.bmsb_long_strategy
    bt = BacktestConfig(max_position_pct=float(L)); rc = RiskConfig(); rc.sizing_method = "fixed_pct"
    def fn(d, i, p):
        r = reg[i] if 0 <= i < len(reg) else "unknown"
        return fnref(d, i, p, regime=r)
    _t, _e, m = Backtester(bt, rc).run(df, fn, "bmsb_long")
    return m


def score(m):
    """Risk-adjusted: total multiple penalized by drawdown (Calmar-like)."""
    mult = 1 + m.total_return_pct / 100
    return mult / (1 + m.max_drawdown_pct / 100)


def main():
    # Grid (kept small & sensible to limit multiple-testing)
    grid = list(itertools.product(
        [0.0, 0.01, 0.02],          # entry_buffer
        [0.0, 0.03, 0.06, 0.10],    # exit_buffer
        [0, 1, 2],                  # confirm_bars
        [False, True],              # require_band_bullish
    ))
    data = {c: _load(c) for c in COINS}
    splits = {c: int(len(data[c][0]) * 0.6) for c in COINS}

    best, best_train = None, -1e9
    for eb, xb, cb, bull in grid:
        ov = {"entry_buffer": eb, "exit_buffer": xb, "confirm_bars": cb,
              "require_band_bullish": bull}
        tr_scores = []
        for c in COINS:
            df, reg = data[c]; cut = splits[c]
            m = _run(df.iloc[:cut].reset_index(drop=True), reg[:cut], ov)
            tr_scores.append(score(m))
        avg = float(np.mean(tr_scores))
        if avg > best_train:
            best_train, best = avg, ov

    print("BEST GLOBAL RULE (by mean train risk-adjusted score):", best)
    print(f"  mean train score = {best_train:.3f}\n")

    # Frozen rule on unseen TEST per coin, vs default and buy&hold
    default = {"entry_buffer": 0.0, "exit_buffer": 0.03, "confirm_bars": 2, "require_band_bullish": False}
    print(f"{'coin':<6}{'TEST default':>14}{'TEST tuned':>12}{'B&H':>9}{'tunedDD%':>10}{'beatsBH':>8}")
    rows = []
    for c in COINS:
        df, reg = data[c]; cut = splits[c]
        te, tre = df.iloc[cut:].reset_index(drop=True), reg[cut:]
        md = _run(te, tre, default); mt = _run(te, tre, best)
        bh = te["close"].iloc[-1] / te["close"].iloc[0]
        dmult = 1 + md.total_return_pct / 100; tmult = 1 + mt.total_return_pct / 100
        beats = "YES" if tmult > bh else "no"
        rows.append({"coin": c, "test_default": round(dmult, 3), "test_tuned": round(tmult, 3),
                     "bh": round(bh, 3), "tuned_dd": round(mt.max_drawdown_pct, 1), "beats": beats})
        print(f"{c:<6}{dmult:>14.3f}{tmult:>12.3f}{bh:>9.3f}{mt.max_drawdown_pct:>10.1f}{beats:>8}")

    tuned = [r["test_tuned"] for r in rows]; bhs = [r["bh"] for r in rows]
    n_beat = sum(1 for r in rows if r["beats"] == "YES")
    print(f"\nTEST: tuned mean {np.mean(tuned):.3f}x vs B&H mean {np.mean(bhs):.3f}x | "
          f"beats B&H on {n_beat}/{len(rows)} coins")
    out = {"best_rule": best, "mean_train_score": best_train, "test_rows": rows,
           "test_tuned_mean": float(np.mean(tuned)), "test_bh_mean": float(np.mean(bhs)),
           "n_beats_bh": n_beat, "n_coins": len(rows)}
    json.dump(out, open(f"{TMP}/bmsb_search_result.json", "w"), indent=1)
    print(f"\nsaved -> {TMP}/bmsb_search_result.json")


if __name__ == "__main__":
    main()
