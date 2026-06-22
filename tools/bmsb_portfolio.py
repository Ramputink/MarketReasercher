"""
bmsb_portfolio.py — equal-weight multi-coin BMSB portfolio (Markowitz diversification).

Each coin runs the real BMSB long-or-cash strategy on its OWN band; capital is split
equally across coins; sleeves compound independently. Thesis: when one coin is in a bear
(its sleeve sits in cash) others may be trending, so the PORTFOLIO drawdown should be far
below any single coin's, at competitive return — the core MPT diversification benefit.

Honest read: report the recent (last-40%, OOS-style) window — portfolio vs the average
single-coin sleeve vs an equal-weight buy&hold of the same coins over the same window.

Run from repo root: ./venv/bin/python tools/bmsb_portfolio.py
"""
import sys, os, json, copy, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester

TMP = "/Users/matveypro/.claude/jobs/1ca96668/tmp"
COINS = ["BTC", "ETH", "BNB", "SOL", "ADA", "LINK"]


def sleeve_equity(coin):
    """Return a Series (indexed by timestamp ms) of BMSB strategy equity, normalized to 1."""
    df = pd.read_pickle(f"{TMP}/{coin}_daily_feat.pkl").reset_index(drop=True)
    df = df[df["bmsb_sma_real"].notna()].reset_index(drop=True)
    reg = df["_regime"].to_numpy()
    mod = importlib.import_module("strategies.bmsb_long")
    mod.PARAMS = copy.deepcopy(mod.PARAMS)  # defaults
    fn0 = mod.bmsb_long_strategy
    bt = BacktestConfig(max_position_pct=1.0); rc = RiskConfig(); rc.sizing_method = "fixed_pct"
    def fn(d, i, p):
        r = reg[i] if 0 <= i < len(reg) else "unknown"
        return fn0(d, i, p, regime=r)
    _t, eq, _m = Backtester(bt, rc).run(df, fn, "bmsb_long")
    s = pd.Series(eq.values, index=pd.to_datetime(df["timestamp"].iloc[-len(eq):].values, unit="ms"))
    return s / s.iloc[0]


def hold_equity(coin):
    df = pd.read_pickle(f"{TMP}/{coin}_daily_feat.pkl").reset_index(drop=True)
    df = df[df["bmsb_sma_real"].notna()].reset_index(drop=True)
    s = pd.Series(df["close"].values, index=pd.to_datetime(df["timestamp"].values, unit="ms"))
    return s / s.iloc[0]


def dd_pct(curve):
    peak = np.maximum.accumulate(curve); return float(((peak - curve) / peak).max() * 100)


def stats(curve):
    curve = np.asarray(curve, dtype=float)
    rets = curve[1:] / curve[:-1] - 1.0
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(365)) if len(rets) > 2 else 0.0
    return curve[-1] / curve[0], dd_pct(curve), sharpe


def main():
    sleeves = {c: sleeve_equity(c) for c in COINS}
    holds = {c: hold_equity(c) for c in COINS}
    # common daily timeline (inner join across sleeves)
    idx = None
    for s in sleeves.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    idx = idx.sort_values()
    S = pd.DataFrame({c: sleeves[c].reindex(idx).ffill() for c in COINS})
    H = pd.DataFrame({c: holds[c].reindex(idx).ffill() for c in COINS})
    n = len(idx); cut = int(n * 0.6)

    for name, lo, hi in [("FULL", 0, n), ("TEST(recent 40%)", cut, n)]:
        seg = S.iloc[lo:hi]; segh = H.iloc[lo:hi]
        # normalize each sleeve/hold to 1 at window start, equal-weight average
        segN = seg / seg.iloc[0]; seghN = segh / segh.iloc[0]
        port = segN.mean(axis=1).to_numpy()           # equal-weight BMSB portfolio
        bh = seghN.mean(axis=1).to_numpy()            # equal-weight buy&hold
        p_mult, p_dd, p_sh = stats(port)
        b_mult, b_dd, b_sh = stats(bh)
        single_dd = [dd_pct((seg[c] / seg[c].iloc[0]).to_numpy()) for c in COINS]
        single_mult = [(seg[c].iloc[-1] / seg[c].iloc[0]) for c in COINS]
        print(f"\n=== {name} ({hi-lo} days) ===")
        print(f"  PORTFOLIO BMSB : mult {p_mult:.3f}x  maxDD {p_dd:.1f}%  Sharpe {p_sh:.2f}")
        print(f"  eq-wt BUY&HOLD : mult {b_mult:.3f}x  maxDD {b_dd:.1f}%  Sharpe {b_sh:.2f}")
        print(f"  avg single BMSB: mult {np.mean(single_mult):.3f}x  avg maxDD {np.mean(single_dd):.1f}% "
              f"(worst single DD {max(single_dd):.1f}%)")
        print(f"  -> portfolio DD {p_dd:.1f}% vs avg-single {np.mean(single_dd):.1f}% vs B&H {b_dd:.1f}%; "
              f"beats B&H mult: {'YES' if p_mult>b_mult else 'no'}; "
              f"DD cut vs avg-single: {'YES' if p_dd<np.mean(single_dd) else 'no'}")
        if name.startswith("TEST"):
            out = {"window": name, "port_mult": p_mult, "port_dd": p_dd, "port_sharpe": p_sh,
                   "bh_mult": b_mult, "bh_dd": b_dd, "avg_single_dd": float(np.mean(single_dd)),
                   "beats_bh": bool(p_mult > b_mult), "cuts_dd": bool(p_dd < np.mean(single_dd))}
            json.dump(out, open(f"{TMP}/bmsb_portfolio_result.json", "w"), indent=1)
    print(f"\nsaved -> {TMP}/bmsb_portfolio_result.json")


if __name__ == "__main__":
    main()
