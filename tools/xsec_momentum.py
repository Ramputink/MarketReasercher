"""
xsec_momentum.py — Cross-sectional momentum basket (APT factor), LONG-ONLY.

Thesis (Arbitrage Pricing Theory / crypto momentum literature): relative strength is a
priced factor. Each rebalance, rank the coin universe by trailing return and hold the
top-k EQUAL-WEIGHT (long-only; only coins with positive momentum, else that slot is
cash). This trades RELATIVE strength across coins, not single-asset timing — the angle
that single-asset intraday strategies (Cycle 2) failed at.

Honest evaluation: train (first 60%) picks (lookback, k, rebalance) by Sharpe; the frozen
rule is measured on the unseen last 40%. Benchmarks: equal-weight-all-coins and BTC hold.
Costs: 0.1% commission + 5 bps slippage on turnover each rebalance.

Run from repo root: ./venv/bin/python tools/xsec_momentum.py
"""
import sys, os, json, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

TMP = "/Users/matveypro/.claude/jobs/1ca96668/tmp"
UNIVERSE = ["BTC","ETH","BNB","XRP","SOL","ADA","DOGE","AVAX","DOT","LINK","LTC","TRX","ATOM","BCH","UNI"]
COST = 0.001 + 0.0005  # commission + slippage per unit turnover

def load_closes():
    cols = {}
    for c in UNIVERSE:
        df = pd.read_pickle(f"{TMP}/{c}_featured.pkl").reset_index(drop=True)
        cols[c] = df["close"].to_numpy(dtype=float)
    n = min(len(v) for v in cols.values())
    M = np.column_stack([cols[c][:n] for c in UNIVERSE])  # (n_bars, n_coins)
    return M

def backtest(M, lookback, k, rebal, lo, hi):
    """Equity curve over [lo,hi) with daily-ish rebalance. Returns (mult, dd, sharpe, ann_turnover)."""
    eq = 1.0; peak = 1.0; maxdd = 0.0
    w = np.zeros(M.shape[1])  # current weights
    eq_curve = []
    turnover_total = 0.0; rebals = 0
    start = max(lo, lookback + 1)
    for t in range(start, hi):
        # rebalance every `rebal` bars
        if (t - start) % rebal == 0:
            mom = M[t] / M[t - lookback] - 1.0
            order = np.argsort(mom)[::-1]
            pick = [i for i in order if mom[i] > 0][:k]   # long-only: positive momentum
            new_w = np.zeros_like(w)
            if pick:
                for i in pick: new_w[i] = 1.0 / k          # equal weight selected
            turnover = np.abs(new_w - w).sum()
            eq *= (1.0 - COST * turnover)                  # pay turnover cost
            turnover_total += turnover; rebals += 1
            w = new_w
        # mark-to-market one bar forward
        if t + 1 < hi:
            bar_ret = np.where(w > 0, M[t + 1] / M[t] - 1.0, 0.0)
            eq *= (1.0 + float((w * bar_ret).sum()))
        peak = max(peak, eq); maxdd = max(maxdd, (peak - eq) / peak * 100)
        eq_curve.append(eq)
    eqc = np.array(eq_curve)
    if len(eqc) > 2:
        rets = eqc[1:] / eqc[:-1] - 1.0
        sharpe = float(np.mean(rets) / (np.std(rets) + 1e-12) * np.sqrt(365 * 24))
    else:
        sharpe = 0.0
    ann_turn = turnover_total / max(rebals, 1) * (365 * 24 / rebal)
    return eq, maxdd, sharpe, ann_turn

def bench_equal_all(M, lo, hi):
    start = lo + 1; eq = 1.0
    w = np.ones(M.shape[1]) / M.shape[1]
    for t in range(start, hi):
        bar_ret = M[t] / M[t - 1] - 1.0
        eq *= (1.0 + float((w * bar_ret).mean() * M.shape[1] / M.shape[1]))  # equal-weight avg
        eq *= 1.0  # buy&hold equal weight, no rebal cost
    # simpler: equal-weight buy&hold = mean of per-coin holding returns
    hold = float(np.mean(M[hi-1] / M[lo] ))
    return hold

def main():
    M = load_closes(); n = M.shape[0]; cut = int(n * 0.6)
    print(f"universe={M.shape[1]} coins, {n} bars, train[:{cut}] test[{cut}:]")
    grid = list(itertools.product([72, 168, 336], [2, 3, 5], [24, 48]))  # lookback h, k, rebal h
    best, best_sh = None, -1e9
    for lb, k, rb in grid:
        _, _, sh, _ = backtest(M, lb, k, rb, 0, cut)
        if sh > best_sh: best_sh, best = sh, (lb, k, rb)
    lb, k, rb = best
    print(f"BEST train rule: lookback={lb}h, k={k}, rebal={rb}h (train Sharpe {best_sh:.2f})\n")

    # OOS test
    eq, dd, sh, turn = backtest(M, lb, k, rb, cut, n)
    bench = bench_equal_all(M, cut, n)
    btc_hold = float(M[n-1, 0] / M[cut, 0])
    print(f"{'metric':<22}{'xsec_momentum':>14}{'eqweight_all':>14}{'BTC_hold':>10}")
    print(f"{'OOS multiple':<22}{eq:>14.3f}{bench:>14.3f}{btc_hold:>10.3f}")
    print(f"{'OOS maxDD %':<22}{dd:>14.1f}{'-':>14}{'-':>10}")
    print(f"{'OOS Sharpe':<22}{sh:>14.2f}")
    print(f"{'ann. turnover (x)':<22}{turn:>14.1f}")
    beats_bench = eq > bench; beats_cash = eq > 1.0
    print(f"\nbeats equal-weight-all OOS: {'YES' if beats_bench else 'no'} | "
          f"beats cash (>1.0): {'YES' if beats_cash else 'no'}")
    out = {"best_rule": {"lookback_h": lb, "k": k, "rebal_h": rb}, "train_sharpe": best_sh,
           "oos_mult": eq, "oos_dd": dd, "oos_sharpe": sh, "bench_eqweight": bench,
           "btc_hold": btc_hold, "beats_bench": bool(beats_bench), "beats_cash": bool(beats_cash)}
    json.dump(out, open(f"{TMP}/xsec_momentum_result.json", "w"), indent=1)
    print(f"\nsaved -> {TMP}/xsec_momentum_result.json")

if __name__ == "__main__":
    main()
