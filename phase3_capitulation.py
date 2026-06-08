"""
phase3_capitulation.py — SPOT LONG-ONLY "buy capitulation" strategy
===================================================================

Thesis (the one signal the project ever found with a pulse — funding — recast as
a SPOT LONG-ONLY ENTRY FEATURE, never as perp PnL):

  Extreme-negative funding = crowded shorts / capitulation / fear. It often marks
  local bottoms. So go LONG SPOT in an asset when its funding is extreme-negative
  (relative to its own history) AND a bounce is confirmed AND the market is not in
  freefall. Otherwise hold CASH. Long-or-cash only; net exposure in [0, 1]; no
  shorts, no leverage, no borrow, funding is NEVER booked as PnL (PnL is purely the
  price move of a long spot position).

Honest protocol: calibrate ONLY on the in-sample window (< 2024-06-07); evaluate
the single chosen config ONCE on the sealed 2-year lockbox. (Perp close is used as
a spot-price proxy — they track within a few bps for these liquid assets; funding
is signal-only, consistent with the spot constraint.)

Run:  python phase3_capitulation.py
"""
from __future__ import annotations

import json, os, itertools
import numpy as np
import pandas as pd

PANEL = "phase2/snapshots"
COST_PER_SIDE = 0.001 + 5e-4   # 10 bps commission + 5 bps slippage, on traded weight


def load_panel():
    man = json.load(open(f"{PANEL}/manifest.json"))
    close = pd.read_parquet(f"{PANEL}/close.parquet")
    funding = pd.read_parquet(f"{PANEL}/funding.parquet")
    cutoff = pd.Timestamp(man["lockbox_cutoff_ts"], unit="ms", tz="UTC").tz_localize(None) \
        if not isinstance(close.index, pd.DatetimeIndex) else pd.Timestamp(man["lockbox_cutoff_ts"], unit="ms")
    cutoff = pd.Timestamp(man["lockbox_cutoff_ts"], unit="ms")
    close.index = pd.to_datetime(close.index)
    funding.index = pd.to_datetime(funding.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None); funding.index = funding.index.tz_localize(None)
    return close, funding, cutoff, man


def signal_weights(close, funding, p):
    """Build the daily long-only target-weight matrix from a config p."""
    L = p["funding_lookback"]
    fz = (funding - funding.rolling(L).mean()) / funding.rolling(L).std(ddof=0)
    capit = fz < -p["z_thresh"]                                   # extreme-negative funding
    # bounce confirmation: price above its short MA
    if p["confirm_ma"] > 0:
        confirm = close > close.rolling(p["confirm_ma"]).mean()
    else:
        confirm = pd.DataFrame(True, index=close.index, columns=close.columns)
    # market risk-on filter: BTC above its macro MA (broadcast)
    if p["risk_ma"] > 0 and "BTC" in close.columns:
        btc_on = (close["BTC"] > close["BTC"].rolling(p["risk_ma"]).mean())
        risk_on = pd.DataFrame(np.repeat(btc_on.values[:, None], close.shape[1], axis=1),
                               index=close.index, columns=close.columns)
    else:
        risk_on = pd.DataFrame(True, index=close.index, columns=close.columns)
    raw = capit & confirm & risk_on & close.notna()
    # sticky hold: stay long for at least hold_days after a trigger
    held = raw.copy()
    for k in range(1, p["hold_days"]):
        held = held | raw.shift(k).fillna(False)
    held = held & close.notna()
    # rank by most-negative funding-z; keep up to max_positions per day
    score = (-fz).where(held)   # higher = more extreme capitulation
    W = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    mp = p["max_positions"]
    for t in held.index[held.any(axis=1)]:
        row = score.loc[t].dropna().sort_values(ascending=False)
        picks = row.index[:mp]
        if len(picks):
            W.loc[t, picks] = 1.0 / mp    # each slot = 1/max_positions; cash = remainder
    return W


def backtest(close, W):
    ret = close.pct_change(fill_method=None).fillna(0.0)
    # weights decided at close t earn the return from t to t+1 (next-bar hold)
    w_exec = W.shift(1).fillna(0.0)
    gross = (w_exec * ret).sum(axis=1)
    turnover = (W - W.shift(1)).abs().sum(axis=1).fillna(0.0)
    cost = turnover * COST_PER_SIDE
    port = gross - cost.shift(1).fillna(0.0)
    eq = (1 + port).cumprod()
    return port, eq, w_exec


def metrics(port, eq, w_exec):
    p = port[port.index >= eq.index[0]]
    ann = 365
    n_years = len(p) / ann
    total = eq.iloc[-1] - 1
    cagr = eq.iloc[-1] ** (1 / max(n_years, 1e-9)) - 1
    sd = p.std()
    sharpe = (p.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0
    dn = p[p < 0].std()
    sortino = (p.mean() / dn * np.sqrt(ann)) if dn and dn > 0 else 0.0
    dd = (eq / eq.cummax() - 1).min()
    exposure = (w_exec.sum(axis=1)).mean()
    active_days = (w_exec.sum(axis=1) > 0).mean()
    return {
        "cagr_pct": round(cagr * 100, 2), "total_ret_pct": round(total * 100, 2),
        "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "max_dd_pct": round(dd * 100, 2),
        "calmar": round(cagr / abs(dd), 3) if dd < 0 else 0.0,
        "avg_exposure_pct": round(exposure * 100, 1),
        "days_in_market_pct": round(active_days * 100, 1),
        "n_days": int(len(p)),
    }


def per_year(port):
    return {int(y): round(g.mean() / g.std() * np.sqrt(365), 3) if g.std() > 0 else 0.0
            for y, g in port.groupby(port.index.year)}


GRID = {
    "funding_lookback": [7, 14, 30],
    "z_thresh": [1.0, 1.5, 2.0],
    "confirm_ma": [0, 3, 5],
    "risk_ma": [0, 50],
    "hold_days": [3, 5, 10],
    "max_positions": [5, 10],
}


def main():
    close, funding, cutoff, man = load_panel()
    ins = close.index < cutoff
    print(f"panel {man['start']}→{man['end']}, lockbox sealed from {man['lockbox_cutoff_date']}")
    print(f"in-sample days {int(ins.sum())}, lockbox days {int((~ins).sum())}\n")

    keys = list(GRID); combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"calibrating {len(combos)} configs on IN-SAMPLE only…")
    best = None
    for vals in combos:
        p = dict(zip(keys, vals))
        W = signal_weights(close, funding, p)
        Wi = W[ins]
        if Wi.abs().sum().sum() == 0:
            continue
        port, eq, wx = backtest(close[ins], Wi)
        if len(port) < 100 or (wx.sum(axis=1) > 0).sum() < 60:   # need real activity
            continue
        m = metrics(port, eq, wx)
        # select by in-sample Calmar (return per unit drawdown) — robust for long-only
        scoreval = m["calmar"] if m["cagr_pct"] > 0 else -9
        if best is None or scoreval > best["score"]:
            best = {"score": scoreval, "p": p, "is": m}

    if best is None:
        print("No viable in-sample config."); return
    print("\nBEST in-sample config:", best["p"])
    print("in-sample:", best["is"])

    # ── single sealed-lockbox evaluation ──
    p = best["p"]
    W = signal_weights(close, funding, p)
    Wl = W[~ins]
    port, eq, wx = backtest(close[~ins], Wl)
    lb = metrics(port, eq, wx)
    lb_year = per_year(port)
    # buy & hold equal-weight basket over lockbox (benchmark)
    bh_ret = close[~ins].pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    bh_eq = (1 + bh_ret).cumprod()
    bh = {"total_ret_pct": round((bh_eq.iloc[-1]-1)*100, 2),
          "max_dd_pct": round((bh_eq/bh_eq.cummax()-1).min()*100, 2),
          "sharpe": round(bh_ret.mean()/bh_ret.std()*np.sqrt(365), 3) if bh_ret.std()>0 else 0}

    out = {"strategy": "spot_longonly_buy_capitulation", "config": p,
           "in_sample": best["is"], "lockbox_oos": lb, "lockbox_per_year": lb_year,
           "lockbox_buy_hold_basket": bh,
           "lockbox_span": [str(close[~ins].index.min().date()), str(close[~ins].index.max().date())]}
    os.makedirs("reports", exist_ok=True)
    json.dump(out, open("reports/phase3_capitulation_lockbox_results.json", "w"), indent=2, default=str)
    eq_full = pd.concat([eq]).reset_index()
    eq_full.columns = ["date", "equity"]
    eq_full.to_csv("reports/phase3_capitulation_equity.csv", index=False)

    print("\n" + "=" * 70)
    print("SEALED LOCKBOX (2-year OOS) — spot long-only buy-capitulation")
    print("=" * 70)
    for k, v in lb.items():
        print(f"  {k:22} {v}")
    print(f"  per-year Sharpe: {lb_year}")
    print(f"\n  Buy&hold equal-weight basket OOS: {bh['total_ret_pct']}% "
          f"(maxDD {bh['max_dd_pct']}%, Sharpe {bh['sharpe']})")
    print(f"\nresults → reports/phase3_capitulation_lockbox_results.json")
    return out


if __name__ == "__main__":
    main()
