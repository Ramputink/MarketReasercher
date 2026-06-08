"""
phase3_five_strategies.py — 5 simple, independent SPOT LONG-ONLY strategies
===========================================================================

Five intentionally SIMPLE, INDEPENDENT long-or-cash strategies on the 30-asset
daily basket. Each uses textbook parameters (NO fitting — avoids overfitting and
keeps them simple). They are evaluated individually AND as an equal-weight
ensemble, on in-sample (2019→2024-06) and ONCE on the sealed 2-year lockbox.

The diversification thesis: 5 independent positive-expectancy long-only sleeves
combine into something smoother and more robust than any single (riskier) system.
Trend/breakout sleeves go to CASH in downtrends (sidestep bears); dip/capitulation
sleeves are counter-trend. Their differences are the point.

Rules honoured: long spot or cash only; no shorts/leverage/borrow; funding is a
signal feature only (never PnL); exposure in [0,1].

Run:  python phase3_five_strategies.py
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from phase3_capitulation import load_panel, backtest, metrics, per_year

MAXPOS = 10                      # each filled slot = 1/MAXPOS; exposure scales with breadth
COST = 0.0015


def _weights_from(mask: pd.DataFrame, score: pd.DataFrame, max_pos=MAXPOS) -> pd.DataFrame:
    """Top-`max_pos` of `score` among `mask`==True each day, equal 1/max_pos each."""
    W = pd.DataFrame(0.0, index=mask.index, columns=mask.columns)
    sc = score.where(mask)
    for t in mask.index[mask.any(axis=1)]:
        row = sc.loc[t].dropna().sort_values(ascending=False)
        picks = row.index[:max_pos]
        if len(picks):
            W.loc[t, picks] = 1.0 / max_pos
    return W


def _rsi(close, n=14):
    d = close.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _ema(close, span):
    return close.ewm(span=span, adjust=False).mean()


# ── the 5 strategies: each returns a daily long-only weight matrix ────────────
def s1_trend(close, funding):
    """TS-momentum: long assets above their 200d EMA; breadth-scaled, cash in bear."""
    mom = close / close.shift(60) - 1
    mask = (close > _ema(close, 200)) & close.notna()
    return _weights_from(mask, mom)


def s2_relative_strength(close, funding):
    """XS-momentum: long top-MAXPOS by 90d return (pure relative strength)."""
    r90 = close / close.shift(90) - 1
    mask = r90.notna() & close.notna()
    return _weights_from(mask, r90)


def s3_dip_in_uptrend(close, funding):
    """Buy oversold dips ONLY inside a confirmed uptrend; hold ~10d."""
    rsi = _rsi(close, 14)
    trig = (rsi < 35) & (close > _ema(close, 150)) & close.notna()
    held = trig.copy()
    for k in range(1, 10):
        held = held | trig.shift(k, fill_value=False)
    held = held & close.notna()
    return _weights_from(held, -rsi)        # most-oversold first


def s4_capitulation(close, funding):
    """Funding extreme-negative + market regime-on (the v2 defensive sleeve)."""
    L = 7
    fz = (funding - funding.rolling(L).mean()) / funding.rolling(L).std(ddof=0)
    capit = (fz < -1.0) & (close > _ema(close, 5))
    held = capit.copy()
    for k in range(1, 10):
        held = held | capit.shift(k, fill_value=False)
    btc_on = close["BTC"] > _ema(close["BTC"], 150) if "BTC" in close else True
    breadth = (close > close.rolling(50).mean()).sum(axis=1) / close.notna().sum(axis=1).replace(0, np.nan)
    reg = (btc_on & (breadth >= 0.5)).fillna(False)
    held = (held & close.notna()).mul(reg, axis=0).astype(bool)
    return _weights_from(held, -fz)


def s5_breakout(close, funding):
    """Donchian-style: new 50d high AND above 200d EMA; hold ~15d."""
    hi = close.rolling(50).max().shift(1)
    trig = (close > hi) & (close > _ema(close, 200)) & close.notna()
    held = trig.copy()
    for k in range(1, 15):
        held = held | trig.shift(k, fill_value=False)
    held = held & close.notna()
    mom = close / close.shift(60) - 1
    return _weights_from(held, mom)


STRATS = {"trend_ts": s1_trend, "rel_strength": s2_relative_strength,
          "dip_in_uptrend": s3_dip_in_uptrend, "capitulation": s4_capitulation,
          "breakout": s5_breakout}


def evaluate(close, mask_window, label):
    cw = close[mask_window]
    rows = {}
    port_store = {}
    for name, fn in STRATS.items():
        W = fn(close, FUND)[mask_window]
        port, eq, wx = backtest(cw, W)
        rows[name] = metrics(port, eq, wx)
        rows[name]["per_year"] = per_year(port)
        port_store[name] = port
    # equal-weight ensemble of the 5 sleeves (each gets 1/5 of capital)
    ens = sum(p.reindex(cw.index).fillna(0.0) for p in port_store.values()) / len(port_store)
    ens_eq = (1 + ens).cumprod()
    rows["ENSEMBLE_5"] = metrics(ens, ens_eq, pd.DataFrame(1, index=cw.index, columns=["x"]) * 0.2)
    rows["ENSEMBLE_5"]["per_year"] = per_year(ens)
    return rows, ens


def main():
    global FUND
    close, funding, cutoff, man = load_panel()
    FUND = funding
    ins = close.index < cutoff
    print(f"panel {man['start']}→{man['end']}, lockbox sealed from {man['lockbox_cutoff_date']}")
    print(f"5 simple long-only sleeves + equal-weight ensemble\n")

    is_rows, _ = evaluate(close, ins, "in-sample")
    lb_rows, lb_ens = evaluate(close, ~ins, "lockbox")

    bh = close[~ins].pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    bh_eq = (1 + bh).cumprod()
    bh_ret = round((bh_eq.iloc[-1] - 1) * 100, 2)
    bh_dd = round((bh_eq / bh_eq.cummax() - 1).min() * 100, 2)

    def line(name, m):
        return (f"{name:16} CAGR {m['cagr_pct']:7.1f}%  Sharpe {m['sharpe']:6.2f}  "
                f"maxDD {m['max_dd_pct']:7.1f}%  expo {m['avg_exposure_pct']:5.1f}%  "
                f"yrs {m['per_year']}")

    print("=== IN-SAMPLE (2019→2024-06) ===")
    for n in list(STRATS) + ["ENSEMBLE_5"]:
        print(" ", line(n, is_rows[n]))
    print(f"\n=== SEALED LOCKBOX (2024-06→2026-06, 2-year OOS) ===")
    for n in list(STRATS) + ["ENSEMBLE_5"]:
        print(" ", line(n, lb_rows[n]))
    print(f"\n  Buy&hold equal-weight basket OOS: {bh_ret}%  (maxDD {bh_dd}%)")

    out = {"in_sample": is_rows, "lockbox_oos": lb_rows,
           "lockbox_buy_hold": {"total_ret_pct": bh_ret, "max_dd_pct": bh_dd},
           "max_positions": MAXPOS, "note": "textbook params, no fitting; funding signal-only"}
    json.dump(out, open("reports/phase3_five_strategies_results.json", "w"), indent=2, default=str)
    pd.DataFrame({"date": close[~ins].index, "ensemble_ret": lb_ens.values}).to_csv(
        "reports/phase3_five_strategies_ensemble_equity.csv", index=False)
    print("\nresults → reports/phase3_five_strategies_results.json")
    return out


if __name__ == "__main__":
    main()
