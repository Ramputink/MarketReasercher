"""
phase3_portfolio_overlay.py — portfolio-level de-risking overlay (Phase 3, P1+P3)
================================================================================

The open issue from the 5-strategy work: the trend_ts+breakout+capitulation
ensemble made +10% on the sealed bear lockbox while the basket lost 46%, BUT its
max drawdown was still −41.8% because the three sleeves co-invest into the same
drops. This module adds a PORTFOLIO-LEVEL overlay that scales total book exposure
down toward cash when risk rises — the piece that separates "interesting" from
"operable".

The overlay is two TEXTBOOK, PRE-SPECIFIED risk controls (no fitting to the
lockbox — these are conventional risk-management constants, not optimised knobs):

  1. VOL TARGETING. Scale exposure so the ensemble's trailing realized vol tracks
     a 15% annual target: e_vol = min(1, target / realized_vol_{t-1}). Caps the
     book exactly when volatility (and clustered drawdown) spikes.

  2. TRAILING-DRAWDOWN CIRCUIT BREAKER with hysteresis. If the live (already
     overlaid) account is more than 20% below its running peak, cut exposure to a
     0.25 floor until it recovers to within 10% of the peak. Hysteresis avoids
     whipsawing in and out at the trigger.

Both are strictly CAUSAL: the exposure scalar for day t uses only returns through
t−1, so there is no look-ahead. The scalar is in [0, 1] — we only ever DE-risk
toward cash, never lever, so the long-only / no-leverage contract holds.

Honest protocol: the sleeves and the overlay constants are fixed ex-ante; we read
the sealed lockbox exactly once and report whether the overlay cut the drawdown
without destroying the return.

Run:  python phase3_portfolio_overlay.py
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from phase3_capitulation import load_panel, backtest, metrics, per_year
from phase3_five_strategies import s1_trend, s4_capitulation, s5_breakout

# ── pre-registered overlay constants (textbook, NOT fitted to the lockbox) ──────
TARGET_VOL_ANN = 0.15      # 15% annualised target volatility for the book
VOL_LOOKBACK = 20          # trailing window for realized vol
DD_TRIGGER = -0.20         # cut to floor when trailing drawdown breaches −20%
DD_REENTRY = -0.10         # restore full scaling once recovered above −10%
RISK_FLOOR = 0.25          # minimum exposure while the breaker is engaged
ANN = 365

# the recommended ex-ante ensemble: the three CASH-GATED sleeves
SLEEVES = {"trend_ts": s1_trend, "breakout": s5_breakout, "capitulation": s4_capitulation}


def ensemble_returns(close, funding, mask):
    """Equal-weight (1/N) daily net return of the cash-gated sleeves on `mask`."""
    cw = close[mask]
    ports = {}
    for name, fn in SLEEVES.items():
        W = fn(close, funding)[mask]
        port, _, _ = backtest(cw, W)
        ports[name] = port.reindex(cw.index).fillna(0.0)
    ens = sum(ports.values()) / len(ports)
    return ens, ports


def _vol_scalar(ens_ret: pd.Series) -> pd.Series:
    """Vol-targeting scalar in [0,1], causal (day t uses vol through t−1)."""
    rv = ens_ret.rolling(VOL_LOOKBACK, min_periods=max(5, VOL_LOOKBACK // 2)).std(ddof=0)
    rv_ann = (rv * np.sqrt(ANN)).shift(1)
    e_vol = (TARGET_VOL_ANN / rv_ann).clip(upper=1.0)
    return e_vol.fillna(1.0)             # warm-up: full exposure until vol defined


def exposure_overlay(ens_ret: pd.Series, breaker: bool = True) -> pd.Series:
    """
    Daily exposure scalar in [0,1], known one day ahead (causal). Vol targeting,
    optionally combined with a hysteretic trailing-drawdown circuit breaker
    computed on the live overlaid equity. `breaker=False` isolates vol targeting
    (which provably preserves Sharpe), so the two controls can be decomposed.
    """
    e_vol = _vol_scalar(ens_ret)
    idx = ens_ret.index
    e = pd.Series(1.0, index=idx)
    eq, peak, engaged = 1.0, 1.0, False
    for i in range(len(idx)):
        if breaker:
            dd = eq / peak - 1.0          # decide from YESTERDAY's state (causal)
            if engaged and dd >= DD_REENTRY:
                engaged = False
            elif (not engaged) and dd <= DD_TRIGGER:
                engaged = True
        scalar = float(e_vol.iloc[i]) * (RISK_FLOOR if (breaker and engaged) else 1.0)
        e.iloc[i] = scalar
        eq *= 1.0 + scalar * float(ens_ret.iloc[i])
        peak = max(peak, eq)
    return e


def apply_overlay(ens_ret: pd.Series, breaker: bool = True):
    e = exposure_overlay(ens_ret, breaker=breaker)
    return e * ens_ret, e


def _summ(ret: pd.Series) -> dict:
    eq = (1 + ret).cumprod()
    return metrics(ret, eq, pd.DataFrame(1.0, index=ret.index, columns=["x"]))


def _variant(ret, e=None, per_yr=True):
    m = _summ(ret)
    if per_yr:
        m["per_year"] = per_year(ret)
    if e is not None:
        m["avg_exposure_scalar"] = round(float(e.mean()), 3)
        m["min_exposure_scalar"] = round(float(e.min()), 3)
        m["days_derisked_pct"] = round(float((e < 0.999).mean()) * 100, 1)
    return m


def evaluate(close, funding, mask, label):
    ens, ports = ensemble_returns(close, funding, mask)
    voln, e_v = apply_overlay(ens, breaker=False)       # vol-target only
    full, e_f = apply_overlay(ens, breaker=True)        # vol-target + DD breaker

    # capitulation sleeve standalone, raw vs full overlay (P3: carry-as-feature)
    cap_port = ports["capitulation"]
    cap_full, e_c = apply_overlay(cap_port, breaker=True)

    rows = {
        "ensemble_raw": _variant(ens),
        "ensemble_voltarget": _variant(voln, e_v),
        "ensemble_capped": _variant(full, e_f),
        "capitulation_raw": _variant(cap_port),
        "capitulation_capped": _variant(cap_full, e_c),
    }
    return rows, (ens, full, e_f)


def main():
    close, funding, cutoff, man = load_panel()
    ins = close.index < cutoff
    print(f"panel {man['start']}→{man['end']}, lockbox sealed from {man['lockbox_cutoff_date']}")
    print("3-sleeve cash-gating ensemble (trend_ts+breakout+capitulation) + de-risking overlay")
    print(f"overlay: target_vol={TARGET_VOL_ANN:.0%} ann, DD trigger {DD_TRIGGER:.0%}/"
          f"reentry {DD_REENTRY:.0%}, floor {RISK_FLOOR:.0%}  (pre-specified, not fitted)\n")

    is_rows, _ = evaluate(close, funding, ins, "in-sample")
    lb_rows, (ens_lb, cap_lb, e_lb) = evaluate(close, funding, ~ins, "lockbox")

    bh = close[~ins].pct_change(fill_method=None).mean(axis=1).fillna(0.0)
    bh_eq = (1 + bh).cumprod()
    bh_out = {"total_ret_pct": round((bh_eq.iloc[-1] - 1) * 100, 2),
              "max_dd_pct": round((bh_eq / bh_eq.cummax() - 1).min() * 100, 2)}

    def line(tag, m):
        return (f"{tag:22} CAGR {m['cagr_pct']:7.1f}%  Sharpe {m['sharpe']:6.2f}  "
                f"Sortino {m['sortino']:6.2f}  maxDD {m['max_dd_pct']:7.1f}%  "
                f"Calmar {m['calmar']:6.2f}")

    for scope, rows in [("IN-SAMPLE (2019→2024-06)", is_rows),
                        ("SEALED LOCKBOX (2-year OOS)", lb_rows)]:
        print(f"=== {scope} ===")
        print(" ", line("ensemble RAW", rows["ensemble_raw"]))
        print(" ", line("ensemble vol-target", rows["ensemble_voltarget"]))
        print(" ", line("ensemble vol+breaker", rows["ensemble_capped"]))
        print(" ", line("capitulation RAW", rows["capitulation_raw"]))
        print(" ", line("capitulation +OVERLAY", rows["capitulation_capped"]))
        if scope.startswith("SEALED"):
            c = rows["ensemble_capped"]
            print(f"   full-overlay activity: avg exposure {c['avg_exposure_scalar']}, "
                  f"min {c['min_exposure_scalar']}, de-risked {c['days_derisked_pct']}% of days")
        print()
    print(f"  Buy&hold basket OOS: {bh_out['total_ret_pct']}%  (maxDD {bh_out['max_dd_pct']}%)\n")

    out = {"in_sample": is_rows, "lockbox_oos": lb_rows,
           "lockbox_buy_hold": bh_out,
           "overlay_config": {
               "target_vol_ann": TARGET_VOL_ANN, "vol_lookback": VOL_LOOKBACK,
               "dd_trigger": DD_TRIGGER, "dd_reentry": DD_REENTRY,
               "risk_floor": RISK_FLOOR, "note": "pre-specified textbook constants, not fitted"},
           "sleeves": list(SLEEVES)}
    json.dump(out, open("reports/phase3_overlay_results.json", "w"), indent=2, default=str)
    pd.DataFrame({"date": close[~ins].index,
                  "ensemble_raw_ret": ens_lb.values,
                  "ensemble_capped_ret": cap_lb.values,
                  "exposure_scalar": e_lb.values}).to_csv(
        "reports/phase3_overlay_lockbox_equity.csv", index=False)
    print("results → reports/phase3_overlay_results.json")
    return out


if __name__ == "__main__":
    main()
