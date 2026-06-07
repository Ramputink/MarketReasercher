"""
phase3_smoke.py — Phase-3 spot long-only infrastructure self-test
=================================================================

Fail-closed checks that must pass before any Phase-3 evolution run. Asserts:

  1. SEALED LOCKBOX PATH IS REAL: the frozen snapshot loads a non-empty in-sample
     window whose max timestamp is strictly BEFORE the lockbox cutoff, and that
     cutoff corresponds to ~2 years before the data end (last 730 days sealed).
  2. LONG-ONLY GATE WORKS: a strategy that emits ONLY short signals produces ZERO
     trades under long_only=True; a long-emitting strategy can trade. No trade in
     any path may have side != 'long'.
  3. SPOT FUNDING = 0: the Phase-3 backtester charges zero perp funding.

Run:  PHASE3_LONG_ONLY=1 python phase3_smoke.py
Exit 0 = all green; non-zero = a hard gate failed (do NOT evolve).
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from config import BacktestConfig, RiskConfig
from engine.backtester import Backtester, Signal

FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def test_sealed_lockbox():
    print("\n[1] Sealed lockbox path")
    from benchmark.spec import BENCHMARK_SPEC as spec
    from benchmark.data_lockbox import DataLockbox
    lb = DataLockbox(spec)
    ins = lb.in_sample()
    cutoff = lb.lockbox_cutoff_ts()
    oos = lb.lockbox()
    n = len(ins)
    check("in-sample non-empty", n > 0, f"{n} bars")
    if n > 0:
        mx = int(ins["timestamp"].max())
        check("in-sample max ts < lockbox cutoff", mx < cutoff,
              f"max={pd.Timestamp(mx, unit='ms', tz='UTC').date()} cutoff={pd.Timestamp(cutoff, unit='ms', tz='UTC').date()}")
        start = pd.Timestamp(int(ins['timestamp'].min()), unit='ms', tz='UTC').date()
        check("in-sample spans multiple years", (mx - int(ins['timestamp'].min())) > 2*365*86400_000,
              f"from {start}")
    check("lockbox (OOS) non-empty", len(oos) > 0, f"{len(oos)} bars sealed")
    cutoff_date = pd.Timestamp(cutoff, unit='ms', tz='UTC').date()
    check("lockbox cutoff ~2024-06 (last 730d sealed)", str(cutoff_date).startswith("2024-06"),
          f"cutoff={cutoff_date}")
    return ins if n > 0 else None


def _short_only_fn(df, i, position):
    if position is not None or i < 30:
        return None
    c = float(df.iloc[i]["close"])
    return Signal(timestamp=int(df.iloc[i].get("timestamp", i)), side="short",
                  strength=1.0, strategy="short_probe",
                  stop_loss=c * 1.02, take_profit=c * 0.96, time_stop_hours=48)


def _long_only_fn(df, i, position):
    if position is not None or i < 30:
        return None
    c = float(df.iloc[i]["close"])
    return Signal(timestamp=int(df.iloc[i].get("timestamp", i)), side="long",
                  strength=1.0, strategy="long_probe",
                  stop_loss=c * 0.98, take_profit=c * 1.04, time_stop_hours=48)


def _toy_df(n=400):
    rng = np.random.default_rng(7)
    price = 100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    ts = np.arange(n) * 3_600_000 + 1_600_000_000_000
    return pd.DataFrame({
        "timestamp": ts, "open": price, "high": price * 1.005,
        "low": price * 0.995, "close": price, "volume": rng.uniform(1e3, 1e4, n),
        "atr_14": price * 0.01, "adx_14": 30.0, "volume_ratio": 1.5, "_regime": "trend",
    })


def test_long_only_gate():
    print("\n[2] Long-only hard gate")
    df = _toy_df()
    bt_cfg = BacktestConfig(); risk_cfg = RiskConfig()
    bt = Backtester(bt_cfg, risk_cfg, funding_bps_per_8h=0.0, long_only=True)

    tr_s, _, _ = bt.run(df, _short_only_fn, "short_probe")
    n_short = len(tr_s)
    check("short-only strategy makes ZERO trades under long_only", n_short == 0, f"{n_short} trades")

    tr_l, _, _ = bt.run(df, _long_only_fn, "long_probe")
    n_long = len(tr_l)
    check("long-only strategy can trade", n_long > 0, f"{n_long} trades")
    if n_long > 0:
        bad = (tr_l["side"] != "long").sum() if "side" in tr_l.columns else -1
        check("no trade has side != long", bad == 0, f"{bad} non-long trades")
        fund = float(tr_l["funding"].abs().sum()) if "funding" in tr_l.columns else -1
        check("spot funding == 0 on all trades", fund == 0.0, f"sum|funding|={fund}")


if __name__ == "__main__":
    print("=" * 64)
    print("PHASE-3 SPOT LONG-ONLY SMOKE TEST")
    print("=" * 64)
    try:
        test_sealed_lockbox()
    except Exception as e:
        check("sealed lockbox path executed", False, f"EXCEPTION: {e}")
    test_long_only_gate()
    print("\n" + "=" * 64)
    if FAILS:
        print(f"RESULT: {len(FAILS)} FAILED -> {FAILS}")
        sys.exit(1)
    print("RESULT: ALL GREEN — Phase-3 infrastructure is long-only & sealed.")
    sys.exit(0)
