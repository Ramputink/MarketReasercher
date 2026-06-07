"""
benchmark.selftest — prove the referee is honest
================================================

A benchmark you cannot trust is worse than none — it launders overfitting with a
green check. These tests validate the HARNESS and STATISTICS themselves, on
deterministic synthetic data, with no network. If any fail, the gauntlet's
verdicts are meaningless until fixed.

What is proven here:

    T1  Determinism            same inputs -> byte-identical results
    T2  No look-ahead          a bar-i signal fills at bar i+1 OPEN, never i CLOSE
    T3  Books balance          equity change == sum of trade pnl_net (all costs booked)
    T4  Known PnL (no costs)   single trade pnl matches the closed-form value
    T5  Cost monotonicity      more slippage/fees can only reduce pnl
    T6  Ruin detection         a leveraged loser trips the hard floor and stops
    T7  Statistics sanity      norm_ppf∘norm_cdf≈id; DSR tightens as N trials grows
"""
from __future__ import annotations

import numpy as np

from engine.backtester import Signal
from benchmark.harness import ConservativeBacktester, ExecModel
from benchmark.data_lockbox import synthetic_ohlcv
from benchmark import statistics as st


# ─────────────────────────────────────────────────────────────────────────────
# Tiny deterministic strategies (return engine.backtester.Signal)
# ─────────────────────────────────────────────────────────────────────────────

def _enter_long_once(fire_at=5):
    """Long exactly once at bar `fire_at`, then never signals again."""
    def fn(df, i, pos):
        if pos is None and i == fire_at:
            return Signal(timestamp=int(df.iloc[i]["timestamp"]), side="long",
                          strategy="t", stop_loss=None, take_profit=None,
                          time_stop_hours=None)
        return None
    return fn


def _always_long(df, i, pos):
    if pos is None:
        return Signal(timestamp=int(df.iloc[i]["timestamp"]), side="long",
                      strategy="t", stop_loss=None, take_profit=None,
                      time_stop_hours=None)
    return None


def _exec(**kw):
    base = dict(commission_rate=0.0, slippage_bps=0.0, funding_bps_per_8h=0.0,
                entry_lag_bars=1, position_pct=0.10, initial_capital=10_000.0,
                warmup_bars=5, bar_hours=1.0,
                default_sl_atr_mult=1e9, default_tp_atr_mult=1e9,
                default_time_stop_hours=10**9)
    base.update(kw)
    return ExecModel(**base)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def t1_determinism():
    df = synthetic_ohlcv(n_bars=1500, seed=11, drift=0.0002)
    a = ConservativeBacktester(_exec()).run(df, _always_long, "t")
    b = ConservativeBacktester(_exec()).run(df, _always_long, "t")
    assert a.n_trades == b.n_trades, "non-deterministic trade count"
    assert abs(a.net_pnl - b.net_pnl) < 1e-9, "non-deterministic pnl"
    return True


def t2_no_lookahead():
    df = synthetic_ohlcv(n_bars=600, seed=3, drift=0.001)
    r = ConservativeBacktester(_exec()).run(df, _enter_long_once(5), "t")
    assert r.n_trades == 1, f"expected 1 trade, got {r.n_trades}"
    entry_price = r.trades.iloc[0]["entry_price"]
    open_6 = float(df.iloc[6]["open"])     # signal at i=5 must fill at i=6 OPEN
    close_5 = float(df.iloc[5]["close"])
    assert abs(entry_price - open_6) < 1e-9, (
        f"fill={entry_price} should equal next-bar open {open_6}, NOT signal close {close_5}")
    assert abs(entry_price - close_5) > 1e-12 or abs(open_6 - close_5) < 1e-12, \
        "entry must not silently use the signal-bar close"
    return True


def t3_books_balance():
    df = synthetic_ohlcv(n_bars=2000, seed=5, drift=0.0)
    x = _exec(commission_rate=0.001, slippage_bps=5.0, funding_bps_per_8h=1.0)
    r = ConservativeBacktester(x).run(df, _always_long, "t")
    # equity change implied by trades must equal the recorded net_pnl exactly
    implied = r.trades["pnl_net"].sum() if r.n_trades else 0.0
    assert abs(implied - r.net_pnl) < 1e-6, "net_pnl inconsistent with trades"
    # and total_return must match net_pnl / initial
    assert abs(r.total_return_pct - r.net_pnl / x.initial_capital * 100) < 1e-6, \
        "total_return_pct does not reconcile with net_pnl"
    return True


def t4_known_pnl_no_costs():
    df = synthetic_ohlcv(n_bars=400, seed=9, drift=0.0008)
    x = _exec()  # zero costs
    r = ConservativeBacktester(x).run(df, _enter_long_once(5), "t")
    assert r.n_trades == 1
    tr = r.trades.iloc[0]
    entry = float(df.iloc[6]["open"])
    exit_ = float(df.iloc[-1]["close"])
    notional = x.initial_capital * x.position_pct
    expected = (exit_ - entry) / entry * notional
    assert abs(tr["pnl_net"] - expected) < 1e-6, \
        f"pnl {tr['pnl_net']} != closed-form {expected}"
    assert tr["exit_reason"] == "end_of_data"
    return True


def t5_cost_monotonicity():
    df = synthetic_ohlcv(n_bars=2500, seed=8, drift=0.0003)
    cheap = ConservativeBacktester(_exec(slippage_bps=0.0, commission_rate=0.0)).run(df, _always_long, "t")
    dear = ConservativeBacktester(_exec(slippage_bps=50.0, commission_rate=0.002)).run(df, _always_long, "t")
    assert dear.net_pnl <= cheap.net_pnl + 1e-9, "higher costs increased pnl (impossible)"
    return True


def t6_ruin_detection():
    # sharp monotonic crash + leverage => MTM must cross the floor and halt
    df = synthetic_ohlcv(n_bars=300, seed=2, drift=-0.01, vol=0.005)
    x = _exec(position_pct=3.0, ruin_equity_floor=0.0)
    r = ConservativeBacktester(x).run(df, _enter_long_once(5), "t")
    assert r.ruined, "leveraged long into a crash should ruin the account"
    return True


def t7_statistics():
    # inverse-normal round trip
    for p in (0.05, 0.5, 0.84, 0.975, 0.999):
        assert abs(st.norm_cdf(st.norm_ppf(p)) - p) < 1e-6, "norm_ppf/norm_cdf mismatch"
    # PSR in [0,1]
    rng = np.random.default_rng(0)
    rets = rng.normal(0.01, 0.05, 200)
    sr = st.sharpe_per_obs(rets)
    sk, ku = st.skew_kurt(rets)
    psr = st.probabilistic_sharpe_ratio(sr, len(rets), sk, ku, 0.0)
    assert 0.0 <= psr <= 1.0
    # expected-max Sharpe grows with N trials
    e10 = st.expected_max_sharpe(10, 1.0)
    e10k = st.expected_max_sharpe(10_000, 1.0)
    assert e10k > e10, "expected max Sharpe must grow with trial count"
    # DSR for the SAME returns must shrink as the trial budget grows
    d_small = st.deflated_sharpe_ratio(rets, n_trials=10, trials_sharpe_std=1.0)["dsr"]
    d_big = st.deflated_sharpe_ratio(rets, n_trials=50_000, trials_sharpe_std=1.0)["dsr"]
    assert d_big <= d_small, "more trials must make the Deflated Sharpe harder to pass"
    return True


_TESTS = [
    ("T1 determinism", t1_determinism),
    ("T2 no-look-ahead", t2_no_lookahead),
    ("T3 books balance", t3_books_balance),
    ("T4 known pnl (no costs)", t4_known_pnl_no_costs),
    ("T5 cost monotonicity", t5_cost_monotonicity),
    ("T6 ruin detection", t6_ruin_detection),
    ("T7 statistics sanity", t7_statistics),
]


def run_selftests() -> bool:
    print("benchmark self-tests (harness + statistics)\n" + "─" * 44)
    ok = True
    for label, fn in _TESTS:
        try:
            fn()
            print(f"  ✅ {label}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  ❌ {label}: {type(e).__name__}: {e}")
    print("─" * 44)
    print("ALL PASSED — the referee is honest." if ok else "FAILURES — fix before trusting any scorecard.")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_selftests() else 1)
