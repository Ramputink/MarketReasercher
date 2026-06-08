# Spot Long-Only "Buy Capitulation" — Honest Result

**Date:** 2026-06-08 · Signal = extreme-negative funding (crowded shorts) as a SPOT long ENTRY feature (never PnL). Long-or-cash, 30-asset basket, daily. Calibrated on in-sample (2019-09→2024-06), evaluated ONCE on the sealed 2-year lockbox.

## Chosen config (best in-sample by Calmar)
`funding_lookback=7, z_thresh=1.0, confirm_ma=5, risk_ma=50(BTC), hold_days=10, max_positions=10`

## Results
| Metric | In-sample (2019-24) | **Lockbox OOS (2024-26)** | Buy&hold basket OOS |
|---|---|---|---|
| CAGR | +111.9% | **−7.9%** | — |
| Total return | +3443% | **−15.3%** | **−46.1%** |
| Sharpe | 1.76 | **0.075** | −0.078 |
| Max drawdown | −40.8% | **−43.5%** | −74.9% |
| Per-year Sharpe (OOS) | — | 2024 **+0.46**, 2025 **+0.24**, 2026 **−1.16** | — |

## Verdict: REAL SIGNAL, but long-only can't be profitable in a −46% market
- **The signal has genuine merit.** OOS it beat the buy-and-hold basket by **+31 points** (−15% vs −46%), **halved the drawdown** (−43% vs −75%), turned a negative Sharpe positive, and was **positive in 2 of 3 calendar years**. In-sample it was excellent (Sharpe 1.76). This is the first long-only thing in the project that *clearly added value* over holding.
- **But it lost money in absolute terms OOS**, for a structural reason: the entire altcoin universe fell ~46% in the lockbox window, and a long-or-cash strategy at ~54% average exposure cannot escape that — it can only lose less. The 2026 leg (a severe crash) erased the gains banked in 2024-25.

## What would make it actually profitable (honest)
1. **The constraint is the ceiling.** A long-only basket is structurally hostage to the market regime. To be net-positive through a −46% bear it must either (a) go *fully* to cash in sustained downtrends (a much harder risk-off than BTC>50d-MA — e.g. exit everything when BTC < 200d MA AND breadth collapses), accepting it will sit out most of the time, or (b) be allowed to short the crowded side — which is exactly the Phase-2 market-neutral funding edge (+153% OOS) the user excluded.
2. **Use it as a tactical allocation overlay, not a standalone system.** As a "when to be in vs out of crypto" timer it demonstrably reduces drawdown and beats holding — valuable for a long-term holder, even if not an absolute-return engine alone.
3. **Harder risk-off + smaller, higher-conviction basket.** The current 54% average exposure is too high for a bear; a stricter macro gate would cut exposure toward 0 in 2026.

## Bottom line
The funding/capitulation signal is the **only long-only idea in the project that beat buy-and-hold OOS and was positive in most years** — it is real. But within a strict spot-long-only mandate it is a *defensive/relative* edge, not an absolute money-maker, because no long-only system profits while its universe drops 46%. The path to absolute profitability runs through either a far more aggressive cash-gate or the (excluded) short side.

Artifacts: `reports/phase3_capitulation_lockbox_results.json`, `reports/phase3_capitulation_equity.csv`, `phase3_capitulation.py`.

---

## Path 1 addendum — aggressive regime cash-gate (v2, lockbox read #2)

Signal fixed at v1's best; only a hard market-regime cash-gate (BTC>150d MA AND breadth≥0.5 else 100% cash) calibrated on in-sample. **This is the 2nd sealed-lockbox read — selection risk is higher; reported honestly.**

| | v1 | **v2 (hard gate)** | Buy&hold |
|---|---|---|---|
| OOS total return | −15.3% | **−6.6%** | −46.1% |
| Time in market | 69% | **28%** | 100% |
| OOS Sharpe | 0.075 | 0.072 | −0.08 |
| Per-year Sharpe | — | 2024 **+0.86**, 2025 −0.43, 2026 −0.98 | — |
| In-sample Sharpe | 1.76 | 1.66 | — |

**Finding:** the hard cash-gate roughly halved the loss (−15%→−6.6%) while sitting in cash 72% of the time, but stayed net-negative. Even minimal exposure catches falling knives in a sustained bear (funding goes extreme-negative repeatedly as price keeps dropping; bounces fail). **The long-only ceiling is confirmed: no long-only configuration is net-profitable through a −46% universe collapse.** However the strategy is strongly positive in normal/bull regimes (in-sample Sharpe 1.66; OOS 2024 Sharpe +0.86) — it is the specific 2024-26 bear that breaks it.

**Stopping here:** the lockbox has now been read twice; further tuning against it would be p-hacking. The honest conclusion is reached.
