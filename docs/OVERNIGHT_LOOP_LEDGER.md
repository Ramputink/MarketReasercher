# Overnight Improvement Loop — Ledger

> Autonomous 10-hour self-paced loop. Each cycle reads this, does ONE focused
> improvement (alternating Part A intraday / Part B BMSB), validates honestly
> (out-of-sample + liquidation-aware + golden gate), commits & pushes to GitHub
> with full reasoning, and updates the documentation.

```
loop_started_at:  2026-06-22 23:09:46 CEST
loop_deadline:    2026-06-23 09:09:46 CEST   (+10h)
branch:           auto/12h-loop  (pushed to origin: Ramputink/MarketReasercher)
substrate:        auto_evolve.py --hours 10 (PID 23068, 9 cores + Metal GPU)
gate:             ./venv/bin/python tests/test_backtester_regression.py --check  (PINNED data, 12/12)
honest evaluator: tools/honest_eval.py  (liquidation-aware + true OOS holdout)
data (1h):        tmp featured caches for 15 coins
data (daily):     tmp *_daily_feat.pkl with REAL BMSB (140d/147d) for 8 majors
```

## Objectives (both parts)

- **Part A — Intraday micro-volatility (long-only, 1h):** find a robust long-only
  edge. Baseline is honest-negative (mr_vwap_reversion / vol_expansion_long land
  <1.0 OOS). Search params across coins; be honest if no robust edge exists.
- **Part B — Real BMSB (long-only, daily, long horizon):** already beats buy&hold
  on BTC/ETH/ADA with ~half the drawdown. Refine: entry/exit buffers, weekly
  resample vs daily-approx, vol-targeting (Barroso–Santa-Clara), multi-coin
  portfolio (Markowitz). Push risk-adjusted return up without overfitting.

## Anti-overfitting discipline (non-negotiable)

1. Every accepted change must improve **out-of-sample** (train/test holdout), not
   just in-sample. 2. Golden gate green after every change. 3. Track the
   multiple-testing budget — many coin×param trials inflate the best by luck;
   prefer mechanism-justified edges and re-validate winners on a second holdout.
   4. No leverage claim without liquidation modeling. 5. Long-only only.

## Stop conditions

- 10h elapsed (deadline above) · user intervenes · 3 consecutive cycles with no
  progress on either part · golden gate unrecoverable.

---

## Cycle log
(each cycle appends: timestamp · part · hypothesis · result/metrics · decision · commit · next)

### Cycle 0 · 2026-06-22 23:09 — setup
- Built & registered 3 new long-only strategies; fixed the fake-BMSB bug (20/21
  bars → real 20/21 weeks on daily). Pinned golden data (no more drift).
- Honest baseline: **Part B BMSB beats buy&hold** (BTC 17.2x vs 9.9x, DD 42% vs
  77%; ETH 22x vs 6.3x; ADA 7.4x vs 1.5x). **Part A intraday <1.0 OOS** (no edge yet).
- Launched 10h evolution substrate. Committed + pushed (commit on branch).
- **Next (Cycle 1 = Part B):** add volatility-targeted sizing to bmsb_long and a
  small entry/exit-buffer search with train/test holdout on BTC/ETH/BNB/SOL.
