# Honest Feasibility Report — "10x in 6 months"

**Date:** 2026-06-03 · **Branch:** `auto/12h-loop` · **Asset:** XRP/USDT 1h (+BTC/ETH/SOL/ADA)
**Mandate:** Build a system that backtests to ≥10x initial capital over 6 months — **without cheating.** If impossible, say so.

## Verdict (one line)

**A legitimate, non-overfit 10x-in-6-months is NOT achievable on this data.** Every path that
reaches 10x in a backtest does so through one of the cheats the mandate forbids: in-sample
curve-fitting, single-window cherry-picking, or leverage with no liquidation modeling. When you
remove those, the honest out-of-sample edge is roughly **1.2–1.3x over ~5–6 months** (≈ break-even
to modestly positive), not 10x. This was confirmed by 4 independent lever studies and 4 adversarial
verifiers that reproduced the numbers and tried to break each result (`any_honest_10x = false`).

## How returns are actually produced (baseline)

Best git-saved genomes, full 364-day backtest, real costs (0.1% commission + 5 bps slippage),
default conservative sizing (`max_position_pct=0.10`, vol-targeted, **no leverage**, one position):

| strategy | total return | multiple | Sharpe | max DD | trades |
|---|---|---|---|---|---|
| donchian_breakout | +4.5% | 1.045x | 1.48 | 3.0% | 149 |
| volatility_squeeze | +3.6% | 1.036x | 2.73 | 0.6% | 48 |
| vol_regime_arb | +2.6% | 1.026x | 3.83 | 0.26% | 36 |
| **buy & hold XRP** | **−44%** | **0.56x** | — | — | — |

The edges are real but tiny in absolute terms because the engine optimizes **Sharpe** and deploys
almost no capital. **Context that makes 10x extreme:** over this year XRP −44%, SOL −33%, ADA −64%,
BTC −19%, only ETH ~flat (+10%). A broadly bearish year — all return must come from alpha/shorting.

## The "10x" mirage and why it's a cheat

Raising the exposure cap (true leverage) makes the in-sample numbers explode:

| vol_regime_arb leverage | in-sample full-period | max DD |
|---|---|---|
| 5x | **10.0x** | 24.8% |
| 8x | **30.0x** | 37.5% |
| 20x | 478x | 74.5% |

This looks like "10x achieved." It is not honest, for three independently fatal reasons:

1. **In-sample selection.** These genomes are the survivors of 15,440 fitness evaluations on this
   exact XRP series. Reading their return on the same data is curve-fitting. Perturbing the params
   ±10% collapses the headline (e.g. 5.93x → median 2.24x) — it sits on a sharp overfit spike.
2. **Single-window dependence.** Split into independent, equity-reset windows, the big number is one
   lucky quarter compounded. **No strategy clears 10x in any independent 6-month window.** Best was
   vol_regime_arb H1=9.49x — but its *other* half was 3.57x, donchian's was a **loss** (0.76x).
3. **Leverage with no liquidation modeling.** The frozen engine fills stops *exactly* at the stop
   price and never models gap-through/liquidation. **XRP printed a −49.75% open→low bar** — a
   leveraged long is liquidated at just **2x** there. So "no ruin at 8x–20x" is a simulation
   artifact; a real account is wiped out.

## The decisive test: true out-of-sample hold-out

Select params on the **first 60%** of data only, freeze, measure on the **last 40% (never seen):**

| | TRAIN | TEST (unseen) |
|---|---|---|
| vol_regime_arb best (liquidation-aware, L=5) | 1.86x | **1.21x** (DD 30%, ~5 mo) |
| volatility_squeeze | train Sharpe 2.82 / huge | **out-of-sample LOSS** (overfit trap) |

The honest forward expectation is **~1.2x over ~5 months** at aggressive 5x leverage — and even that
is positive in only 2 of 4 independent windows. That is **~8–40x short of the 10x target.**

## Legitimate levers tested (all fail to reach 10x honestly)

- **Multi-asset portfolio (A):** *Hurts.* The XRP-tuned genome loses on BTC/ETH/ADA (proof of
  overfit); equal-weighting dilutes the one winner. Best portfolio 2.12x, two of four windows negative.
- **Regime-switch ensemble (B):** *Overfit.* In-sample 10.4x, but walk-forward (learn map on H1, test
  H2) → **0.725x, a loss**; the optimal regime→strategy map is unstable across periods (fit to noise).
- **Short-alpha (C):** *Crash beta, not alpha.* trend_following's entire return is short-side, but it
  evaporates in Q4 when XRP stopped falling. Not robust across windows.
- **Out-of-sample hold-out (D):** the decisive test above — no honest 10x survives.

## What 10x in 6 months would actually require (and why it's ruin-guaranteed)

Return ≈ Sharpe × volatility. The best robust edge found is Sharpe ≈ 3.8. To make ~1000% in 6 months
(~1400% annualized) at Sharpe 3.8 requires **≈370% annualized volatility**. At that volatility the
probability of a >100% drawdown (ruin) *before* reaching the target is ≈1. The same leverage that
makes the backtest "hit 10x" is the leverage that liquidates the live account on a single −50% bar.

## Honest, defensible frontier (at the ≤50% DD budget you set)

- **Most robust:** vol_regime_arb, ~5x leverage → ~1.2x out-of-sample over ~5 months, all-quarter
  consistency at lower leverage; live-plausible if (and only if) liquidation risk is respected.
- **Do NOT trust** any single in-sample full-period multiple as a forward number.

## Improvements delivered this session

1. **`tools/honest_eval.py`** — a reusable **liquidation-aware, out-of-sample** evaluator. Models
   per-trade Maximum Adverse Excursion → realistic liquidation at leverage, and does a genuine
   train/test param search (no contaminated holdout). This is the missing honesty guard.
2. Confirmed the frozen engine has **no lookahead** (regime/features are causal) and applies real
   costs — so the framework itself is honest; only mis-reading its output produces fake 10x.
3. Ran a full evolution substrate (9 cores + Metal GPU, ~1h, 3,760 evals, 424 robust) — no new
   champion beat the git-saved ones (fresh runs restart from gen 0; resume is the highest-ROI fix).

## Recommendations

1. **Reframe the objective** from "10x in 6 months" (ruin-guaranteed) to "maximize out-of-sample
   risk-adjusted return at a stated DD budget." Honest target: ~1.2–1.5x per 6 months at moderate
   leverage.
2. **Adopt `tools/honest_eval.py` as the reporting gate** — never report an in-sample full-period
   multiple again; report the out-of-sample, liquidation-aware number.
3. **Fix evolution resume** (checkpoint → seed next run) so multi-run searches compound.
4. If leverage is ever used, **model liquidation/funding in the live path**, not just the backtest.

## Process note (data refresh & golden harness)

Baseline scripts triggered a Binance re-fetch that overwrote the gitignored
`data/binance_XRP_USDT_1h.parquet`, advancing the rolling 365-day window ~1 day. This invalidated the
data-dependent golden snapshot (engine code is **bit-identical** — verified via git diff). The golden
was **regenerated against current data** and is green (12/12, tol 1e-9). Recommend pinning a fixed
data snapshot for the golden so it doesn't drift when data refreshes.
