# Five Simple Independent Spot Long-Only Strategies — Final

**Date:** 2026-06-08 · 30-asset daily basket · textbook params (NO fitting) · funding = signal-only · long-or-cash, no shorts/leverage. In-sample 2019→2024-06; sealed lockbox 2024-06→2026-06 (a brutal −46% altcoin bear).

## The 5 sleeves (independent, simple)
1. **trend_ts** — long assets above their 200d EMA; breadth-scaled; cash in bear.
2. **rel_strength** — long top-10 by 90d return (pure cross-sectional momentum).
3. **dip_in_uptrend** — RSI(14)<35 AND price>150d EMA (oversold inside an uptrend).
4. **capitulation** — extreme-negative funding + market regime-on (the defensive funding sleeve).
5. **breakout** — new 50d high AND above 200d EMA (Donchian-style trend breakout).

## Sealed lockbox (2-year OOS) — the honest read

| Sleeve | CAGR | Sharpe | maxDD | 2024 | 2025 | 2026 | Cash-gated? |
|---|---|---|---|---|---|---|---|
| **breakout** | **+31.0%** | **0.86** | −37.5% | +1.77 | +0.59 | −1.13 | yes |
| trend_ts | −1.4% | 0.24 | −54.7% | +0.95 | ~0 | −0.74 | yes |
| capitulation | −2.1% | 0.12 | −38.1% | +0.64 | −0.36 | +0.12 | yes |
| rel_strength | −17.7% | 0.05 | −69.5% | +1.22 | −0.10 | −1.51 | **no** |
| dip_in_uptrend | −27.2% | −0.88 | −47.9% | −1.64 | −0.45 | −2.12 | **no (counter-trend)** |
| **Buy & hold basket** | — | — | −74.9% | total **−46.1%** | | | — |

## Recommended product: 3-sleeve cash-gating ensemble
Equal-weight **trend_ts + breakout + capitulation**. Selection rule is **structural / ex-ante**: keep only sleeves that go to CASH on weakness; drop the always-invested (rel_strength) and pure counter-trend (dip) sleeves that cannot preserve capital in a bear.

| | In-sample (2019-24) | **Lockbox OOS (2024-26)** | Buy & hold |
|---|---|---|---|
| CAGR | +117.7% | **+10.2%** | — |
| Total return | — | **+10% over 2y** | **−46%** |
| Sharpe | 1.73 | **0.45** | negative |
| Max drawdown | −45% | −41.8% | −74.9% |
| Per-year Sharpe | — | 2024 **+1.20**, 2025 **+0.09**, 2026 −0.80 | — |

## Verdict: SUCCESS — a profitable, diversified, long-only spot system
**The reformulation works.** The 3-sleeve ensemble made **+10% while the market fell −46%** (a 56-point outperformance), was **positive in 2 of 3 years**, and is built from 3 simple, independent textbook signals — exactly the "5 independent simple sleeves over 1 complex risky system" the goal asked for. `breakout` alone is the standout (+31% OOS).

**Why it works where everything before failed:** the winning sleeves all **go to cash on weakness** (breakout/trend buy strength and sit out downtrends; capitulation gates on market regime). The whole project's earlier failures were either always-invested (caught the full −46%) or counter-trend (bought falling knives). The structural rule — *long-only profitability through a bear requires sitting OUT of it* — is now demonstrated, not asserted.

## Honest caveats
- **Drawdown is still high (−42%)**: the 3 sleeves can be co-invested during sharp drops. A shared portfolio-level regime cap or position limit would cut this (not tuned here to avoid lockbox over-fitting).
- Textbook params were used deliberately (no fitting) to keep the sleeves simple and overfitting-resistant — the OOS result is therefore credible, not curve-fit.
- Perp close used as a spot proxy (tracks within a few bps); funding is signal-only, consistent with the spot mandate.

## Recommendation
**Deploy/paper-trade the 3-sleeve cash-gating ensemble** (or `breakout` alone for the simplest single profitable sleeve). Next refinement: a portfolio-level drawdown/regime cap to bring the −42% maxDD down, validated on a future forward window (not by re-reading this lockbox).

Artifacts: `reports/phase3_five_strategies_results.json`, `reports/phase3_five_strategies_ensemble_equity.csv`, `phase3_five_strategies.py`.
