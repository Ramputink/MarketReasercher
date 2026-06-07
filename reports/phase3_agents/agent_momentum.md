loosely formatted

# Momentum Strategy — Phase 3 Spot Long-Only Research Report

## 1. Diagnosis

**File:** `strategies/momentum.py`
**Registry status:** PRUNED in Round 2 (0.5% positive fitness across 14h / 36K genome evaluations)

The strategy combines three momentum filters for entry: RSI_14 in a narrow band bracketed by `rsi_long_threshold` (evolved: 63.4) and `rsi_overbought` (evolved: 69.9), `roc_12` exceeding `roc_min_magnitude` (0.55%), and `adx_14` above `adx_min` (19.3). Volume confirmation is required by default (`volume_ratio > 1.311`). A regime gate restricts entries to `["trend", "breakout"]` only.

**The strategy emits both long and short signals.** The short branch fires when `rsi < 45.5 AND rsi > rsi_oversold (19.7) AND roc < -0.55%`. This is a momentum-deterioration short, not a mean-reversion oversold short — it bets on continued downward momentum. This branch must be removed for Phase 3.

**Exits:** ATR stop at 2.844x ATR below entry (long) or above entry (short), take-profit at 6.05x ATR, and a 36h time stop. Signal strength = `min((rsi - 50) / 30, 1.0)` — well-designed for vol-scaled sizing.

**Core weakness:** The RSI entry band of 63.4 to 69.9 is only 6.5 RSI points wide. Given that RSI_14 spends ~15% of time above 60 in uptrending crypto and the overbought cutoff is at 69.9, the effective firing window is approximately 5-8% of bars. Combined with the ADX and volume requirements, the joint probability of a long signal is probably under 2% of bars. This produces very few trades per fold, failing the minimum-30-trade threshold and generating near-zero positive fitness in the GA. The genetic optimizer explored ~180 momentum-specific configurations and found no viable solution.

**Edge claim:** The PARAMS comment claims "Sharpe=0.90, WF_Sharpe=1.58, 234 trades, $409 PnL" after 500 generations. These numbers are unreliable because the 0.5% positive fitness rate means 99.5% of evaluations failed — the survived genome is a survivor-bias artifact across a very large search.

---

## 2. Spot Long-Only Edge Hypothesis

**Hypothesis:** In crypto spot markets, price momentum (as measured by RSI in a broad bullish zone and positive 12-bar rate of change) is directionally predictive over 12-48h holding periods during confirmed uptrend regimes. The edge is a reflection of:

1. Crypto's documented positive return autocorrelation at 1-6h horizons during trending markets
2. Institutional and retail momentum-chasing behavior that persists trends longer than classical finance models predict
3. Volume-confirmed breakouts from consolidation patterns following ADX expansion

The rehabilitated strategy converts to: **long when RSI (55-75) + ROC_12 > 0 + ADX > threshold + volume_ratio > threshold + bmsb_bullish = True; cash otherwise.** No shorts. The wider RSI band (vs. the 6.5-point evolved band) is the critical fix to generate enough trades for statistical validity.

---

## 3. Market Techniques

- **ATR-normalized stops and targets:** Adapt to current volatility regime. In crypto, realized vol swings from 40% to 200% annualized, making fixed-percentage stops unworkable.
- **Regime filtering (trend/breakout only):** Momentum signals fail in ranging regimes. The regime gate is the single most important structural filter.
- **Volume confirmation:** Distinguishes genuine momentum breakouts from low-liquidity noise. `volume_ratio` measures current volume vs. 20-bar SMA.
- **RSI momentum band entry (long only):** RSI as a momentum proxy — not an overbought/oversold tool here. A broad bullish zone signals that recent gains have been large and recent losses small.
- **ROC directional filter:** Ensures actual price movement (not oscillator noise) confirms direction.
- **ADX trend strength gate:** Prevents momentum entries during choppy/sideways markets.
- **Time stop as portfolio refresh:** Forces position exit after N hours, limiting tail exposure on failed momentum moves.
- **BB bandwidth percentile filter (optional):** Expanding volatility (bb_bw_percentile > 50) confirms live momentum rather than compression squeeze.
- **BMSB macro gate:** `bmsb_bullish` = price above both 20-SMA and 21-EMA. Hardcode as a mandatory filter (not an evolvable param) to avoid entering during sustained bear markets.

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Momentum | `rsi_14` | Primary entry gate: bullish zone 55-75 |
| Momentum | `roc_12` | Rate-of-change filter, confirms directional velocity |
| Trend | `adx_14` | Trend strength gate, filters ranging markets |
| Volatility | `atr_14` | Stop and TP denominator |
| Volume | `volume_ratio` | Volume confirmation of breakout |
| Volume | `volume_zscore` | Higher-resolution volume spike detection |
| Momentum | `price_acceleration` | Second derivative, acceleration filter (currently OFF) |
| Trend | `bmsb_bullish` | Macro long-only gate: price above 20-SMA AND 21-EMA |
| Volatility | `bb_bandwidth`, `bb_bw_percentile` | Expanding volatility confirmation |
| Momentum | `ret_zscore_20` | Z-score of 1-bar returns, detects anomalous bars |
| Market Regime | `regime` (passed as arg) | Restrict to trend/breakout only |
| Drawdown | `dist_from_high_20` | Near 0 = approaching breakout from recent high |
| Microstructure | `pressure_imbalance` | Intra-bar buying vs. selling pressure |

---

## 5. Params That Should Enter the Sweep/Genome

| Parameter | Type | Low | High | Note |
|---|---|---|---|---|
| `rsi_long_threshold` | float | 50.0 | 68.0 | Lower bound of RSI bullish zone; MUST be widened from evolved 63.4 |
| `rsi_overbought` | float | 65.0 | 82.0 | Upper bound; must remain > `rsi_long_threshold + 5` |
| `roc_min_magnitude` | float | 0.2 | 2.0 | Minimum ROC_12 percent for long entry |
| `adx_min` | float | 12.0 | 35.0 | Minimum ADX_14 |
| `volume_surge_threshold` | float | 0.8 | 2.5 | Minimum volume_ratio |
| `stop_loss_atr_mult` | float | 1.5 | 5.0 | ATR multiplier for stop |
| `take_profit_atr_mult` | float | 3.0 | 9.0 | ATR multiplier for TP; constrained > SL_mult |
| `time_stop_hours` | int | 12 | 96 | Hold period cap in hours |
| `use_acceleration_filter` | bool | 0 | 1 | Whether to require `price_acceleration > 0` |
| `require_volume_confirmation` | bool | 0 | 1 | Whether volume_ratio gate is active |

---

## 6. Params That Should NOT Be Optimized (Overfitting Traps)

- **`rsi_period` (fixed at 14):** Standard, sweeping it multiplies search space without signal quality gain and risks data-mining.
- **`rsi_short_threshold` and `rsi_oversold`:** Belong to the deleted short branch. Must not appear in the long-only genome.
- **`allowed_regimes` (fixed as `["trend", "breakout"]`):** Sweeping over regime list combinations risks overfit to the specific regime labeler's in-sample behavior.
- **`require_trend_regime` (fixed True):** Allowing False removes the most important structural guard and risks spurious momentum signals in choppy regimes.
- **`roc_period` (fixed at 12):** Adding a second lookback creates near-collinear redundancy with RSI period; only one momentum period parameter should be optimized.

---

## 7. Long-Only Risk Management

**Stop loss:** ATR-based hard stop at `close - stop_loss_atr_mult * ATR_14`. The backtester honors intra-bar lows vs. the stop level, so bars that gap through the stop are filled at the stop price (not at the bar low). Range: 1.5x-5x ATR. Never use fixed-dollar or fixed-percent stops in crypto.

**Take profit:** ATR-based TP at `close + take_profit_atr_mult * ATR_14`. Range: 3x-9x ATR. Enforce `TP_mult > SL_mult` in genome initialization and crossover. The wide evolved TP (6x) is appropriate for momentum — let winners run.

**Trailing stop (optional):** Once unrealized PnL > 2x ATR, trail the stop to breakeven + 0.5x ATR. Not currently in the Signal interface; would require post-entry bar-by-bar logic.

**Volatility targeting:** Use `volatility_scaled` sizing_method in RiskConfig. Target 20-25% annualized vol. Position size scales inversely with `realized_vol_20` so the strategy is not oversized during crypto bear-market vol spikes.

**Max position size:** Cap at 50% of equity per trade (max_position_pct = 0.50). This preserves cash for cost drag and prevents ruin on a single stop-loss event.

**Cooldown:** Minimum 6-bar (6h on hourly data) cooldown between entries via RiskManager `min_time_between_trades`. After a stop-loss exit, enforce an additional 12-bar cooldown (the same momentum failure is likely still in play).

**Risk-off regime gate:** Require `bmsb_bullish = True` (hardcoded, not evolvable). If BTC equity curve drops > 15% from its 30-day rolling peak, activate RiskManager circuit-breaker halt (already implemented). During "unknown" or "ranging" regimes, suppress all new long entries.

---

## 8. Relevant Metrics

| Metric | Target | Notes |
|---|---|---|
| CAGR | > 15% OOS | Annualized on 2024-2026 lockbox |
| Sharpe (annualized) | > 0.8 OOS | Evolved in-sample claim of 0.90 is suspect given 0.5% positive fitness |
| Sortino | > 1.0 | More relevant than Sharpe for asymmetric crypto returns |
| Max drawdown | < 25% | Full-sample drawdown (not fold-by-fold) per backtester WalkForwardValidator |
| Calmar | > 0.5 | CAGR / MaxDD; screens for return efficiency |
| Hit rate | 40-55% | Acceptable given evolved R:R ~2.1 (6x TP / 2.84x SL) |
| Trade count (WF) | >= 30 OOS trades | Hard minimum; current narrow RSI band likely fails this |
| Market exposure | 20-50% | Time-in-position; higher suggests over-trading |
| Avg trade duration | Compare to time_stop_hours | If >80% time-stopped, momentum signal is not predictive |
| Cost sensitivity | Sharpe degradation < 50% at 2x costs | Re-run with doubled commission and slippage |
| OOS degradation (wf_degradation) | < 50% | Evolved params showed 0.5% positive fitness = severe collapse |
| Bootstrap 5th pct Sharpe | > 0 | 1000 bootstrap resamples of OOS trade PnLs |
| Per-year Sharpe std dev | < 0.5 | Stability across individual years; reject if too regime-dependent |
| Turnover | 2-4 round-trips/week | Given 36h time stop; verify against cost budget |

---

## 9. Concrete Changes to Implement in the Sweep

1. **Remove the short branch** from `momentum_strategy`: delete the `elif rsi < rsi_short_threshold...` block (lines 77-89 in the current file). Return `None` instead of a short Signal.

2. **Rename the registry entry** from `momentum` (pruned) to `momentum_long` to signal this is a Phase 3 rehabilitation, and add it back to `STRATEGY_REGISTRY` in `auto_evolve.py`.

3. **Widen RSI entry band** in the new `param_space`: set `rsi_long_threshold` range to `(50.0, 68.0)` and `rsi_overbought` to `(65.0, 82.0)`. Add a genome-level constraint `rsi_overbought - rsi_long_threshold >= 5.0` in `random_genome` and `mutate_genome`.

4. **Remove `rsi_short_threshold` and `rsi_oversold`** from the `param_space` entirely.

5. **Add `bmsb_bullish` as a hardcoded gate** in the strategy function body: `if not current.get("bmsb_bullish", True): return None`. This is a structural guard, not an evolvable param.

6. **Add the new `momentum_long` entry** to `STRATEGY_REGISTRY` with param_space covering the 10 genome params listed in Section 5.

7. **Walk-forward window adjustment:** Use `train_days=90, val_days=20, test_days=30` for this strategy specifically to generate enough trades per fold. Document this in the registry entry.

8. **Signal count diagnostic:** In the genome evaluation function, log the per-fold trade count before computing fitness. Add an explicit warning if any fold produces < 5 trades (likely to produce Sharpe = 0 from insufficient data).

9. **Cross-asset targets:** Include BTC/USDT and ETH/USDT as mandatory cross-asset generalization targets for this strategy. Momentum is expected to generalize across large-cap crypto, and failing cross-asset generalization is a strong disqualifier.

10. **Enforce TP > SL constraint** in the new registry entry initialization (already handled generically in `random_genome` and `mutate_genome` — verify it applies to the new entry name).

---

## 10. Methodological Risks

1. **Critical: 0.5% positive fitness across 36K evaluations.** This is the strongest evidence that the momentum signal construction has no reliable edge in this data. Widening the RSI band is necessary but may not be sufficient to rehabilitate the strategy.

2. **Survivor-bias in evolved params.** The "ROBUST" evolved genome (Sharpe=0.90, WF_Sharpe=1.58) is one survivor from ~180 momentum-specific evaluations. The reported metrics may substantially overstate the true edge.

3. **Filter conjunction degeneracy.** Requiring RSI range AND ROC AND ADX AND volume simultaneously creates a conjunction of four independent conditions. If each fires with ~30% probability independently, the joint probability is ~0.8% of bars, which can produce <5 trades per walk-forward fold — below the statistical minimum for meaningful Sharpe estimation.

4. **Time stop as primary exit = random hold.** If >80% of trades exit via the 36h time stop rather than SL or TP, the outcome distribution is driven by the random 36h return from a momentum entry, not by the signal predictive power. This requires explicit monitoring (exit reason breakdown).

5. **ATR leakage at fold boundaries.** ATR_14 is computed on a rolling 14-bar window. At walk-forward fold boundaries, test fold bars 1-13 share raw price inputs with the final training bars. The 120-bar embargo in `WalkForwardValidator` covers the 100-bar BB percentile lookback but does not eliminate ATR (14-bar) leakage; the embargo is still sufficient, but researchers should be aware.

6. **RSI implementation divergence.** The codebase uses EWM-based RSI (alpha=1/period, adjust=False) which differs from Wilder's classical SMMA implementation. Evolved thresholds (63.4, 69.9) are calibrated to this specific implementation. Any comparison to published momentum research or external RSI signals requires implementation matching.

7. **Volume ratio denominator staleness in low-activity periods.** The 20-bar SMA volume denominator captures the recent baseline; during sustained low-volume periods, the bar may appear as a spike. This could generate false volume confirmations during quiet market hours in hourly data.

8. **OOS lockbox contamination risk.** The comment in PARAMS states optimization occurred on 2026-03-22 data. The OOS lockbox is defined as 2024-06-07 to 2026-06-07, meaning the lockbox overlaps significantly with the optimization period. If the genetic optimizer saw any 2024-2026 data, the lockbox is compromised. This must be audited before treating any reported WF metrics as truly OOS.

9. **No re-entry during strong trends.** The strategy exits fully and waits for the next conjunction. During 3-7 day trending moves, the 36h time stop forces exit and the ADX+RSI+ROC+volume conjunction may not re-fire until well into the move. This creates a systematic underexposure to the regime where momentum is strongest.

10. **Regime labeler quality unverified.** The strategy's core performance depends on `regime in ["trend", "breakout"]`. If the regime labeler embeds lookahead (e.g., uses future bars to classify current regime), the entire entry gate is contaminated. The labeler source is not inspected in this analysis and represents an unquantified risk.

---

## Classification: DOUBTFUL

The strategy has a documented failure history (pruned Round 2, 0.5% positive fitness), emits shorts (must be removed), and has a pathological parameter configuration (6.5-point RSI window) that produces statistically insufficient trades. The core momentum mechanism (RSI + ROC + ADX) is theoretically sound for spot long-only crypto, but the implementation requires significant structural changes before it merits compute budget. A rehabilitated `momentum_long` version with widened RSI band and short branch removed is worth a limited re-evaluation sweep (no more than 5h / 10K evaluations), with a hard kill criterion of < 1% positive fitness rate or < 30 OOS trades. Do not advance to Phase 3 OOS lockbox until the in-sample walk-forward shows consistent fitness > 0.5 across multiple seeds.
