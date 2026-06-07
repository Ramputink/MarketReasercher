# Fisher Transform Strategy — Phase-3 Spot Long-Only Research Report

## Classification: DOUBTFUL

The Fisher Transform strategy has a coherent theoretical edge (mean-reversion at range extremes) but faces serious structural obstacles for long-only spot deployment: extreme signal sparsity after short-branch removal, a recursive warm-up bias in the per-bar slice computation, and evolved parameters that were optimized jointly across both long and short branches.

---

## 1. Diagnosis

**Current logic.** The strategy computes Ehlers' Fisher Transform on the midpoint of the rolling high-low range over a configurable `period` (default 18). The raw value is double-smoothed — once before the log transform (0.5-weight blend with prior) and once after (another 0.5-weight recursive blend). The result is compared to a `signal_threshold` (default 1.982). Two signal branches:

- **Short (bearish reversal):** `ft_prev > threshold AND ft < ft_prev AND ft < ft_signal` → `side="short"` with stop above entry and TP below.
- **Long (bullish reversal):** `ft_prev < -threshold AND ft > ft_prev AND ft > ft_signal` → `side="long"` with stop below and TP above.

**Does it emit shorts?** Yes. The short branch is fully active. The evolved PARAMS were optimized with both branches live. Removing the short branch is mandatory for Phase-3.

**Current filters.** ADX cap at 35.3 (filter=True by default) blocks entries in strong trends. An optional divergence filter exists (disabled). Volume filter disabled. Regime gate allows `["lateral", "mean_reversion", "trend"]` — the inclusion of "trend" is inconsistent with the ADX cap.

**Evolved result context.** The genetic report (gen 190, 1104 evals) shows WF_Sharpe=2.19 with only 34 trades. At N=34 total (both branches), the expected long-only trade count is ~17, yielding near-zero statistical power. The Sharpe estimate's 95% CI likely spans from negative to very high values.

**Core edge.** Identifying price exhaustion at range extremes via a Gaussian-normalized oscillator. Theoretically sound for mean-reversion in ranging/consolidating markets. **Weakness:** In crypto's predominantly trending environment (2020-2021 bull, 2022 bear, 2023-2024 recovery), sustained directional moves keep the Fisher pinned at extremes for extended periods, making crossovers unreliable.

**Computation note.** The strategy recomputes Fisher from scratch on each bar using a 150-bar trailing slice (`lookback = min(bar_idx+1, 150)`). Because the recursive smoothing initializes `value=0.0` and `fisher[i-1]=0` at the start of each slice, the warm-up period produces Fisher values that are biased toward zero regardless of actual price history. This creates spurious crossovers in the first ~period bars of each slice, which could generate false signals.

---

## 2. Spot Long-Only Edge Hypothesis

After a sustained price decline that drives the Fisher Transform below a deeply negative threshold (approximately -1.8 to -2.2), the 50% halving of selling pressure — signaled by the Fisher crossing above its own lagged signal line — identifies high-probability oversold-bounce entry points. Combined with a macro regime gate (price above BMSB), an RSI oversold confirmation, and the existing ADX cap to avoid entering during strong downtrends, the strategy aims to capture 3-7% bounces from range lows with defined risk (ATR stop below entry).

The long-or-cash framing: position = LONG when Fisher long crossover fires with all filters satisfied; position = CASH otherwise. No short exposure at any time.

This edge is most credible in BTC/ETH during periods of sideways consolidation or moderate corrections within broader bull markets, where Fisher extremes represent genuine sentiment exhaustion rather than the onset of sustained downtrends.

---

## 3. Market Techniques

1. **Mean-reversion entry at oscillator extremes** — Fisher crossover from deeply negative levels captures oversold-bounce dynamics.
2. **ATR-normalized stop and take-profit** — Volatility-adaptive risk management ensures the R:R structure holds across low-vol and high-vol regimes.
3. **Regime gating via ADX cap** — Prevents mean-reversion entries during strong trending markets where Fisher extremes persist.
4. **BMSB macro filter** — Bull-market support band determines the macro regime; long signals below BMSB are skipped to avoid catching falling knives in bear markets.
5. **Optional bullish divergence confirmation** — Price lower low + Fisher higher low strengthens the reversal hypothesis at the cost of further reducing trade frequency.
6. **Time stop as secondary exit** — Caps exposure when the anticipated bounce does not materialize; reduces drawdown from stale positions.
7. **Strength-weighted position sizing** — Fisher magnitude above threshold scales position size, ensuring largest exposure when the signal is most extreme.
8. **Walk-forward calibration with embargo gaps** — The engine's WalkForwardValidator with 120-bar embargo prevents rolling-feature leakage across train/val/test boundaries, critical for the 100-bar bb_bandwidth_percentile feature.

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Momentum | `rsi_7` | Fast oversold confirmation; require < 35 at Fisher long signal |
| Momentum | `rsi_14` | Medium-term momentum; RSI rising while Fisher crosses suggests absorption |
| Trend | `adx_14` | Existing ADX gate; consider also gating on ADX direction (falling = trend exhaustion) |
| Volatility | `atr_14` | Stop and TP sizing (already used) |
| Volatility | `gk_vol` | Garman-Klass vol regime; skip entries when gk_vol is in top decile of 90-day distribution |
| Volatility | `bb_pct_b` | Below 0.2 corroborates price at range low; strengthens Fisher long signal |
| Volatility | `bb_bw_percentile` | Avoid entries during extreme compression (breakout rather than mean-reversion risk) |
| Volume | `volume_ratio` | Above-average volume on crossover bar signals buyer absorption |
| Volume | `obv_divergence` | Positive divergence (OBV rising vs. price falling) confirms accumulation |
| Price/return | `dist_from_low_20` | Gate: price within 15% of 20-bar low confirms signal is at genuine range extreme |
| Price/return | `ret_zscore_20` | Very negative z-score corroborates oversold condition |
| Market regime | `bmsb_bullish` | Macro long-only gate: skip all longs when price is below BMSB |
| Drawdown | `dist_from_high_20` | Deeply negative value confirms entry after a drawdown; consistent with bounce hypothesis |

Funding/sentiment features: NOT used for PnL; may be used as informative filters only (e.g., extreme negative funding as an additional oversold confirmation, never as a direct carry trade).

---

## 5. Genome Parameters (Sweep)

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `period` | int | 8 | 24 | Fisher rolling window; wider = more lag but smoother |
| `signal_threshold` | float | 1.4 | 2.6 | Fisher extreme level; higher = rarer signals |
| `adx_max` | float | 25.0 | 45.0 | Trend-strength gate ceiling |
| `stop_loss_atr_mult` | float | 1.5 | 3.5 | Hard stop below entry |
| `take_profit_atr_mult` | float | 2.0 | 5.5 | TP above entry; keep TP/SL >= 1.5 |
| `time_stop_hours` | int | 12 | 72 | Currently fixed at 24; must enter sweep |
| `require_divergence` | bool | 0 | 1 | Price-Fisher bullish divergence confirmation |
| `divergence_lookback` | int | 8 | 20 | Only active when require_divergence=True |
| `rsi_oversold_threshold` | float | 25.0 | 45.0 | NEW: RSI_14 must be below this at signal |
| `require_bmsb_above` | bool | 0 | 1 | NEW: Macro long-only gate via BMSB |

---

## 6. Parameters NOT to Optimize (Overfitting Traps)

- **`allowed_regimes`** — Evolving regime label strings couples the strategy to the classifier's internal output distribution; the ADX filter provides a cleaner and more stable structural gate.
- **`volume_threshold`** (when `require_volume=False`) — With only ~17 expected long-only trades, adding a volume threshold to the sweep fits noise from a handful of bars.
- **`divergence_lookback`** — If `require_divergence` is False (a separate bool in the sweep), this parameter is completely irrelevant, creating a dead genome dimension that absorbs search budget without benefit.
- **Fisher smoothing constant (0.5)** — Ehlers' original formulation; changing it makes the oscillator non-standard with no theoretical backing.

---

## 7. Long-Only Risk Management

**Stop loss:** Hard ATR stop below entry at `stop_loss_atr_mult * atr_14` (sweep 1.5-3.5x). Set at signal time, never moved downward. The backtester already handles stop_loss via `position.stop_loss` check against bar lows.

**Take profit:** Fixed ATR-based TP at `take_profit_atr_mult * atr_14` (sweep 2.0-5.5x). Require TP/SL ratio >= 1.5 in the genome constraint to maintain positive expected value even at 45% hit rate.

**Trailing stop:** Not recommended for mean-reversion trades. Bounces are typically fast and shallow; trailing stops tend to give back gains. Rely on fixed TP or time stop.

**Volatility targeting:** Use `volatility_scaled` sizing method (already supported by the backtester's `_compute_position_size`): target annualized vol fraction 0.20 of equity per position, sized using `atr_14/close` as the daily vol proxy.

**Max position size:** Hard cap at 30% of equity per position. With one position allowed at a time (the strategy returns None when `position is not None`), maximum instantaneous exposure is 30%.

**Cooldown:** 24-hour minimum between new long entries after a stop-loss exit. Prevents rapid re-entry into a continuing downtrend.

**Risk-off regime gate:**
1. ADX gate (already implemented): skip if `adx_14 > adx_max`.
2. NEW macro gate: skip if `bmsb_bullish == False` (price below both BMSB SMA and EMA).
3. Extreme vol gate: skip if `gk_vol` is in the top 10% of trailing 90-day distribution.
4. Circuit-breaker: the engine's RiskManager already halts new entries after sustained equity drawdown; ensure it is active (`use_risk_manager=True`).

---

## 8. Relevant Metrics

| Metric | Target (2yr OOS) | Notes |
|---|---|---|
| CAGR | > 15% | Compare to BTC buy-and-hold baseline over same window |
| Sharpe (annualized) | > 0.9 | WF estimate of 2.19 will not hold with N~17; expect significant haircut |
| Sortino ratio | > 1.2 | Long-only; downside vol is the relevant risk measure |
| Max drawdown % | < 25% | Full 2-year equity curve, not per-fold maximum |
| Calmar ratio | > 0.6 | In crypto's volatile environment this is achievable only in favorable regimes |
| Hit rate | 40-55% | Counter-trend entries typically below 50%; 40% requires R:R > 1.5 |
| Number of trades (OOS) | >= 20 | Below 20 trades, no statistical inference is valid; flag as insufficient if < 20 |
| Avg trade duration | 12-48h | Longer suggests time stop is dominant exit mechanism |
| Market exposure % | 5-20% | Long-only mean-reversion; mostly in cash is expected and acceptable |
| Cost sensitivity | < 20% Sharpe degradation at 3x costs | Wide ATR stops mean large cost as % of PnL; verify |
| Bootstrap CI on Sharpe | 95% CI lower bound > 0.5 | With N~17, this is the critical diagnostic; CI likely spans negative values |
| Per-year Sharpe stability | No year with Sharpe < 0 | Year-over-year consistency check on pre-lockbox data |

---

## 9. Concrete Changes to Implement in the Sweep

1. **Remove the short branch completely.** Delete lines 119-121 (`side = "short"`) and lines 155-157 (short stop/TP calculation) from `fisher_transform.py`. Only return a Signal when `side == "long"`. Add a unit test asserting no Signal with `side="short"` is ever emitted.

2. **Add `time_stop_hours` to the `param_space` in `auto_evolve.py`** with range `("int", 12, 72)`. The current fixed value of 24 is an untested assumption.

3. **Add `rsi_oversold_threshold` parameter** (float, 25.0-45.0): at the signal bar, check `current.get("rsi_14", 50) < p["rsi_oversold_threshold"]`; if not, return None. Read `rsi_14` from the already-computed feature column.

4. **Add `require_bmsb_above` parameter** (bool): when True, check `current.get("bmsb_bullish", True)` at signal bar; if False, return None. This implements the macro long-only gate via the bull-market support band.

5. **Add `require_bmsb_above` and `rsi_oversold_threshold` to `STRATEGY_REGISTRY["fisher_transform"]["param_space"]`** in `auto_evolve.py`.

6. **Add `dist_from_low_20` gate** (informative hardcoded threshold, not swept): add a check `dist_from_low_20 = current.get("dist_from_low_20", -1.0)` and skip if > -0.05 (price not near 20-bar low); this is a logic constraint, not a genome param, preventing mid-range Fisher triggers.

7. **Fix the Fisher warm-up bias** by increasing the minimum slice requirement: if `lookback < period * 5`, return None (ensures at least 5x period bars for the recursive smoothing to stabilize before the signal bar).

8. **Remove "trend" from `allowed_regimes`** in PARAMS: a mean-reversion strategy should not trade in trend regimes; allowed should be `["lateral", "mean_reversion"]` only.

9. **Add `obv_divergence` as an informative strength booster** (not a gate): `if current.get("obv_divergence", 0) > 0: strength += 0.05` — adds signal confidence without reducing trade count.

10. **In `auto_evolve.py` param_space, extend `period` upper bound from 20 to 24** to allow slower configurations appropriate for hourly crypto data.

---

## 10. Methodological Risks

1. **Extreme sparsity.** With 34 total trades (both branches) in-sample and ~17 expected long-only trades in the 2-year OOS window, all performance metrics have confidence intervals spanning from deeply negative to very high. The WF_Sharpe of 2.19 is statistically meaningless at this sample size. Bootstrap resampling with 1,000 replications is mandatory before any promotion decision; if the lower 95% CI Sharpe bound is below 0.3, do not promote.

2. **Recursive warm-up bias in per-bar slice computation.** The Fisher is computed fresh on a 150-bar slice starting from `value=0.0` and `fisher[i-1]=0`. This means the first ~period bars of each slice produce Fisher values biased toward zero, not the true running state. A price history that was genuinely extended at the Fisher extremes before the slice starts will appear to return to neutral, generating false crossovers. Fix: maintain a persistent Fisher state across bars or use a sufficiently large initialization window (>= period * 5) before relying on any signal.

3. **Short-branch co-evolution contamination.** The evolved params (including `signal_threshold=1.982`, `adx_max=35.3`, `period=18`) were optimized with both long and short signals contributing to the fitness function. The genetic algorithm may have selected a threshold that balanced both signal types, not one that is optimal for long-only mean-reversion. Re-evolving from scratch with short branch removed is necessary before trusting the param_space ranges.

4. **ADX cap inconsistency with regime gate.** The ADX cap (adx_max=35) blocks trending entries, but `allowed_regimes` includes "trend". These two filters are logically opposed: the ADX cap excludes high-trend environments while the regime gate permits them. The inconsistency means one of the two filters is effectively a no-op, wasting genome complexity.

5. **24-hour time stop dominance.** With a 24-hour fixed time stop and only hourly bars, the strategy has at most 24 candles to produce a profitable exit. If the bounce takes 2-5 days (common for crypto mean-reversion), the position is stopped out at break-even or a small loss before the TP is reached. This structurally caps the P&L on the long branch and may explain why few trades in the training sample showed large winners.

6. **OOS lockbox statistical power.** With ~17 long-only trades expected over 2 years, the OOS lockbox has near-zero power to distinguish a Sharpe of 0.5 from 1.5. A single large losing trade (BTC drop of 20%+ against a Fisher long) can condemn the strategy even if the edge is real. Conversely, one large win can mask a structurally unprofitable strategy. This is a fundamental limitation that cannot be solved by the sweep — the strategy simply may not generate enough trades to be evaluated rigorously.

7. **BMSB feature period mismatch.** The `bmsb_bullish` feature in `features.py` uses SMA/EMA periods of 20 and 21 (in the timeframe's native units, which for hourly data = 20 and 21 hours, not weeks). The BMSB concept is intended as a weekly indicator (20-week SMA, 21-week EMA ≈ 3360h and 3528h). Using 20h and 21h periods produces a very short-term band that is not the macro regime indicator it is intended to be. Verify the actual period values passed to `build_all_features` before relying on `bmsb_bullish` as a macro filter.
