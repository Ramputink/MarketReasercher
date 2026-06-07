# Trend Following Strategy — Phase 3 Spot Long-Only Research Report

## 1. Diagnosis

**Strategy file:** `strategies/trend_following.py`  
**STRATEGY_REGISTRY entry:** `auto_evolve.py` lines 73–86  
**Evolved state:** Generation ~400, 1112 evaluations, Walk-forward Sharpe IS=1.62 / OOS=1.96, 181 trades

### Current Logic

The strategy is a 1–2 day horizon trend-following system with five stacked entry conditions:

1. **BMSB Gate:** Price must be above the 20-bar SMA and 21-bar EMA (bmsb_bullish) for a long, below both for a short.
2. **EMA Trend Filter:** Price above/below a 50-bar EMA defines is_uptrend/is_downtrend.
3. **ADX Strength Filter:** ADX-14 must exceed 30.17 (evolved). No trade in weak-trend/choppy conditions.
4. **Fibonacci Pullback Entry:** Price must be within 2.10% of a 0.382, 0.50, or 0.618 retracement level computed from the 50-bar swing high/low.
5. **Volume + RSI Filters:** volume_ratio >= 0.98, RSI-14 in [34, 70].

Exits: ATR-based SL (4.55x ATR-14 below entry), TP (5.58x ATR-14 above entry), 48h time stop.

### Short Signal Emission — CONFIRMED

The strategy emits `side='short'` signals when `is_downtrend=True` (price below EMA50 AND `bmsb_bullish==False`). The short SL is set above entry, short TP below entry. **This must be eliminated for Phase 3.**

### Likely Edge

The long-side edge is the "buy the dip within a trend" pattern: entering at Fibonacci retracement zones during confirmed uptrends (BMSB + EMA50 + ADX). This is a documented momentum-continuation pattern in crypto. The wide SL/TP creates positive skew when the trend resumes.

### Key Weaknesses

- BMSB uses 20/21-bar periods on hourly data — not 20/21-week as intended canonically. Makes the filter reactive, not structural.
- Strategy returns `None` when `position is not None`, so signal-exits are impossible. BMSB reversals mid-trade do not trigger early exits.
- With only 181 trades, OOS inference is statistically weak. WF Sharpe improving over IS Sharpe is a red flag.
- Fibonacci zone matching at 2.1% tolerance may be nearly always satisfied, rendering the filter cosmetic.

---

## 2. Spot Long-Only Edge Hypothesis

In a confirmed uptrend (bmsb_bullish=True AND price > EMA-50 AND ADX > threshold), crypto assets exhibit positive return autocorrelation at the 1–2 day horizon. Shallow pullbacks to Fibonacci retracement zones (0.382–0.618) within these regimes represent temporary profit-taking by weak holders, not structural reversals. Entering long at these levels — with the macro trend filter as a gating condition — should capture the trend resumption move with a positive expected payoff per trade.

The long-only conversion is natural: when is_downtrend=True, the strategy exits to cash (no position) rather than going short. This maintains the full BMSB + ADX regime filter while removing perp/short exposure entirely. The hypothesis predicts that the long-only version will have fewer trades but higher per-trade quality in bull regimes, and zero drawdown from short positions during bear-market rallies.

The hypothesis is independently testable pre-lockbox: historical hit-rate at Fibonacci entries with bmsb_bullish=True should exceed 50% with positive average net P/L, and the strategy should show positive Sharpe in 2019–2021 (bull), negative/near-zero in 2022 (bear, expecting cash), and recovering in 2023–2024.

---

## 3. Market Techniques

- **BMSB as macro bull-regime gate** — retain, but document the period scaling (20/21 bars ≠ 20/21 weeks on hourly data). Consider adding a long-term SMA (e.g., 200-bar or 500-bar) as an additional macro filter.
- **Fibonacci retracement entry timing** — retain 0.382/0.5/0.618 zones. Sweep `fib_lookback` and `fib_zone_tolerance_pct` to validate structural sensitivity.
- **ADX-14 trend-strength filtering** — retain. Sweep threshold 18–40.
- **RSI anti-extremes gate (long-only reframe)** — block entry only if RSI > upper_bound (overbought); drop the lower-bound filter entirely (oversold pullbacks are desirable entry points in trend-following).
- **ATR-based exit placement** — retain SL/TP. Tighten SL sweep range (1.5–4.0x) relative to evolved value.
- **Time stop (48h default)** — retain, sweep 24–96h.
- **Volume confirmation** — retain volume_ratio filter. Expand sweep range.
- **Buying-pressure microstructure** — add buying_pressure >= threshold as a new entry confirmation: confirms buyers are stepping in at the Fibonacci zone.
- **OBV divergence as accumulation filter** — add optional check: obv_divergence > 0 confirms underlying accumulation during price pullback.
- **Volatility-targeted position sizing** — size inversely with gk_vol or realized_vol_20 to maintain constant dollar risk across regimes.
- **Walk-forward validation with 120-bar embargo** — mandatory for all sweep evaluations on pre-2024 data.

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Trend | bmsb_bullish | Primary bull-regime gate |
| Trend | adx_14 | Trend-strength entry filter |
| Trend | roc_12 | Rate-of-change momentum confirmation |
| Price/Return | dist_from_high_20 | Depth of pullback from 20-bar high |
| Price/Return | dist_from_low_20 | Prevents entry near exhaustion low |
| Momentum | rsi_14 | Anti-overbought filter for long entry |
| Momentum | rsi_7 | Near-term momentum at entry bar |
| Momentum | ret_zscore_20 | Normalized momentum alignment |
| Volatility | atr_14 | Stop/TP placement and position sizing |
| Volatility | gk_vol | Regime volatility for size scaling |
| Volatility | vol_ratio | Short/long vol ratio; expansion confirms trend resumption |
| Volatility | bb_bw_percentile | Pre-breakout compression / extended trend signal |
| Volume | volume_ratio | Participation confirmation at entry |
| Volume | obv_divergence | Accumulation vs distribution during pullback |
| Volume | volume_zscore | Volume spike at entry bar |
| Microstructure | buying_pressure | Buyer activity at Fibonacci zone |
| Microstructure | pressure_imbalance | Net buying vs selling at entry |
| Microstructure | close_location | Bar close near high confirms rejection of low |
| Regime | regime (string) | Engine-provided regime gate (trend/breakout) |
| Funding/Sentiment (informative only) | bb_pct_b | Bollinger %B as extension/compression proxy; filter only |

---

## 5. Genome Parameters (Sweep Candidates)

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| adx_threshold | float | 18.0 | 40.0 | Evolved at 30.17; re-sweep for long-only |
| fib_zone_tolerance_pct | float | 0.5 | 4.0 | Evolved at 2.10%; wider = more entries |
| fib_lookback | int | 30 | 100 | Swing high/low window; currently fixed at 50 |
| stop_loss_atr_mult | float | 1.5 | 5.0 | Evolved at 4.55x; tighter sweep for long-only |
| take_profit_atr_mult | float | 2.0 | 8.0 | Evolved at 5.58x; maintain R/R >= 1.2 |
| min_volume_ratio | float | 0.5 | 1.8 | Evolved at 0.98; expand range |
| rsi_upper_bound | int | 60 | 82 | Long-only only: block overbought entries |
| time_stop_hours | int | 24 | 96 | Currently fixed at 48; add to sweep |
| require_higher_lows | bool | 0 | 1 | Currently off; test if it improves long-only quality |
| structure_lookback | int | 3 | 10 | Relevant only if require_higher_lows=True |
| buying_pressure_min | float | 0.35 | 0.65 | NEW: microstructure confirmation at fib level |
| obv_divergence_positive | bool | 0 | 1 | NEW: require OBV above its SMA at entry |

---

## 6. Parameters That Should NOT Be Optimized

- **bmsb_sma_period (20) / bmsb_ema_period (21):** These define the indicator identity. Sweeping them does not improve the strategy; it just selects the lookback that happened to work on training data.
- **fib_levels [0.382, 0.5, 0.618]:** Fixed mathematical Fibonacci ratios. Optimizing which ratios to include is pure data-mining with no theoretical backing.
- **trend_ema_period (50):** Standard trend EMA. Redundant with BMSB and creates multicollinearity if swept alongside BMSB periods.
- **volume_ma_period (20):** Lookback for the volume SMA denominator. Sweeping this at hourly granularity produces spurious correlations with training-period volume regimes.
- **allowed_regimes list:** Domain judgment, not a tunable signal. Sweeping regime label combinations is equivalent to fitting a regime classifier on training data.
- **require_bmsb_bullish:** Must be True for Phase 3 long-only. This is a hard constraint, not a tunable parameter.
- **rsi_lower_bound:** Should be removed entirely for long-only (see Section 2). If retained as a code variable, fix it at 15 (effectively disabled) and exclude from sweep.

---

## 7. Long-Only Risk Management

**Stop Loss:** ATR-14 based — `stop_loss = entry_price - stop_loss_atr_mult * atr_14`. Sweep 1.5–4.0x. The backtester fills stops at the exact stop price when `bar.low <= stop_loss`, which is the correct mechanism. The current evolved 4.55x is very wide for spot; tightening reduces per-trade loss but increases stop-out frequency.

**Take Profit:** `take_profit = entry_price + take_profit_atr_mult * atr_14`. Maintain R/R >= 1.2 (TP/SL ratio). Evolved 5.58x is appropriate for trend-following where large winners must offset moderate losers.

**Trailing Stop (optional, not in current code):** After price moves 2x ATR above entry, trail stop to breakeven + 0.5x ATR. After 3x ATR move, trail at 1x ATR below current close. Add as a boolean param `trail_stop_enabled` in genome.

**Volatility Targeting:** Use `volatility_scaled` sizing from RiskConfig with `vol_target_annualized=0.40`. Position size = equity * (0.40 / annualized_vol_from_ATR). Cap at 25% of equity. During high-ATR regimes (gk_vol > 2x median), this naturally reduces exposure.

**Max Position Size:** 25% of equity per trade. Signal-strength scalar (0.5–1.0) already provides further reduction.

**Cooldown:** Minimum 4-hour wait between trade entries at the RiskManager level. The engine's `entry_lag_bars=1` enforces 1-bar delay (no same-bar look-ahead).

**Risk-Off Gate:**
1. Primary: `regime not in ['trend', 'breakout']` → cash (already implemented).
2. Secondary: `bmsb_bullish == False` → cash only (enforced by long-only conversion).
3. Tertiary (recommended addition): if `gk_vol > 2.5x its 50-bar median` → skip entry or halve max_position_pct.
4. Circuit breaker: RiskManager halts trading if full-sample drawdown exceeds `max_drawdown_pct` (25% target).

---

## 8. Relevant Metrics

| Metric | Target | Notes |
|---|---|---|
| CAGR | > 15% | Benchmark vs passive BTC hold |
| Sharpe (annualized) | >= 1.0 OOS | Current IS=1.62; must survive long-only conversion |
| Sortino ratio | >= 1.2 | More relevant for long-only spot |
| Max Drawdown % | <= 25% | Hard limit; engine computes full-sample DD |
| Calmar ratio | >= 0.5 | CAGR / maxDD composite |
| Hit rate | 40–55% | Expected for wide SL/TP trend-following |
| Avg trade P/L net | >= +0.3% | Must survive 0.06% commission + 0.05% slippage |
| Trades per month | 5–15 | Too few = low confidence; too many = cost drag |
| Market exposure % | 15–40% | Lower exposure → better risk-adjusted returns |
| Cost sensitivity | Sharpe > 0 at 2x costs | Robustness test |
| Bootstrap CI (Sharpe) | Lower bound > 0 | 95% CI from 1000 trade resamples |
| Per-year Sharpe | >= 4/5 years positive | 2022 bear market is key stress test |
| OOS degradation % | <= 40% | Suspicious: current WF shows IS→OOS *improvement* |
| Profit factor | > 1.3 | Engine's robustness check requires > 1.0 |

---

## 9. Concrete Sweep Changes

1. **MANDATORY — Remove short branch:** Delete `is_downtrend` logic and `side='short'` Signal emission. Replace with `if not is_uptrend: return None`. This is the only Phase 3 conversion requirement.

2. **MANDATORY — Drop rsi_lower_bound for longs:** Remove or fix at 15 the check `if is_uptrend and rsi_val < rsi_lower_bound: return None`. Oversold pullbacks are valid Fibonacci entries.

3. **ADD fib_lookback to genome:** Sweep int [30, 100]. Currently hard-coded at 50.

4. **ADD time_stop_hours to genome:** Sweep int [24, 96]. Currently hard-coded at 48.

5. **ADD buying_pressure_min parameter:** After the volume filter, check `current['buying_pressure'] >= buying_pressure_min`. Add to genome float [0.35, 0.65].

6. **ADD obv_divergence_positive parameter (bool):** If True, require `current['obv_divergence'] > 0` at entry. Add to genome.

7. **ADD require_higher_lows to genome:** Currently evolved off. Re-expose as bool [0, 1] in the long-only sweep.

8. **CHANGE rsi_upper_bound sweep:** In long-only mode, the upper bound (65–82) is the only relevant RSI constraint. Sweep as int [60, 82].

9. **VALIDATE BMSB period scaling:** On hourly data, periods 20/21 make BMSB a 20-hour filter. Add a comment and consider a supplementary `close > close.rolling(168).mean()` (1-week SMA) as a separate bull-market macro filter that does not conflict with BMSB.

10. **ADD signal-exit for BMSB reversal:** Change the position check — when position is open, if `bmsb_bullish == False` (BMSB has turned bearish), emit an exit signal. Currently the strategy returns `None` for all bars with open position, making signal-exits impossible.

11. **ENFORCE pre-lockbox sweep:** All sweep evaluations use data from 2019-01-01 to 2024-06-06 only. The 2024-06-07 to 2026-06-07 window is sealed OOS. Use WalkForwardValidator with embargo_bars=120 for all internal validation.

---

## 10. Methodological Risks

1. **Short-branch co-evolved parameters:** The genome was evolved with both long and short branches active. ADX threshold (30.17), SL/TP multiples (4.55/5.58), and volume floor (0.98) were optimized for a combined PnL stream. These values may be suboptimal or actively harmful in long-only mode. Full re-evolution is required.

2. **BMSB period mismatch:** Using 20/21-bar periods on hourly data conflates a short-term momentum indicator with the intended long-term market-structure filter. BMSB on hourly data fires on intra-day bounces within bear markets, potentially triggering false long entries. This is a systematic model misspecification risk.

3. **Fibonacci filter activity:** With `fib_zone_tolerance_pct=2.1%` and three Fibonacci levels covering 38.2%, 50.0%, and 61.8% of the 50-bar range, price is within tolerance of at least one level a large fraction of the time. Run a filter-ablation test to confirm the Fibonacci condition meaningfully filters entries beyond pure BMSB+ADX.

4. **Statistical fragility (n=181 trades):** Bootstrap 95% CI on Sharpe with 181 observations will be very wide (likely ±0.8). The reported OOS Sharpe improving over IS Sharpe (1.96 vs 1.62) in walk-forward is statistically consistent with lucky OOS windows rather than genuine generalization. Long-only conversion will further reduce trade count, worsening this problem.

5. **Evolved parameter precision overfitting:** ADX threshold of 30.168 and fib_tolerance of 2.0956 suggest the optimizer found a local optimum with spurious precision. These fine-grained values are almost certainly training-data-specific. Sweep around these values in OOS-consistent fashion.

6. **Impossible signal-exits:** The strategy returns `None` when `position is not None`, so signal-driven exits never occur. All exits are mechanical (SL/TP/time). A BMSB reversal mid-trade does not trigger an exit. For long-only spot, this means the strategy continues to hold through BMSB bearish crossovers until either the SL or 48h time stop triggers. Adding a signal-exit on BMSB reversal is a priority improvement.

7. **Regime label dependency:** The `allowed_regimes` gate requires the engine to pass an accurate `regime` string. If the regime classifier is retrained or recalibrated, strategy behavior changes without any change to the strategy file. The gate should be implemented as a reproducible, feature-based condition within the strategy.

8. **Cost drag at low trade frequency:** With evolved wide stops (4.55x ATR), losing trades carry large absolute losses. Round-trip cost of 0.11% (commission + slippage) is modest per trade, but the asymmetry between small winners that TP early vs large losers that SL out creates non-trivial cost sensitivity. Always verify Sharpe at 2x assumed costs.

---

## Classification: CANDIDATE

The trend-following strategy is classified as a **candidate** for Phase 3 spot long-only conversion. The core long-side logic (BMSB + ADX + Fibonacci pullback entry) is structurally sound and targets a well-documented crypto momentum edge. The short branch is cleanly separable. The main risks — BMSB period scaling, small trade count, evolved parameters co-optimized with shorts, and impossible signal-exits — are all fixable through the sweep changes detailed above. The strategy is not fragile (it has genuine multi-filter structure) and not doubtful (the edge hypothesis is theoretically grounded). However, full re-evolution on long-only data is mandatory before drawing OOS conclusions.
