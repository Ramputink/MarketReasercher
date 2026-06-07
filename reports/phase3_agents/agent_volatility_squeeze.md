# Volatility Squeeze — Phase 3 Spot Long-Only Research Report

## 1. Diagnosis

**Strategy mechanic.** The strategy implements a TTM Squeeze variant. It identifies bars where the 20-period Bollinger Band (2.0 std) is nested inside a 20-period Keltner Channel (EMA ± ATR × keltner_atr_mult). When the price has been squeezed for at least squeeze_min_bars consecutive bars (identified via bb_bw_percentile < 20), and the current bar is no longer squeezed, the strategy checks for a volume surge (volume_ratio > volume_surge_threshold) and determines direction via a linear regression slope over the last momentum_lookback bars.

**Short signals are emitted.** Lines 108–118 of volatility_squeeze.py emit `Signal(side="short")` when slope < 0. This must be converted to long-or-flat for Phase 3.

**Current evolved PARAMS (gen 472, 5375 evals):**
- keltner_atr_mult = 1.1686 (tight; detects narrow compressions)
- squeeze_min_bars = 3
- momentum_lookback = 13
- require_volume_surge = True, volume_surge_threshold = 1.9008 (high bar)
- stop_loss_atr_mult = 3.7943 (wide stop)
- take_profit_atr_mult = 4.5961
- time_stop_hours = 36 (very short for breakout trades)
- allowed_regimes = ["breakout", "lateral"]
- adx_min = 0.0 (disabled)

**Reported walk-forward metrics:** Sharpe=3.31, WF_Sharpe=3.94, PF=3.72, 49 trades.

**Edge assessment.** The core mechanism is sound: volatility compressions followed by volume-confirmed expansions represent genuine informational events in crypto markets. The positive drift bias of crypto spot favors a long-only variant. The weakness is that removing shorts halves signal frequency (49 → ~25 expected long-only trades), increasing Sharpe uncertainty. The 36-hour time stop is the single largest value-destruction mechanism for a breakout strategy.

**Likely weakness.** ADX gate is disabled; high volume threshold (1.9x) may generate too few entries on some assets; slope direction computed on pre-release bars can disagree with actual expansion direction.

---

## 2. Spot Long-Only Edge Hypothesis

After a verified volatility compression (BB inside Keltner ≥ N bars, bb_bw_percentile < threshold), crypto spot assets exhibit sustained upward momentum when the release is volume-confirmed and the regression slope is positive. The long-only hypothesis:

> Enter long when: (1) squeeze has held ≥ squeeze_min_bars, (2) current bar is no longer squeezed, (3) volume_ratio > threshold, (4) linear regression slope > 0, (5) close is near or above the 20-bar high, (6) price is above the Bull Market Support Band. Hold with trailing stop or extended time stop. Go to cash otherwise.

The crypto spot positive drift bias (~60% CAGR for BTC 2019-2026) means capturing upside squeeze releases while sitting in cash on downside releases is expected to outperform the market-neutral original, especially after removing short-side slippage and funding costs.

---

## 3. Market Techniques

1. **TTM Squeeze** (Bollinger inside Keltner) — core compression detection mechanism
2. **Linear regression slope** — directional momentum at release, avoids reliance on a single bar signal
3. **Volume surge confirmation** — filters false releases where price expands on low conviction
4. **bb_bw_percentile rolling rank** — normalizes squeeze depth across different volatility regimes
5. **ATR-based stop and take-profit** — adapts to the asset's realized volatility
6. **Regime gating** — restricts entries to market states compatible with breakout dynamics
7. **BMSB (Bull Market Support Band)** — 20-SMA / 21-EMA macro trend filter
8. **Trailing stop** — replaces the rigid 36h time stop to let multi-day breakouts run
9. **ADX threshold** — post-release trend strength gate to confirm the move has follow-through
10. **Walk-forward validation with 120-bar embargo** — prevents rolling-feature leakage at fold boundaries

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Volatility | `bb_bw_percentile` | Core squeeze depth ranking; low = deep squeeze |
| Volatility | `bb_bandwidth` | Raw bandwidth; delta at release bar confirms expansion |
| Volatility | `atr_14` | Stop/TP sizing and Keltner channel width |
| Volatility | `gk_vol` | Garman-Klass realized vol; cleaner than OHLC vol during compression |
| Volatility | `vol_ratio` (short/long) | Secondary compression indicator |
| Trend | `adx_14` | Post-release trend strength gate |
| Trend | `bmsb_bullish` | Macro bull-market gate; only enter long above BMSB |
| Momentum | `roc_12` | 12-bar rate of change, supplementary direction signal |
| Momentum | `rsi_14` | Avoid overbought entries (rsi_14 > 70) at release |
| Momentum | `price_acceleration` | Second derivative confirms momentum building |
| Volume | `volume_ratio` | Primary volume surge gate (already used) |
| Volume | `volume_zscore` | Z-score spike detector, more robust than ratio |
| Volume | `obv_divergence` | OBV vs SMA; positive divergence confirms accumulation during squeeze |
| Price | `dist_from_high_20` | Price near 20-bar high confirms upward breakout direction |
| Price | `close_location` | High close_location at release confirms bullish bar |
| Sentiment (filter only) | Funding rate direction | Do NOT enter long when funding extremely negative; treat as warning flag, never as PnL source |

---

## 5. Parameters for the Sweep/Genome

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `keltner_atr_mult` | float | 0.8 | 2.0 | How tight Keltner must be; explore looser definitions |
| `squeeze_min_bars` | int | 2 | 12 | Bars of confirmed compression; allow deeper squeezes |
| `momentum_lookback` | int | 5 | 30 | Regression slope window |
| `volume_surge_threshold` | float | 1.0 | 3.0 | Volume at release; current 1.9 may be too selective |
| `stop_loss_atr_mult` | float | 1.5 | 5.0 | Wide stops needed for breakout; current 3.79 sensible |
| `take_profit_atr_mult` | float | 2.5 | 9.0 | Let long-only winners run further |
| `time_stop_hours` | int | 24 | 120 | Extend for multi-day breakout holding |
| `adx_min` | float | 0.0 | 30.0 | Allow evolution to enable ADX gate |
| `squeeze_depth_pct_max` | float | 10.0 | 30.0 | NEW: make bb_bw_percentile threshold evolvable |
| `rsi_overbought_filter` | float | 65.0 | 85.0 | NEW: skip long if rsi_14 > this at release |

---

## 6. Parameters NOT to Optimize

| Parameter | Reason |
|---|---|
| `bb_period` (=20) | TTM Squeeze standard; changing it redefines the indicator identity |
| `bb_std` (=2.0) | TTM Squeeze standard; arbitrary deviations create non-comparable squeeze definitions |
| `keltner_period` (=20) | Must match bb_period for the BB-inside-KC comparison to be meaningful |
| `require_volume_surge` (=True) | Boolean gate; toggling it is a free parameter that invites overfitting |
| `allowed_regimes` | Regime list changes without principled basis invite curve fitting to regime labels |

---

## 7. Long-Only Risk Management

**Stop loss:** Hard ATR stop at entry_price - stop_loss_atr_mult × atr_14. Enforced intrabar via the backtester's low-price check. Range 1.5–5.0× ATR. Important note: ATR is suppressed during the squeeze, so the stop is set against compressed ATR and may be effectively tighter than intended once vol expands; consider using post-release ATR (one bar after squeeze exit) for stop calculation.

**Take profit:** Hard ATR TP at entry_price + take_profit_atr_mult × atr_14. Range 2.5–9.0× ATR in long-only context; TP must exceed SL (enforced in genome mutation).

**Trailing stop:** Once unrealized gain > 2.0 × atr_14 × (position_size / entry_price), raise stop to breakeven. Once gain > 3.0 × atr_14, trail at max(current_stop, rolling_high_since_entry - 1.5 × atr_14). This is the primary exit mechanism for winning breakouts and replaces the rigid 36h time stop.

**Volatility targeting:** Use `sizing_method = "volatility_scaled"` in RiskConfig. Target 15–20% annualized vol per position. At typical crypto ATR/price ~2–4%, this implies 15–30% of equity per trade.

**Maximum position size:** 25% of equity per trade (config.max_position_pct = 0.25). Any single squeeze release cannot dominate the book.

**Cooldown:** Minimum 12h between new entries. Prevents rapid re-entry after a time-stop or stop-loss exit in a choppy market.

**Risk-off regime gate:**
1. Regime must be in ['breakout', 'lateral'] (current gate; keep).
2. Skip long entries if close < bmsb_sma (below Bull Market Support Band).
3. Skip entries if gk_vol z-score (20-bar rolling) > 2.5 (extreme realized vol signals dislocation, not tradeable breakout).

---

## 8. Relevant Metrics

| Metric | Target / Interpretation |
|---|---|
| CAGR | Primary return metric on equity curve |
| Sharpe ratio (annualized) | Risk-adjusted return; target > 1.5 for long-only spot |
| Sortino ratio | More appropriate than Sharpe for right-skewed breakout P&L distributions |
| Maximum drawdown % | Hard constraint; target < 20% |
| Calmar ratio (CAGR / maxDD) | Combined return-risk scalar |
| Hit rate % | Expect 45–60% for directional breakout; confirm improvement from short removal |
| Profit factor | Must exceed 1.5 after removing short arm |
| Average trade PnL (net) | Must be clearly positive vs. ~2× commission + slippage |
| Turnover (trades/month) | 1–5/month expected; verify cost drag is negligible at this frequency |
| Market exposure % | Target 15–35%; selective approach |
| Cost sensitivity | Recompute Sharpe at 2× commission and 2× slippage; edge must survive |
| Bootstrap/MC Sharpe CI | 1000 iterations, trade-level resampling; target lower bound > 1.0 |
| Per-calendar-year performance | Must show profitable years in both bull (2020-21, 2023-24) and sideways/bear (2022) markets |
| OOS degradation % | Walk-forward test Sharpe / train Sharpe; target < 40% degradation |
| Deflated Sharpe Ratio (DSR) | Must clear dsr_min threshold after deflating by cumulative genome evaluations |

---

## 9. Concrete Sweep Changes

1. **Remove short branch** in `volatility_squeeze.py` lines 108–118: replace `elif slope < 0: signal = Signal(side="short", ...)` with `return None`. Only emit `Signal(side="long")` when slope > 0.

2. **Make squeeze depth threshold evolvable:** replace hardcoded `past_bb_bw_pct < 20` with `past_bb_bw_pct < p["squeeze_depth_pct_max"]`. Add to STRATEGY_REGISTRY param_space: `"squeeze_depth_pct_max": ("float", 10.0, 30.0)`.

3. **Add RSI overbought filter:** before emitting long signal, check `if current.get("rsi_14", 50) > p["rsi_overbought_filter"]: return None`. Add to param_space: `"rsi_overbought_filter": ("float", 65.0, 85.0)`.

4. **Add BMSB macro gate:** `if p.get("require_bmsb", True) and not current.get("bmsb_bullish", True): return None`. Keep as fixed True initially; make evolvable if needed.

5. **Add dist_from_high_20 confirmation:** `if current.get("dist_from_high_20", -0.1) < -0.05: return None` (price must be within 5% of 20-bar high for an upward breakout signal).

6. **Add obv_divergence filter:** `if p.get("require_obv_pos", False) and current.get("obv_divergence", 0) < 0: return None`. Add bool to param_space: `"require_obv_pos": ("bool",)`.

7. **Extend time_stop_hours range** in STRATEGY_REGISTRY from current `("int", 24, 36)` (implied by fixed value) to `("int", 24, 120)`. This is the single highest-value change for long-only breakout capture.

8. **Extend take_profit_atr_mult high** from 8.0 to 9.0 in STRATEGY_REGISTRY to allow the evolution to discover wider profit targets for long breakouts.

9. **Add volume_zscore as secondary gate:** `if p.get("require_vol_zscore", False) and current.get("volume_zscore", 0) < p.get("vol_zscore_min", 1.5): return None`. Adds `"vol_zscore_min": ("float", 0.5, 2.5)` to param_space.

10. **Post-release ATR stop calculation:** compute stop using the ATR at the first non-squeezed bar rather than the compressed ATR during the squeeze. This prevents stop-outs from ATR mean-reversion immediately post-release.

---

## 10. Methodological Risks

1. **Sparse trade count (49 trades):** the SE of Sharpe ≈ 1/√49 ≈ 0.14. Removing shorts may leave ~25 long-only trades over the full history—insufficient for reliable walk-forward folds. The evolution must penalize genomes with fewer than 30 trades (already in fitness function as low_trade_penalty).

2. **Short-removal frequency halving:** half the setups become cash. The remaining long-only signal frequency may be too low for the backtester's minimum trade count thresholds to be satisfied in walk-forward fold test windows.

3. **Regime label leakage:** the regime gate uses `_regime` column labels. If these were produced with any forward-looking computation, every gated entry inherits that leakage. Verify regime computation is strictly causal.

4. **High volume threshold regime-dependence:** volume_surge_threshold=1.9 was evolved on specific historical data. In low-liquidity regimes or on altcoins, the threshold may be chronically unsatisfied, generating near-zero trades OOS.

5. **Compressed ATR stop fragility:** ATR is suppressed during the squeeze (definitional). Setting the stop at compressed ATR × 3.79 produces a stop that is effectively tighter than normal-regime ATR. Post-release ATR mean-reversion can cause stop-outs before the breakout actually fails.

6. **bb_bw_percentile lookback bias:** the rolling percentile requires 100 bars (min_periods=20). With a backtester warm-up of 50 bars, the first 50 bars of strategy operation see < 100-bar percentile estimates, biased low, potentially generating phantom squeezes in early data.

7. **Concentration in 2020-2021 bull regime:** if the 49 original trades cluster in the 2020-2021 and 2023-2024 bull runs, the strategy may be a bull-regime detector rather than a general breakout strategy. Per-year trade attribution analysis is essential.

8. **Linear regression slope as direction signal:** slope is computed on the bars leading into the squeeze release, not the release bar itself. A sudden reversal on the release bar (e.g., a wick) would still trigger an entry if the preceding slope was upward. Adding close_location > 0.5 (closed in upper half of release bar) as a confirmation would mitigate.

9. **Time stop destroying long-only edge:** the 36h time stop forcibly exits multi-day moves. For the long-only conversion, the single biggest enhancement is extending the time stop and adding a trailing stop. Not doing this correctly will systematically understate the long-only edge.

10. **Cross-asset parameter transferability:** keltner_atr_mult=1.1686 and volume_surge_threshold=1.9 are optimized for the primary asset. These values may be poorly calibrated for cross-asset validation (ETH, SOL, BNB), causing the cross-asset penalty to systematically disadvantage this genome even if the mechanism generalizes.

---

## Classification: CANDIDATE

The volatility squeeze mechanism is grounded in a well-documented microstructure phenomenon (compression → expansion). The long-only conversion is straightforward (drop short arm), and the crypto spot positive drift bias provides a structural tailwind. The primary concerns are sparse trade count and the overly tight 36h time stop. With the sweep changes above—especially extended time stop, trailing stop, and relaxed volume threshold—this strategy has a realistic path to a long-only Sharpe > 1.5 with walk-forward robustness. It is not fragile (the mechanism is interpretable) and not doubtful (the evolved genome shows WF_Sharpe > 3 even with short arm noise); it is a genuine candidate requiring surgical modifications.
