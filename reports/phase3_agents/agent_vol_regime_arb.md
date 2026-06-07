## vol_regime_arb — Phase 3 Spot Long-Only Research Report

**Classification: CANDIDATE**

---

### 1. Diagnosis

`vol_regime_arb` (`/strategies/vol_regime_arb.py`) is a dual-mode Garman-Klass (GK) volatility z-score strategy. It precomputes multi-scale GK volatility columns once per backtest run using vectorized rolling operations (a good performance design), then per bar reads O(1) precomputed values.

**Mode 1 — VOL_EXPANSION (low vol → breakout):** When the fast GK vol z-score (fast window over a baseline window) falls below `expansion_zscore_threshold` (-1.2 default) for at least `expansion_min_bars_compressed` consecutive bars (5 default) AND ADX > 18, the strategy fires. It uses the sign of a linear-regression slope over the last `expansion_momentum_period` bars to determine direction. Positive slope → LONG. Negative slope → **SHORT**. This violates Phase-3 constraints.

**Mode 2 — VOL_CONTRACTION (extreme vol → fade):** When the fast GK vol z-score exceeds `contraction_zscore_threshold` (2.0 default), the strategy fades the move using RSI. RSI < 30 → LONG (fade the crash). RSI > 70 → **SHORT** (fade the rip). The short branch here also violates Phase-3 constraints.

**Current short emission:** Yes, in both modes. The strategy emits `side="short"` Signals with stop_loss above current price and take_profit below.

**Edge:** The core edge is statistically grounded — volatility mean-reversion (vol compression predicts expansion, extreme vol tends to contract) is well-documented. In crypto, the asymmetry of the asset class (positive long-run drift) means the long branches of both modes are more likely to be profitable than the short branches. Dropping shorts is expected to improve — not harm — the Sharpe in a spot long-only context.

**Weakness:** The ADX gate (expansion mode) may conflict with the vol-compression gate — truly compressed vol periods often have low ADX by definition, potentially filtering many legitimate compression setups. The module-level GK cache using `id(df)` as part of the key is fragile in sequential (non-spawn) contexts where Python can reuse object addresses.

**Registry param_space (auto_evolve.py):** 14 params currently registered. After Phase-3 conversion, `contraction_rsi_overbought` should be removed (dead code) and `expansion_momentum_period` and `time_stop_hours` should be fixed outside the genome.

---

### 2. Spot Long-Only Edge Hypothesis

**Long-or-cash edge:** After a statistically significant GK vol compression period (z-score below threshold for N consecutive bars), the market has coiled. In crypto, where the multi-year drift is strongly positive, breakout resolutions skew upward more frequently than downward, particularly when confirmed by a positive momentum slope. The strategy should enter long when compression is confirmed AND slope > 0; when slope < 0, stay in cash (no trade).

In contraction mode (extreme high vol), RSI-oversold readings during extreme vol spikes represent crash exhaustion / capitulation. Buying these setups in spot long-only is a valid mean-reversion bounce edge — distinct from shorting the overbought case.

The combined hypothesis: **deploy capital into high-conviction, asymmetric long setups characterized by either (a) vol-compression breakouts with upward momentum confirmation, or (b) extreme-vol capitulation bounces with RSI exhaustion and volume support. Hold cash otherwise.**

---

### 3. Market Techniques

1. **Volatility regime filtering (GK z-score)** as primary entry gate — the strategy's core differentiator versus simpler BB/Keltner-based squeezes.
2. **Momentum slope confirmation (linear regression)** for breakout direction — long-only branch retained; short branch dropped.
3. **RSI mean-reversion confirmation** for contraction-mode longs (capitulation bounce entry).
4. **ATR-based dynamic stop loss and take profit** — adapts risk to current volatility regime.
5. **ADX trend-strength filter** to avoid false breakouts in low-energy compressions.
6. **Volume ratio filter** (volume_ratio from features.py) — confirms breakout energy.
7. **Walk-forward / expanding-window cross-validation** on pre-lockbox data (2019–2024) for parameter selection; sealed OOS (2024-06-07 → 2026-06-07) for final evaluation.
8. **BMSB (Bull Market Support Band) macro regime gate** — suppress all entries when price is below bmsb_sma and bmsb_ema (bear cycle suppression).
9. **BB width percentile corroboration** — require bb_bw_percentile < 40 for expansion-mode longs (dual confirmation of compression).
10. **OBV divergence confirmation** for contraction-mode bounces — require positive OBV divergence to filter dead-cat bounces.

---

### 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Volatility | `_gk_fast`, `_gk_zscore`, `_gk_ratio` | Primary entry gates (strategy-internal) |
| Volatility | `gk_vol` (features.py, 20-bar) | Cross-check / secondary confirmation |
| Volatility | `vol_ratio` | Corroborating short/long vol ratio |
| Volatility | `bb_bandwidth`, `bb_bw_percentile` | Secondary compression confirmation |
| Trend | `adx_14` | Expansion-mode gate: avoid low-energy compressions |
| Trend | `bmsb_bullish` | Macro bear-market suppression gate |
| Momentum | `rsi_14` | Contraction-mode oversold gate (long branch only) |
| Momentum | `rsi_7` | Faster RSI for divergence check |
| Momentum | `roc_12`, `price_acceleration` | Multi-timeframe slope confirmation |
| Momentum | `ret_zscore_20` | Statistical return extreme check |
| Volume | `volume_ratio`, `volume_zscore` | Breakout energy confirmation |
| Volume | `obv_divergence` | Buying-pressure confirmation for bounces |
| Price/Return | `dist_from_high_20` | Avoid entries at already-extended levels |
| Microstructure | `pressure_imbalance`, `buying_pressure` | Intra-bar buying confirmation |

---

### 5. Params for the Sweep / Genome

| Param | Type | Low | High | Note |
|---|---|---|---|---|
| `gk_fast_period` | int | 5 | 20 | Fast GK lookback bars |
| `gk_slow_period` | int | 30 | 80 | Slow GK lookback bars |
| `gk_baseline_period` | int | 60 | 150 | Z-score rolling window |
| `expansion_zscore_threshold` | float | -2.5 | -0.8 | Compression signal threshold |
| `expansion_min_bars_compressed` | int | 3 | 10 | Bars of sustained compression required |
| `expansion_adx_min` | float | 10.0 | 28.0 | Trend confirmation gate |
| `contraction_zscore_threshold` | float | 1.5 | 3.5 | Extreme-vol threshold for bounce |
| `contraction_rsi_oversold` | float | 20.0 | 40.0 | RSI oversold threshold for capitulation long |
| `enable_contraction_mode` | bool | 0 | 1 | Toggle capitulation-bounce mode |
| `volume_threshold` | float | 0.8 | 2.0 | Volume ratio floor |
| `stop_loss_atr_mult` | float | 1.5 | 4.5 | ATR-based stop |
| `take_profit_atr_mult` | float | 2.5 | 7.0 | ATR-based TP (enforced > SL mult) |

---

### 6. Params NOT to Optimize (Overfitting Traps)

- **`contraction_rsi_overbought`** — dead code in Phase-3 (short branch dropped); remove from genome entirely.
- **`expansion_momentum_period`** — sweeping this independently from `gk_fast_period` creates redundant correlated degrees of freedom; fix at 10 (or tie to gk_fast_period).
- **`vol_ratio_min` / `vol_ratio_max`** — two-parameter box filter on a continuous ratio adds dimensions without signal; leave at defaults (0.0 / 3.0, effectively off).
- **`require_volume` (bool)** — with `enable_contraction_mode` already a bool in the genome, adding a third binary multiplies discrete combinations; hard-code `require_volume=True`.
- **`time_stop_hours`** — interacts badly with TP/SL in short walk-forward folds; fix at 36–48h.
- **`allowed_regimes` (list)** — categorical multi-select cannot be meaningfully interpolated by genetic operators; use BMSB gate instead.

---

### 7. Long-Only Risk Management

**Stop loss:** ATR-based — `close - atr_14 * stop_loss_atr_mult`. Backtester enforces on bar LOW (not close). Range: 1.5–4.5x ATR in genome.

**Take profit:** ATR-based — `close + atr_14 * take_profit_atr_mult`. Enforced on bar HIGH. Genome constraint: TP mult > SL mult + 0.5 (already enforced in auto_evolve.py mutation/crossover).

**Trailing stop (optional):** Once unrealized PnL exceeds 1.5x ATR, trail stop to entry + 0.5x ATR. This locks partial profit while allowing continuation in genuine breakouts. Implemented as per-bar stop update (not currently in backtester — would require strategy-side stop update logic or a new backtester hook).

**Volatility targeting:** Use `sizing_method='volatility_scaled'` with `vol_target_annualized=0.15–0.20`. During compressed vol (expansion-mode entries), ATR is small → formula sizes up automatically, appropriate for breakout trades. During extreme-vol periods (contraction mode), ATR is large → formula sizes down, appropriate for risk-off bounce trades.

**Max position size:** Hard cap at 80% of equity (`max_position_pct=0.80`). No leverage in spot.

**Cooldown:** 4-bar minimum between trades (RiskManager `min_time_between_trades`). After stop-loss exit, impose additional 12-bar cooldown before re-entering the same mode, preventing whipsaw re-entries.

**Risk-off regime gate:**
1. BMSB gate: no long entries if `bmsb_bullish = False` (price below both BMSB lines → bear regime).
2. Regime gate: require `regime in ['lateral', 'breakout']` for expansion mode.
3. RiskManager circuit breaker: halt new entries if rolling drawdown exceeds `max_drawdown_pct`.
4. Volume filter: `volume_ratio > volume_threshold` (already in strategy, keep as hard requirement).

---

### 8. Relevant Metrics

| Metric | Target | Notes |
|---|---|---|
| CAGR (net) | > 30% | 2-year OOS lockbox (sealed) |
| Sharpe (annualized) | > 1.2 | Hourly equity curve, rf = 0 |
| Sortino ratio | > 1.5 | Long-only: downside dev is key |
| Max drawdown | < 25% | Full-sample WalkForward check enforced |
| Calmar ratio | > 1.0 | CAGR / max-DD |
| Hit rate | 40–55% | Low win rate OK if avg-win/avg-loss > 2.0 |
| Profit factor | > 1.5 | Gross profit / gross loss |
| Avg trade duration | 12–36h | Clustering near time_stop suggests marginal R |
| Turnover | 8–20 trades/month | Statistical significance floor |
| Exposure | 15–40% of time | Selective dual-gate; cash otherwise |
| Cost sensitivity | < 30% Sharpe degradation at 2x costs | Validates non-cost-driven edge |
| Bootstrap Sharpe (5th pct) | > 0.5 | MC resample 1000x of trade P&L |
| Per-year stability | ≥ 3 of 4 half-years positive Sharpe | 2024H1, 2024H2, 2025H1, 2025H2 |
| OOS degradation | < 50% | WF test Sharpe / train Sharpe |

---

### 9. Concrete Changes for the Sweep

1. **Drop short branches:** In VOL_EXPANSION mode, replace `elif slope < 0: return Signal(side='short', ...)` with `return None`. In VOL_CONTRACTION mode, replace `elif rsi > contraction_rsi_overbought: return Signal(side='short', ...)` with `return None`.

2. **Remove `contraction_rsi_overbought`** from `PARAMS` dict and from `STRATEGY_REGISTRY['vol_regime_arb']['param_space']`.

3. **Add BMSB gate:** After the existing regime check, insert: `if not current.get('bmsb_bullish', True): return None`.

4. **Add BB compression corroboration:** After the z-score compression check, insert: `bb_pct = current.get('bb_bw_percentile', 50); if not pd.isna(bb_pct) and bb_pct > 50: return None` (or make 50 an evolvable genome param `bb_bw_pct_max` in range 20–60).

5. **Strengthen momentum confirmation:** Replace bare `slope > 0` check with: `slope > 0 AND current.get('ret_zscore_20', 0) > -0.5 AND current.get('roc_12', 0) > 0`. This requires multi-timeframe momentum alignment.

6. **Add OBV confirmation for contraction-mode longs:** Before issuing the capitulation-bounce long, require `current.get('obv_divergence', 0) > 0`.

7. **Fix GK cache integrity:** Change cache validation to compare the full `(id(df), fast, slow, baseline)` tuple, and add a check that `'_gk_fast' in df.columns` only accepts it if columns were written by the same params. Best fix: store params in a df attribute (`df.attrs['_gk_params'] = (fast, slow, baseline)`) and check that on every call.

8. **Remove `expansion_momentum_period`, `time_stop_hours`, `vol_ratio_min`, `vol_ratio_max`, `require_volume`, `allowed_regimes` from the genome** (keep in PARAMS as fixed constants).

9. **Raise `contraction_zscore_threshold` genome ceiling** from 3.0 to 3.5 to allow the GA to explore genuine crash-volatility thresholds.

10. **Add `bb_bw_pct_max`** as a new genome param (float, 20–60) replacing the hard-coded 50 threshold, allowing the GA to discover the optimal BB compression level that complements the GK z-score.

---

### 10. Methodological Risks

1. **Trade count collapse post-conversion:** Removing short branches halves trade count. Walk-forward folds with < 5 trades receive -5.0 fitness, masking genuine long-only performance. Mitigation: increase `test_days` from 15 to 30 for this strategy, or lower the `min_trades` threshold in `BacktestConfig` for Phase-3 long-only strategies.

2. **GK baseline warmup vs. fold length:** The z-score requires `gk_baseline_period` clean GK fast values before it is reliable. At baseline=100 + fast=10 + min_bars_compressed=5 + backtester warmup=50, the first ~165 bars of each fold are dead. On 15-day (360-bar) folds, ~45% of each fold is warmup — severe efficiency loss. Mitigation: use 30-day folds minimum, or pre-seed fold DataFrames with 150 bars of prior context.

3. **Cache contamination in sequential contexts:** The `id(df)` cache key is unreliable when WalkForwardValidator creates `.copy().reset_index()` slices; Python may assign the same id to a new DataFrame object after the old one is garbage collected. This causes stale GK columns from a prior fold to silently persist. This is a correctness bug, not just a performance issue.

4. **ADX / compression anti-correlation:** Genuinely compressed vol periods often have ADX < 18 (market is directionless by definition). The `expansion_adx_min` gate may therefore filter the majority of true compression setups, leaving only atypical compressions (e.g., tightening in a trending market) which are structurally different and less predictive of breakouts. Monitor expansion-mode trade frequency across ADX-threshold values.

5. **Contraction-mode longs in bear markets:** RSI-oversold + extreme vol = potential bear-market continuation, not bounce. Without the BMSB gate, this mode will fire repeatedly during crypto bear cycles, producing serial correlated losses that bootstrap Monte Carlo (i.i.d. resample) will severely underestimate. The BMSB gate recommended in Section 9 is the primary mitigation.

6. **Non-stationary slope magnitude:** Linear regression slope on raw close prices is proportional to price level. At BTC = \$100,000 vs. \$20,000, the same slope threshold of 0 (sign-only check) is robust, but if any future variant filters on slope magnitude or normalizes differently, non-stationarity will create regime-dependent selection bias.

7. **Hourly OHLCV artifacts:** GK vol uses log(high/low)^2. During thin hourly candles where high ≈ low ≈ open ≈ close, log_hl ≈ 0, which creates GK vol ≈ 0, pushing z-score deeply negative and spuriously triggering expansion-mode longs. A minimum bar-range filter (e.g., `(high - low) / close > 0.0005`) should guard against this.

8. **Walk-forward fold Sharpe standard error:** With 5–10 trades per 15-day fold, the standard error of Sharpe ≈ 0.35–0.45 per fold. The aggregate across N folds narrows this by 1/sqrt(N), but requires many folds to achieve < 0.2 SE. The strategy must accumulate a minimum of 30 trades across all WF folds before the aggregate Sharpe is meaningful — enforced by the `MIN_TRADES_FOR_POSITIVE_FITNESS = 30` constant in `auto_evolve.py`.
