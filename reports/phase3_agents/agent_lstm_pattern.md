
# Strategy Analysis: `lstm_pattern` — Phase-3 Spot Long-Only Conversion

**Classification: DOUBTFUL**

---

## 1. Diagnosis

### Current Logic

`lstm_pattern` is a two-stage classification pipeline:

1. **Clustering stage** (`_precompute_all`, lines 153-261): Rolling OHLCV windows of `window_size=10` bars are extracted and normalized via a pre-trained scaler, then assigned cluster IDs by one of 7 clustering variants (`kmeans_8/20/50`, `hier_20/50`, `bisect_20/50`). Cluster gaps are forward-filled from the last valid label.

2. **LSTM stage** (`_precompute_all`, lines 196-261): For each bar with index >= `seq_length + window_size`, a sequence of `seq_length=20` bars is built. Each timestep encodes the cluster label as a one-hot vector plus 8 technical feature values: `rsi_14`, `adx_14`, `volume_ratio`, `ret_zscore_20`, `bb_pct_b`, `pressure_imbalance`, `gk_vol`, `vol_ratio`. A single batch inference call produces 3-class directional probabilities (down/neutral/up) for all bars.

3. **Signal combination** (lines 329-337): `combined_up = (cluster_prob_up * cw + lstm_prob_up * lw) / (cw + lw)`. Default weights: `cluster_weight=0.4`, `lstm_weight=0.6`.

4. **Filters**: ADX >= 15, volume_ratio >= 1.0 (optional), RSI overbought guard on longs (RSI < 75), RSI oversold guard on shorts (RSI > 25).

**Does it emit shorts?** YES. Lines 359-376 constitute a live short branch: when `combined_down > combined_up > min_combined` and `cluster_prob_down >= min_cluster_prob`, the function returns `Signal(side="short", stop_loss=close + atr*mult, take_profit=close - atr*mult)`. This branch must be completely removed for Phase-3.

**Likely edge**: LSTM pattern recognition of recurring bullish/bearish candlestick sequence structures. In bull markets, bullish cluster transitions (e.g., accumulation → momentum bursts) may have genuine predictive value over 1-4 bar horizons. The edge is conditioned on trend (ADX) and volume confirmation.

**Likely weaknesses**:
- Silent fallback to uniform predictions when model files are absent or shape-mismatched
- Pre-trained models may be partially fit to data overlapping the OOS lockbox
- In long-only mode, ~50% of directional signals (bearish) are discarded, halving frequency
- Cluster label forward-fill creates stale-signal artifacts near window boundaries
- Model weights are static across walk-forward folds — no adaptation

---

## 2. Spot Long-Only Edge Hypothesis

Trained LSTM pattern clusters capture recurrent bullish candlestick microstructures (accumulation wicks, compressed volatility followed by momentum bursts) that persist at multi-hour horizons. When the LSTM's `P(up)` and the cluster's historical win-rate both exceed threshold, under rising volume (`volume_ratio > threshold`) and confirmed trend (`ADX > adx_min`), price has historically continued upward over the next 18-36 hours.

**Conversion to long-or-cash**: Emit `long` only when `combined_up > min_combined`. Treat the short signal (or neutral) as "go/remain in cash." Gate all entries on the BMSB (price above both `bmsb_sma` and `bmsb_ema`) to restrict long exposure to bull-market regimes where the training distribution's bull-cluster patterns are most likely to generalize.

The LSTM's added value over a simple cluster profile is its ability to condition on the *sequence* of recent cluster transitions — not just the current cluster snapshot — capturing momentum through state machine transitions (e.g., "cluster 7 → cluster 12 → cluster 3" reliably precedes a rally).

---

## 3. Market Techniques

1. **Batch LSTM inference** — already implemented; all bar predictions computed in one forward pass at backtest start. Critical for computational tractability. Preserve as-is.
2. **Multi-variant clustering** — evolution selects best algorithm (kmeans/hierarchical/bisecting) and granularity (k=8/20/50) via `cluster_variant` choice param.
3. **Cluster profile historical win-rate** as a Bayesian prior; LSTM posterior as the likelihood update.
4. **Walk-forward re-training** of cluster models on expanding in-sample windows — currently NOT implemented but critical to address model staleness.
5. **BMSB regime gate** — price above 20-period SMA and 21-period EMA as bull-market condition for long entry.
6. **ATR-based dynamic stop and TP sizing** — adapts risk envelope to realized volatility.
7. **RSI overbought filter** — prevents chasing extended moves on entry.
8. **Volume confirmation filter** — enforces above-average participation on entry bars.
9. **ADX trend-strength gate** — filters signals in ranging, choppy markets.
10. **Time-stop** — closes positions that fail to reach TP/SL within expected LSTM horizon.

---

## 4. Useful Features

### Price / Return
- `ret_zscore_20` — already in LSTM feature vector; also standalone entry filter
- `roc_12` — 12-bar rate of change for directional momentum confirmation
- `dist_from_high_20` — proximity to 20-bar high; low distance signals breakout potential

### Volatility
- `atr_14` — stop and TP sizing (already used)
- `gk_vol` — Garman-Klass vol estimator; already in LSTM feature vector
- `vol_ratio` — short-term vs long-term volatility ratio; low ratio = favorable entry
- `bb_bandwidth` — Bollinger Band width; low = compression before breakout
- `bb_bw_percentile` — rolling percentile rank of bandwidth; bottom decile = squeeze

### Volume
- `volume_ratio` — existing filter; keep as continuous, sweep threshold
- `volume_zscore` — spike detector for abnormal bar activity
- `obv_divergence` — OBV vs 20-SMA; accumulation signal
- `buying_pressure` — bar-level bullish microstructure

### Trend / Momentum
- `adx_14` — existing filter; continuous param in sweep
- `rsi_14`, `rsi_7` — overbought filter + entry floor
- `bmsb_bullish` — binary outer regime gate (price above both BMSB bands)
- `bmsb_spread` — spread between BMSB SMA and EMA as trend-conviction proxy

### Microstructure
- `pressure_imbalance` — already in LSTM feature vector; standalone directional filter
- `close_location` — where bar closed in its range; high = bullish bar structure

### Regime / Drawdown Context
- `dist_from_low_20` — distance from 20-bar low; recovery context
- `bb_pct_b` — already in LSTM feature vector; position within Bollinger Bands

---

## 5. Parameters for the Sweep / Genome

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `cluster_variant` | choice | — | — | 7 variants: kmeans_8/20/50, hier_20/50, bisect_20/50 |
| `min_cluster_prob` | float | 0.50 | 0.75 | Floor at 0.50 — below this the cluster favors down; useless for long-only |
| `min_lstm_confidence` | float | 0.35 | 0.65 | Max P over 3 classes; below 0.35 the model is near-random |
| `min_combined_confidence` | float | 0.45 | 0.72 | Primary selectivity gate |
| `cluster_weight` | float | 0.20 | 0.60 | Derive lstm_weight = 1 - cluster_weight; remove lstm_weight from genome |
| `adx_min` | float | 10.0 | 30.0 | Trend-strength threshold |
| `volume_threshold` | float | 0.70 | 2.00 | Minimum volume_ratio on entry bar |
| `rsi_overbought` | float | 65.0 | 85.0 | Long entry ceiling |
| `rsi_entry_min` | float | 30.0 | 55.0 | NEW: long entry floor to avoid dead-cat bounces |
| `require_bmsb` | bool | — | — | NEW: BMSB bull-market gate |
| `stop_loss_atr_mult` | float | 1.5 | 4.5 | ATR multiplier for stop |
| `take_profit_atr_mult` | float | 2.5 | 7.0 | ATR multiplier for TP; must > SL mult |
| `time_stop_hours` | int | 18 | 72 | Max hold time in hours |

---

## 6. Parameters That Should NOT Be Optimized

- **`window_size`** — must match pre-trained cluster model input shape. Changing it requires full retraining of both the clusterer and the LSTM. Fix to the trained value.
- **`seq_length`** — must match LSTM `input_shape[1]`. Mismatch causes silent fallback to uniform 1/3 probabilities. Fix to the trained value.
- **`lstm_weight`** — made redundant by enforcing `lstm_weight = 1 - cluster_weight`. Keeping it as a separate evolvable param allows inadvertent non-normalized weighting and wastes a search dimension.
- **`rsi_oversold`** — only used in the short branch guard, which is removed entirely. Eliminate from PARAMS and param_space.
- **`require_regime` / `allowed_regimes`** — string-based regime matching is high-cardinality with unclear semantics. The BMSB gate replaces this cleanly. Keeping both creates an interaction that is nearly impossible to optimize without overfitting.

---

## 7. Long-Only Risk Management

- **Stop-loss**: Hard ATR-based stop below entry: `stop = entry_price - atr_14 * stop_loss_atr_mult`. Checked intra-bar on bar low (backtester already implements). Not trailed by default.
- **Take-profit**: ATR-based TP above entry: `tp = entry_price + atr_14 * take_profit_atr_mult`. Checked intra-bar on bar high. Enforce TP mult > SL mult via genome constraint (already in `auto_evolve.py`).
- **Trailing stop**: Once the position is in profit by >= 1.0 × `atr_14` from entry, trail the stop to `entry_price + 0.5 × atr_14` (guaranteed scratch at worst). Fixed rule — not evolvable — to avoid overfitting the trailing threshold.
- **Volatility targeting**: Use the backtester's `volatility_scaled` sizing method: `size = equity × (vol_target / realized_annual_vol)`, capped at `max_position_pct = 0.25`. Default `vol_target_annualized = 0.20` (20%). This auto-reduces size in high-vol regimes.
- **Max position size**: 25% of equity per trade hard cap. With vol-targeting and `stop_loss_atr_mult` in 1.5-4.5 range, expected per-trade loss at stop = approximately 1-3% of equity.
- **Cooldown**: Minimum 4 hours between new entries via `RiskManager.min_time_between_trades`. Prevents re-entry chasing after a fast stop-out on the same cluster state.
- **Risk-off gate**: Outer BMSB gate — if `close < bmsb_sma AND close < bmsb_ema`, block all new long entries regardless of LSTM signal strength. Additionally: circuit-breaker halt at 15% peak-to-trough drawdown (already in `RiskManager`); daily loss halt at 5% of daily starting equity.

---

## 8. Relevant Metrics

| Metric | Target / Notes |
|---|---|
| CAGR | Primary return metric on 2-year OOS lockbox |
| Sharpe ratio (annualized) | Target > 1.0 OOS; minimum > 0.5 to report |
| Sortino ratio | Preferred over Sharpe for skewed crypto returns |
| Max drawdown % | Hard constraint: reject if > 25% OOS |
| Calmar ratio | CAGR / max drawdown; target > 0.5 |
| Hit rate (win rate) | > 50% expected if directional edge is real |
| Profit factor | > 1.2 minimum to cover round-trip costs |
| Avg trade net PnL (bps) | Must be positive after ~10-20bps round-trip; verify cost sensitivity |
| Turnover (trades/month) | Expected 4-15 with long-only filter; < 4 = insufficient statistical power |
| Exposure % | Target 30-60% for spot long-only uptrend capture |
| Cost sensitivity | Sharpe at 5bps vs 15bps slippage; must remain > 0.5 at 15bps |
| Bootstrap / MC | 95th pct max drawdown and P(ruin) over 1000 trade-sequence resamples |
| Per-year Sharpe stability | Positive Sharpe in >= 3 of 4 pre-lockbox calendar years |
| Walk-forward OOS degradation | (IS Sharpe - OOS Sharpe) / IS Sharpe < 30% |
| Deflated Sharpe Ratio | DSR >= 0.95 for global HoF admission; already implemented in `auto_evolve.py` |

---

## 9. Concrete Changes to Implement in the Sweep

1. **Remove the short branch** (lines 359-376 in `lstm_pattern.py`): delete the entire `elif combined_down > combined_up and combined_down > min_combined` block. Short signal becomes "do nothing / stay in cash."

2. **Remove `rsi_oversold` from PARAMS and param_space**: was only used in the removed short branch guard.

3. **Remove `require_regime` and `allowed_regimes` from PARAMS and param_space**: replace with single `require_bmsb` bool.

4. **Add `require_bmsb` to PARAMS** (default `True`) and to registry param_space as `("bool",)`. Add gate in strategy function:
   ```python
   if p["require_bmsb"]:
       bmsb_ok = current.get("bmsb_bullish", True)
       if not bmsb_ok:
           return None
   ```

5. **Add `rsi_entry_min` to PARAMS** (default `35.0`) and to registry param_space as `("float", 30.0, 55.0)`. Add long-entry floor check:
   ```python
   if not pd.isna(rsi) and rsi < p["rsi_entry_min"]:
       return None
   ```

6. **Remove `lstm_weight` from PARAMS and param_space**. Derive in strategy function: `lw = 1.0 - p["cluster_weight"]`. Remove from the weighted average.

7. **Tighten `min_cluster_prob` lower bound** in registry param_space from 0.45 to 0.50 (below 0.50 the cluster statistically favors down — useless for long-only).

8. **Add `time_stop_hours` to registry param_space** as `("int", 18, 72)`.

9. **Add safety net in `auto_evolve.py` fitness evaluation**: after `backtester.run()`, assert no trades with `side == "short"` in `trades_df`; if any found, set `fitness = -999.0`.

10. **Validate model availability before genome evaluation**: in `evaluate_genome`, check if the selected `cluster_variant` model files exist before running the backtest; if missing, return `fitness = -999.0` with a clear error message.

---

## 10. Methodological Risks

1. **Pre-trained model lookahead**: if cluster or LSTM models were trained on data that extends into the 2024-2026 OOS lockbox, every backtest is contaminated. Training cutoff must be audited and documented as strictly before 2024-06-07.

2. **Silent fallback to uniform predictions**: when `_load_models` or `_precompute_all` fails (missing files, shape mismatch), `_bar_lstm_preds` is filled with 1/3 everywhere. The backtest then measures only the ADX/volume/RSI filter performance, not the LSTM — but it is reported as `lstm_pattern`. Add an explicit ERROR-level log and `fitness = -999.0` return.

3. **Cluster label forward-fill artifacts**: stale cluster labels in the transition zone between windows can produce directional signals from old pattern states. Mitigate by requiring that the current bar's cluster was assigned from a window ending no more than 2 bars ago.

4. **Static models across walk-forward folds**: the LSTM and cluster weights never update across the 90/15/45-day walk-forward folds. A model trained on 2019-2023 data is applied unchanged to 2024-2026 — distributional shift from bear-to-bull or macro regime changes can cause severe OOS degradation.

5. **Nested model selection overfitting**: evolving `cluster_variant` over 7 pre-trained models is a hyperparameter search on fixed model artifacts. If each model was optimized on the same in-sample dataset, the evolution compounds overfit — the selected variant is likely the one that happened to fit in-sample noise rather than genuine out-of-sample patterns.

6. **Insufficient trade count in long-only mode**: dropping the short branch roughly halves signal frequency. With ADX >= 15, volume_ratio >= 1, and min_combined >= 0.55, the strategy may generate fewer than 30 trades per 90-day walk-forward window — below the minimum for statistically meaningful DSR and Sharpe estimates.

7. **TP multiplier dominates PnL**: because signal frequency is low, gross return per trade is determined almost entirely by the TP ATR multiplier. This creates a high-variance, brittle Sharpe that is highly sensitive to TP overfitting — small changes in TP mult produce large swings in reported performance.

8. **Cluster profile label leakage**: `prob_up` and `prob_down` in cluster profiles are computed by labeling windows with their subsequent return. If the label horizon extends past the in-sample boundary, cluster profiles carry forward-looking information that is not available in live trading.

9. **Global mutable model cache in multiprocessing spawn workers**: the module-level `_models_loaded`, `_cluster_model`, `_bar_cluster`, `_bar_lstm_preds` are reset only when `(variant_id, timeframe)` changes or when `id(df)` changes. If the spawn worker reuses a cached model from a prior genome evaluation with a different `df` object that happens to have the same `id()` (CPython address reuse), predictions from the wrong dataset will silently persist.
