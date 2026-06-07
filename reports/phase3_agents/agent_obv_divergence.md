# OBV Divergence — Phase-3 Spot Long-Only Research Report

## 1. Diagnosis

**Strategy logic.** `obv_divergence_strategy` detects divergence between OBV and price over a rolling `lookback` window (default 20 bars). Two code paths exist:

- **Fast path** (used when `obv_divergence` column is non-NaN and `|obv_div| > threshold`): uses the precomputed feature directly. However, in `features.py` the column `obv_divergence` is computed as `OBV - OBV_SMA(20)` — an OBV momentum oscillator, NOT a price/OBV divergence. This is semantically incorrect for the stated purpose.
- **Slow path** (manual fallback): normalises both price and OBV to [0,1] via min-max scaling, fits linear regression slopes to each, and computes `obv_slope - price_slope`. This is a genuine divergence measure.

In practice the fast path fires first whenever the precomputed feature column is populated (which is always the case when `build_all_features` is called), so the slow path is effectively dead code. The strategy operates on a semantically wrong signal.

**Short signals.** Yes — the strategy emits `side="short"` in the bearish branch. This must be removed for Phase 3.

**RSI logic error.** The RSI confirmation for the short branch reads: `if rsi_confirm and rsi < rsi_oversold: return None`. This blocks shorts when RSI is already oversold, which is precisely the most obvious short entry — a logic inversion.

**Dead parameter.** `adx_max` is declared in `PARAMS` but never referenced in the strategy body.

**Registry status.** `obv_divergence` was **pruned in Round 1** with **0% positive fitness** across 29,000 evaluations. This is the strongest available signal about the strategy's current edge.

**Likely edge.** Accumulation detection (smart-money OBV rising while price is flat/declining) is a real phenomenon with academic support in equities. The weakness here is the implementation: the fast path reads the wrong feature, the short branch is inverted, and the threshold is not normalised for OBV level changes across time.

---

## 2. Spot Long-Only Edge Hypothesis

Go long when OBV is rising faster than price (true slope-comparison divergence), indicating institutional accumulation at depressed or sideways prices, and hold until price catches up or the time/stop expires. Go to cash otherwise (treat the bearish divergence signal as an exit cue only, not as a short).

The hypothesis requires:
1. The correct divergence metric (slope comparison, not OBV-minus-SMA).
2. Price must be weak at entry (RSI < rsi_oversold, price near 20-bar low, bb_pct_b < 0.4).
3. Volume must confirm: volume_ratio > 1.0 at the accumulation bar.
4. Macro regime must be supportive: bmsb_bullish = True (no bear-market traps).

This is a **mean-reversion / dip-buying** hypothesis within a macro uptrend filter, not a trend-following approach.

---

## 3. Market Techniques

- **OBV slope divergence** (linear regression on normalised price vs OBV, fixed as the primary signal replacing the feature-column fast path).
- **RSI oversold filter on longs**: entry only when RSI < rsi_oversold, ensuring price weakness accompanies accumulation.
- **ATR-based stop and take-profit** with configurable multipliers.
- **ADX low filter** (adx < adx_max): divergence is most meaningful in low-trend environments; in a strong trend, OBV and price move together.
- **Volume surge confirmation** (volume_ratio >= volume_ratio_min): confirm accumulation with above-average traded volume.
- **BMSB macro gate**: only take longs above both BMSB bands (20-period SMA and 21-period EMA as bull-market proxy).
- **Bollinger Band width percentile filter**: prefer entries when bb_bw_percentile < 30 (compression before breakout).
- **Time stop** (time_stop_hours): limit exposure when divergence fails to resolve into a price move.
- **Trailing stop** (breakeven at +1 ATR, trail at +2 ATR): lock in gains during genuine post-accumulation breakouts.
- **Walk-forward expanding-window calibration** of threshold on pre-lockbox history only.

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Volume | `obv` | Primary divergence input (slope computation) |
| Volume | `obv_divergence` (OBV-SMA) | Secondary momentum filter only (not primary signal) |
| Volume | `volume_ratio` | Confirm accumulation with above-average volume |
| Volume | `volume_zscore` | Spike detector for breakout confirmation |
| Momentum | `rsi_14` | Oversold filter at entry |
| Momentum | `rsi_7` | Faster timing for tighter RSI confirmation |
| Trend | `adx_14` | Low-ADX gate (divergence valid in non-trending regimes) |
| Volatility | `atr_14` | Stop / take-profit sizing |
| Volatility | `bb_bw_percentile` | Pre-breakout compression detector |
| Volatility | `gk_vol` | Vol-targeting position sizing |
| Price | `dist_from_low_20` | Confirm price near recent lows at accumulation entry |
| Price | `bb_pct_b` | Price near lower Bollinger band (< 0.4) |
| Regime | `bmsb_bullish` | Macro bull-market gate |
| Microstructure | `buying_pressure` | Bar-level confirmation of accumulation footprint |
| Microstructure | `pressure_imbalance` | Demand/supply imbalance at signal bar |

---

## 5. Params That Should Enter the Sweep / Genome

| Param | Type | Low | High | Notes |
|---|---|---|---|---|
| `lookback` | int | 10 | 40 | Slope-comparison window; critical sensitivity param |
| `obv_divergence_threshold` | float | 0.10 | 0.80 | Primary filter; primary overfitting axis — watch carefully |
| `rsi_oversold` | float | 25.0 | 50.0 | Confirms price weakness on longs |
| `adx_max` | float | 20.0 | 45.0 | Max ADX to accept divergence signal |
| `stop_loss_atr_mult` | float | 1.0 | 4.0 | ATR multiplier for stop |
| `take_profit_atr_mult` | float | 1.5 | 6.0 | ATR multiplier for target |
| `time_stop_hours` | int | 12 | 96 | Max hold duration if divergence does not resolve |
| `volume_ratio_min` | float | 0.8 | 2.0 | Minimum volume confirmation |
| `require_bmsb_bull` | bool | 0 | 1 | Macro regime gate on/off |

---

## 6. Params That Should NOT Be Optimized

- **`min_price_move_pct`**: currently dead code — not referenced anywhere in the function body. Remove from genome to avoid spurious degrees of freedom.
- **`rsi_overbought`**: with the short branch removed, this param has zero effect. Remove entirely.
- **`require_regime` / `allowed_regimes`**: the regime classifier is not well-calibrated for tuning; fixing require_regime=False avoids fitting to regime label noise.
- **`rsi_confirm` (bool)**: fix to True for the long branch — allowing the genome to toggle off a sanity check allows overfitting to lucky periods where the filter was unhelpfully tight.

---

## 7. Long-Only Risk Management

| Rule | Setting |
|---|---|
| **Stop loss** | `entry_price - stop_loss_atr_mult * atr_14`; minimum 1.0x ATR enforced at signal generation; uses bar low check each tick |
| **Take profit** | `entry_price + take_profit_atr_mult * atr_14`; enforce minimum R:R ratio 1.5 at signal time; reject signals where ATR-implied R:R < 1.5 |
| **Trailing stop** | At +1x ATR unrealised: move stop to breakeven. At +2x ATR: trail at 1x ATR below running high close |
| **Vol targeting** | Use `sizing_method=volatility_scaled`; target annualised vol 30-40% using `gk_vol`; cap at `max_position_pct` |
| **Max position size** | 20% of equity per trade; reduce to 10% when gk_vol > 90th percentile of trailing 90-day distribution |
| **Cooldown** | Minimum 6 bars (6 hours on hourly data) between new entries via RiskManager min_interval; 48-hour pause after 3 consecutive stop-outs |
| **Risk-off regime gate** | Block new longs when `bmsb_bullish=False` AND `adx_14 < 15`; also block when drawdown from equity peak > 15% (circuit breaker) |

---

## 8. Relevant Metrics

- **CAGR**: target > 20% annualised for long-only crypto (buy-and-hold BTC is ~40-80% depending on period; strategy must justify lower exposure with better risk-adjusted return)
- **Sharpe ratio** (risk-free=0): target > 0.8 OOS; any positive result from the repaired strategy vs 0% fitness baseline would be meaningful
- **Sortino ratio**: target > 1.0 (more appropriate for long-only where upside is not penalised)
- **Max drawdown %**: hard cap 30%; measure full-sample, not per-fold maximum
- **Calmar ratio** (CAGR/maxDD): target > 0.5
- **Hit rate**: expect < 50%; monitor that R:R compensates (profit factor > 1.3)
- **Turnover**: target 4-12 round-trips/month on hourly data; below 2/month gives insufficient OOS statistics
- **Gross/net exposure**: target 20-50% time in market; very low exposure (< 15%) raises question of why not just hold cash
- **Average trade duration vs time_stop_hours**: if > 70% of exits are time stops, the divergence is not resolving — tighten or abandon
- **Cost sensitivity**: sweep commission_bps 2→20; target < 20% Sharpe degradation from 5bp to 15bp
- **Bootstrap Sharpe CI** (1000 resamples): lower 5th percentile must be > 0.3 to be credible
- **Per-year Sharpe stability**: flag if any pre-lockbox year < -0.5 (regime fragility)
- **OOS degradation**: target < 40% train-to-test Sharpe drop; Round 1 result implies this will likely be very high

---

## 9. Concrete Changes to Implement in the Sweep

1. **Remove short branch**: replace the `elif obv_div < -threshold` block with `return None`. Delete `rsi_overbought` from PARAMS and genome.
2. **Fix RSI confirmation for longs**: change gate from `rsi > rsi_overbought` to `rsi > rsi_oversold` (block longs when RSI is NOT depressed — i.e., require oversold to confirm accumulation).
3. **Remove the precomputed-feature fast path**: always use the manual slope-comparison (or build a correct `obv_price_divergence` column in `features.py` using polyfit slopes). The current `obv_divergence` column is semantically wrong for this strategy.
4. **Activate dead ADX filter**: add `if current.get('adx_14', 0) > p['adx_max']: return None` to the bullish branch.
5. **Add volume_ratio_min check**: `if current.get('volume_ratio', 1.0) < p['volume_ratio_min']: return None`.
6. **Add bmsb_bullish gate**: if `require_bmsb_bull` is True, return None when `bmsb_bullish` is False.
7. **Add dist_from_low_20 context filter**: only enter long when `dist_from_low_20 > -0.15` (price within 15% of 20-bar low).
8. **Re-add to STRATEGY_REGISTRY** in `auto_evolve.py` with the cleaned param_space covering: `lookback` (int 10-40), `obv_divergence_threshold` (float 0.1-0.8), `rsi_oversold` (float 25-50), `adx_max` (float 20-45), `stop_loss_atr_mult` (float 1.0-4.0), `take_profit_atr_mult` (float 1.5-6.0), `time_stop_hours` (int 12-96), `volume_ratio_min` (float 0.8-2.0), `require_bmsb_bull` (bool).
9. **Fix edge case guards**: validate that `len(window) >= lookback + 1` before polyfit; handle `p_range == 0` or `o_range == 0` (already handled in slow path, verify fast path removal does not reintroduce).
10. **Enforce minimum R:R at signal time**: add `if atr * take_profit_mult / (atr * stop_loss_mult) < 1.5: return None`.

---

## 10. Methodological Risks

1. **OBV non-stationarity**: OBV is a cumulative absolute-volume series. Its level is meaningless across time and market cap changes. Min-max normalisation per window reduces but does not eliminate regime-dependent scaling. The threshold 0.3 on slopes calibrated on 2019-2023 data will not mechanically transfer to 2024-2026 without re-normalisation.

2. **Feature mismatch (fast path vs slow path)**: because the precomputed `obv_divergence` feature is always available, the slow-path (correct) logic is never reached. This means the strategy as-coded has always been measuring OBV momentum (OBV-SMA) not true divergence — explaining the 0% Round 1 fitness. The fix (removing the fast path) constitutes a meaningfully different strategy, so Round 1 history is not informative about the corrected version.

3. **Round 1 fitness failure as prior**: 0% positive fitness across 29,000 evaluations is a strong empirical prior against this strategy class. Even after code fixes, the bar for re-inclusion must be high (Sharpe > 1.0 in walk-forward, not just any positive result).

4. **Threshold overfitting**: `obv_divergence_threshold` swept over [0.1, 0.8] is the primary overfitting axis. The optimal threshold varies by market regime and volume structure. Monitor IS/OOS threshold consistency in walk-forward folds.

5. **Signal sparsity and statistical unreliability**: with RSI oversold filter, volume_ratio filter, and ADX max filter active simultaneously, valid signals may occur fewer than 2-3 times per month. With 15-day walk-forward test folds, this yields 1-2 trades per fold — too few for reliable Sharpe estimation. The 2-year OOS lockbox might produce only 30-50 trades total, making OOS conclusions statistically weak.

6. **RSI-OBV joint condition rarity**: requiring both OBV bullish divergence AND price oversold (RSI < 35) simultaneously is rare, especially in macro bull markets where price rarely revisits deeply oversold territory. This filter combination may produce effectively zero signals in strong uptrends (2020-2021, 2024).

7. **Time-stop dominance risk**: if most positions exit via time stop (rather than take-profit or stop-loss), the strategy generates a negative-expectancy stream of round-trip costs without capturing the divergence resolution. Monitor exit-reason distribution in walk-forward.

8. **OOS lockbox integrity**: the 2-year lockbox (2024-06-07 to 2026-06-07) must not be touched for any threshold or filter calibration. Given the strategy's fragility, even a single look at OOS performance before finalising the param sweep would constitute leakage.

---

## Classification: DOUBTFUL

The strategy's core concept (OBV/price divergence detecting smart-money accumulation) has legitimate academic and practitioner support. However, three compounding problems make it a doubtful candidate for Phase-3:

1. The implementation has been using the wrong signal (OBV momentum oscillator vs. true slope divergence) for all prior evaluations, explaining the 0% fitness.
2. The code corrections required are substantial enough that the resulting strategy is materially different — prior negative evidence does not help, but also does not validate.
3. Even with fixes, the joint filter conditions (RSI oversold + OBV divergence + volume + ADX + BMSB) will produce very sparse signals whose OOS statistics will be unreliable.

Promote to **candidate** only if the corrected slow-path version shows Sharpe > 1.0 with > 50 trades in pre-lockbox walk-forward and < 40% OOS degradation. Otherwise keep as doubtful / archive.
