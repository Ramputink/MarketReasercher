# vwap_reversion — Phase-3 Spot Long-Only Research Report

## 1. Diagnosis

**Strategy file:** `/strategies/vwap_reversion.py`
**Registry status:** PRUNED in Round 1 (10h / 29K evaluations, 0% positive fitness). No active param_space in `STRATEGY_REGISTRY`.

### Current Logic

The strategy fires when `close` deviates more than `vwap_deviation_pct` (default 1.5%) from the 20-bar rolling VWAP (`vwap_20` from `volume_features()`), confirmed by:
- RSI-14 oversold (< 30) for longs, overbought (> 70) for shorts
- ADX-14 below `adx_max` (30) — range/lateral regime filter
- Optional: `volume_ratio > 1.5` (panic-selling spike)
- Optional: `buying_pressure > 0.4` (lower-wick rejection — long only) or `selling_pressure > 0.4` (short only)

**It emits both long AND short signals.** The short branch (lines 78-95) fires when price is >1.5% above VWAP with RSI > 70. The take-profit for both sides is set exactly at `vwap_20`, and the stop is `close ± 2x atr_14`.

### Does it emit shorts? YES.

The short branch at lines 78-95 is fully implemented and symmetric with the long branch.

### Likely Edge

Mean reversion to VWAP is a real microstructure phenomenon, driven by passive orders, TWAP/VWAP algorithms, and short-term liquidity imbalances. In ranging/lateral markets, a 1.5-3% VWAP deviation that coincides with capitulation-level RSI and volume spikes can reliably snap back to VWAP within 24-48 hours.

### Likely Weakness

1. **Short branch poisons the spot long-only context.** In Phase 2 (perp-swap context), the short branch captures funding carry + price reversion. In spot long-only, the short branch is simply forbidden.
2. **1.5% VWAP deviation is too tight for crypto.** Crypto regularly sees 2-5% intraday swings against VWAP that do not revert. This creates many false signals — catching falling knives.
3. **RSI-30 + ADX-30 regime coincides with capitulation events**, which in crypto are more likely to continue falling than to revert immediately.
4. **Round-1 0% fitness is a strong negative prior.** Even with long-only conversion, the strategy must be viewed as a hypothesis to test, not a confirmed edge.

---

## 2. Spot Long-Only Edge Hypothesis

In crypto spot markets, prices that trade more than ~2-4% below their 20-bar rolling VWAP during a regime of compressed volatility (low bb_bw_percentile) and macro constructiveness (above BMSB) tend to revert toward VWAP within 12-48 hours. This reversion reflects VWAP-anchored passive demand absorption, short-term panic exhaustion (RSI oversold, high-volume wick rejection), and OBV divergence (demand accumulating while price falls).

**Long-or-cash only.** The strategy enters a spot long when:
- Price is 2-4% below vwap_20 (calibrated via sweep)
- RSI-14 < 30 (or RSI-7 < 25 for faster signal)
- ADX-14 < 25 (non-trending regime)
- bb_bw_percentile < 40 (volatility compressed, reversion more reliable)
- buying_pressure > 0.4 (wick rejection at lows)
- bmsb_bullish = True (macro regime is constructive for longs)

It exits to cash when price reaches vwap_20 (primary TP), a hard ATR-based ceiling TP, or the time stop triggers.

The short overbought condition becomes a cash/no-signal condition (not a short signal).

---

## 3. Market Techniques

1. **VWAP-anchored mean reversion** in lateral/compressed-vol regimes — the core anchor
2. **Dual momentum-volume confirmation** (RSI oversold + volume spike) to qualify capitulation events
3. **Microstructure wick-rejection filter** (buying_pressure) to ensure buyers are absorbing at lows
4. **Bollinger Band compression gate** (bb_bw_percentile < 40) for regime selection favorable to reversion
5. **Bull Market Support Band (BMSB) macro gate** — longs only when price above 20-week SMA and 21-week EMA equivalent
6. **OBV divergence filter** — rising OBV while price falls signals demand absorption beneath surface
7. **Volatility-scaled position sizing** (ATR normalized) — reduce size in high-volatility environments where reversions fail
8. **Trailing stop** once gain reaches 0.5x ATR to protect profits from reversion overshoots
9. **Mandatory cooldown** after stop-loss events to avoid cascading re-entries during directional trends
10. **ATR-based secondary TP ceiling** to prevent holding through a full overshoot when VWAP is far above entry

---

## 4. Useful Features

| Category | Feature | Role |
|----------|---------|------|
| Mean-reversion | `vwap_20` | Primary anchor (already used) |
| Mean-reversion | `dist_from_low_20` | Confirms depth of retracement into support zone |
| Mean-reversion | `bb_pct_b` | Entry when < 0.2 confirms price at lower BB band |
| Volatility | `bb_bw_percentile` | Regime gate: low percentile = compression = reversion favorable |
| Volatility | `atr_14` | Stop sizing and position sizing |
| Volatility | `gk_vol` | Secondary vol regime filter |
| Momentum | `rsi_14` | Oversold primary filter (already used) |
| Momentum | `rsi_7` | Faster RSI for divergence cross-check |
| Momentum | `ret_zscore_20` | Confirms bar is a statistically unusual down move (gate: < -1.0) |
| Momentum | `price_acceleration` | Negative acceleration confirms exhaustion |
| Volume | `volume_ratio` | Panic-selling spike filter (already used) |
| Volume | `volume_zscore` | >2.0 sigma spike on down bars = capitulation signal |
| Volume | `obv_divergence` | Rising OBV while price falls = demand absorption |
| Microstructure | `buying_pressure` | Lower-wick rejection (already used) |
| Microstructure | `pressure_imbalance` | Combined buy/sell balance confirmation |
| Trend | `adx_14` | Regime filter — ADX < 25 = ranging (already used) |
| Trend | `bmsb_bullish` | Macro long gate: price above 20-SMA and 21-EMA |
| Market regime | `ret_zscore_20` rolling | Risk-on/off proxy (informative/filter only) |

---

## 5. Parameters to Sweep

| Parameter | Type | Low | High | Notes |
|-----------|------|-----|------|-------|
| `vwap_deviation_pct` | float | 1.5 | 5.0 | Wider deviation = rarer, higher-quality signals. Current 1.5% is too tight. |
| `rsi_oversold` | float | 20.0 | 38.0 | Tighter = fewer but stronger; looser = more trades. |
| `adx_max` | float | 18.0 | 35.0 | Controls ranging-regime filter width. |
| `volume_spike_threshold` | float | 1.2 | 3.0 | Volume ratio to confirm panic selling. |
| `wick_ratio_min` | float | 0.3 | 0.7 | Minimum buying_pressure for wick rejection confirmation. |
| `stop_loss_atr_mult` | float | 1.5 | 4.0 | ATR multiple for stop below entry. |
| `take_profit_atr_mult` | float | 1.0 | 3.5 | ATR ceiling TP above entry (secondary to VWAP target). |
| `time_stop_hours` | int | 12 | 72 | Max hold duration for non-moving reversion trades. |
| `require_volume_spike` | bool | 0 | 1 | Toggle volume filter (evolution decides). |
| `require_wick_rejection` | bool | 0 | 1 | Toggle wick filter (evolution decides). |
| `bb_bw_pct_gate` | float | 20.0 | 60.0 | Maximum bb_bw_percentile to allow entry (volatility compression gate). |
| `require_bmsb_bullish` | bool | 0 | 1 | Toggle macro regime gate (evolution decides). |

---

## 6. Parameters NOT to Optimize

| Parameter | Reason |
|-----------|--------|
| `rsi_overbought` | The short branch is fully removed; this param is obsolete. |
| `allowed_regimes` (string list) | Regime-label-dependent and non-differentiable; replace with quantitative gates (ADX, bb_bw_percentile, bmsb_bullish). |
| `require_regime` (boolean) | Replaced by individual quantitative feature gates. |
| Signal `strength` formula divisor (`/3`) | Optimizing the scaling constant within a single param creates false precision; the genome already controls position size via other params. |

---

## 7. Long-Only Risk Management

- **Stop:** `stop_loss = entry_price - stop_loss_atr_mult * atr_14`. Range 1.5-4.0x ATR. Hard floor: if `close` falls below the 20-bar low, stop triggers regardless of ATR stop.
- **Take-profit:** Primary: `vwap_20` (price reverts to VWAP). Secondary ceiling: `entry_price + take_profit_atr_mult * atr_14`. If VWAP is further than ATR-based TP, use ATR-based TP to avoid holding through reversals.
- **Trailing:** Once unrealized gain > 0.5x ATR, trail stop at `entry_price + 0.3x ATR` (locks minimum profit, prevents full round-trip).
- **Volatility targeting:** Size = `equity * (0.15 / max(annual_vol_estimate, 0.05))`, capped at `max_position_pct = 0.20`. Halves position in high-vol regimes (gk_vol above 80th percentile).
- **Max position size:** 20% of equity per trade. Mean-reversion strategies have negative-skew risk; smaller size limits ruin.
- **Cooldown:** 4 hours minimum between trades via RiskManager. After any stop-loss event, mandatory 12-hour cooldown before next entry.
- **Risk-off gate:** No new entry if ANY of: (1) `adx_14 > adx_max`; (2) `bb_bw_percentile > bb_bw_pct_gate`; (3) `bmsb_bullish == False` when `require_bmsb_bullish = True`; (4) RiskManager circuit breaker active (equity drawdown > `max_drawdown_pct`); (5) active cooldown period.

---

## 8. Metrics

| Metric | Target / Notes |
|--------|---------------|
| CAGR (annualized) | > 15% on pre-lockbox walk-forward OOS |
| Sharpe ratio | > 1.0 walk-forward; this strategy's small-move nature requires Sharpe to confirm signal-to-noise |
| Sortino ratio | > 1.5 (prioritized over Sharpe given asymmetric loss profile of mean-reversion) |
| Maximum drawdown | < 25% full-sample |
| Calmar ratio | > 0.6 |
| Hit rate | > 55% (mean-reversion must win majority of trades for positive expectancy) |
| Average trade duration | 12-36 hours target; > 72 hours suggests failed reversions held too long |
| Trade frequency | 4-20 trades/month; fewer risks stat insignificance; more risks cost drag |
| Exposure time | 10-30% in-position |
| Cost sensitivity | Measure net Sharpe at 0bp, 5bp, 10bp, 20bp; edge must survive 10bps round-trip |
| Walk-forward OOS degradation | < 50% Sharpe degradation IS to OOS |
| Bootstrap / MC | 1000-resample distribution; 5th-percentile Sharpe > 0.3 |
| Per-year stability | Positive Sharpe in majority of years 2019-2023 (pre-lockbox), especially ranging years |
| Min trades per fold | > 30 trades for statistical validity; folds with fewer are excluded from aggregate |

---

## 9. Sweep Changes to Implement

1. **Remove the short branch** (lines 78-95 of `vwap_reversion.py`) entirely. All overbought/above-VWAP conditions result in `return None` (no signal, exit to cash if in position).
2. **Add `bb_bw_pct_gate` parameter** with entry gate: `if current.get("bb_bw_percentile", 100) > p["bb_bw_pct_gate"]: return None`.
3. **Add `require_bmsb_bullish` parameter**: when True, gate on `current.get("bmsb_bullish", False)`.
4. **Widen `vwap_deviation_pct` to 1.5-5.0** in the genome param_space.
5. **Add `take_profit_atr_mult`** to the genome param_space (1.0-3.5). Implement dual TP logic: `take_profit = min(vwap, entry_price + take_profit_atr_mult * atr)`.
6. **Add `obv_divergence` optional filter**: when `require_obv_divergence=True`, gate on `current.get("obv_divergence", 0) > 0`.
7. **Add `ret_zscore_gate`** parameter: require `ret_zscore_20 < -ret_zscore_gate` (range 0.5-2.0). Filters out trivial sub-threshold moves.
8. **Fix the naming inconsistency**: original line 44 fetches `body_wick_ratio` as `body_wick` but the wick filter actually uses `buying_pressure`. Standardize to `buying_pressure` throughout and document that `body_wick_ratio` (available in features) is a different feature (body-to-range ratio, not lower wick size).
9. **Raise warm-up guard to bar_idx < 120** to match the 120-bar embargo and ensure `bb_bw_percentile` (100-bar lookback) is fully populated before any signal.
10. **Re-add `vwap_reversion` to STRATEGY_REGISTRY** with the new param_space after implementation. Remove the `allowed_regimes` and `require_regime` params from PARAMS dict; replace with `bb_bw_pct_gate` and `require_bmsb_bullish`.

---

## 10. Methodological Risks

1. **Strong negative prior from Round-1 0% fitness.** Even with long-only conversion, this strategy must be treated as a hypothesis test. If it shows positive fitness in the Phase-3 sweep, treat results skeptically and require higher statistical thresholds before promotion.
2. **VWAP window sensitivity to timeframe.** A 20-bar VWAP on 1h data is an 80-bar VWAP on 4h data. The deviation threshold must be calibrated relative to the actual timeframe. Document and fix the timeframe before sweep.
3. **Regime label dependency eliminated but ADX/bb correlation risk remains.** ADX and bb_bw_percentile are both computed from the same price data as VWAP and RSI. Optimizing all thresholds simultaneously in a genetic sweep can select parameter combinations that exploit training-set correlation structures rather than genuine regime states.
4. **Mean-reversion in crypto is asymmetric.** Oversold bounces are far more reliable in bull markets. Without the BMSB macro gate, the strategy's worst losses cluster in bear markets where RSI oversold can persist for weeks (2022 data). The BMSB gate is not optional for Phase-3 viability.
5. **Small-N per fold.** The triple-filter (RSI < 30 + VWAP -2%+ + volume spike) is highly restrictive. A 15-day walk-forward test window may see 2-8 trades. Fold-level Sharpe with N<10 is statistically meaningless. Enforce a minimum of 30 trades per fold; folds with fewer should be discarded from the aggregate. Consider longer fold windows (30-60 days) for this strategy.
6. **TP set at VWAP creates R-multiple inversion risk.** If VWAP has drifted 6% above entry during a sustained sell-off, the TP is 6% away while the stop is 2-3 ATR (~3-4%) below. The resulting R-multiple < 1 destroys expected value. The ATR-based secondary TP ceiling is essential and must be validated as a hard constraint, not an optional param.
7. **Cost sensitivity.** Expected gross edge per trade is 1.5-4%. At 10bps round-trip cost on a CEX, 20-50 trades/month generates 2-5% annual cost drag. Thin mean-reversion edges erode quickly. Validate that net Sharpe remains positive at 20bps stress-test.
8. **OOS lockbox (2024-2026) covers a mixed market regime.** 2024 was a strong bull, 2025-2026 contains unknown regime shifts. A strategy that only works in bull-market lateral regimes will pass the lockbox on the 2024 portion and fail on the 2025+ portion, creating a misleading partial-period pass. Per-year breakdowns within the OOS lockbox are essential.
9. **Feature availability check:** The strategy references `body_wick_ratio` (line 44) and `buying_pressure` (line 63) — both exist in `microstructure_features()`. However, `bb_bw_percentile` requires `bb_bw_percentile` (note: feature column in `build_all_features()` is `bb_bw_percentile`). Verify the column name matches exactly before sweep launch to avoid silent NaN gates.
