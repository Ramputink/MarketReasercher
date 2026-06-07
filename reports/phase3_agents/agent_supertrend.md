# SuperTrend Strategy — Phase-3 Spot Long-Only Research Report

**Strategy file:** `/strategies/supertrend.py`
**Classification:** DOUBTFUL
**Date:** 2026-06-07

---

## 1. Diagnosis

### Current Logic
The strategy computes the classic ATR-based SuperTrend indicator from scratch at every evaluation bar (using a capped 200-bar lookback). It fires **only on a direction flip** — when the SuperTrend switches from bearish (-1) to bullish (+1) or vice versa. Three pre-entry gates filter signal quality:

1. **Regime gate:** `_regime` must be in `["trend", "breakout"]`
2. **ADX floor:** `adx_14 >= adx_min` (evolved: 17.2 — barely above average)
3. **Volume floor:** `volume_ratio >= volume_threshold` (evolved: 1.06 — nearly always true)

**Signal strength** scales with ADX > 30 (+0.15) and volume_ratio > 1.5 (+0.10), capped at 1.0.

The evolved PARAMS after ~300 generations and 1,019 evaluations:
- `multiplier = 3.8957` — wide band, reducing flip frequency
- `stop_loss_atr_mult = 3.03`, `take_profit_atr_mult = 5.84` — 1.93 R:R
- `time_stop_hours = 48`

**Reported IS performance:** Sharpe 1.20, Walk-Forward Sharpe 1.10, 69 trades.

### Emits Shorts?
**YES.** Lines 132-146 explicitly construct a `Signal(side="short")` when `dir_now == -1`. The short branch computes `stop_loss = close + stop_loss_atr_mult * atr` and `take_profit = close - take_profit_atr_mult * atr`. This is fully active and **must be removed** for Phase-3 compliance.

### Likely Edge
Bullish SuperTrend flips in strongly trending markets (ADX > 17, regime = trend/breakout) represent a lagging trend-resumption signal. The edge, if any, is capturing post-flip continuation as the dynamic ATR band provides dynamic support. The wide multiplier (3.9) makes flips relatively rare and therefore somewhat meaningful when they occur.

### Likely Weakness
- **0.3% positive-fitness rate** from 36,000 genetic evaluations — the second-lowest of all pruned strategies (pruned in Round 2). This is the most critical finding.
- Both evolved filter thresholds (ADX 17.2, vol_ratio 1.06) are near-trivial, suggesting the GA found no meaningful filtering threshold and simply maximized trade count within a weak signal.
- 69 IS trades with Sharpe 1.20 has enormous sampling uncertainty — a handful of trend trades in bull runs can explain the entire edge.
- SuperTrend direction flips are inherently lagging: price has already moved significantly before the signal fires. Entry at bar N+1 open (1-bar lag in backtester) adds further delay.

---

## 2. Spot Long-Only Edge Hypothesis

**Hypothesis:** Bullish SuperTrend flips (direction -1 → +1) that co-occur with (a) confirmed ADX trend strength > 20, (b) volume expansion above 20-bar average, (c) price structurally above the Bull Market Support Band (BMSB), and (d) BB bandwidth percentile > 40 (volatility expanding, not compressing) represent genuine trend-resumption entries in spot crypto markets. The flip signals that price has reclaimed a dynamic ATR-based support band after a pullback, and the confluence of trend strength, volume, structural position, and volatility expansion suggests the move has momentum. The long-only variant exits to cash on bearish flips (treating them as signals to close) rather than initiating shorts.

**Confidence:** LOW. The 0.3% positive-fitness prior is a strong empirical constraint. The hypothesis requires additional confirmation filters to have any viable chance of OOS profitability.

---

## 3. Market Techniques

| Technique | Purpose |
|---|---|
| ATR-trailing SuperTrend flip (bullish only) | Lagging trend-resumption entry on dynamic band reclaim |
| ADX strength gate | Suppress whipsaws in ranging markets |
| BB bandwidth percentile expansion filter | Require volatility expansion at flip — reduces noise |
| BMSB structural bull filter | Only trade in macro uptrend phases |
| Volume ratio + z-score confirmation | Require institutional participation at flip bar |
| ATR-scaled SL and TP | Adaptive risk levels to realized volatility |
| RSI overbought cap | Prevent buying into exhaustion tops |
| Time stop (48-120h) | Cap exposure when trend fails to develop |
| Post-stop cooldown (6-24h) | Avoid re-entry into whipsawing markets |
| Walk-forward expanding-window IS optimization | Prevent lockbox contamination |

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Trend | `adx_14` | Primary trend-strength gate (keep in sweep) |
| Trend | `bmsb_bullish` (bool) | Structural macro bull filter |
| Trend | `bmsb_spread` | SMA-EMA spread width as secondary trend quality |
| Volatility | `atr_14` | SuperTrend band computation and SL/TP sizing |
| Volatility | `bb_bw_percentile` | Volatility expansion confirmation at entry |
| Volatility | `gk_vol` | Alternative vol-regime filter (GK more stable than realized vol) |
| Volatility | `vol_ratio` (short/long realized vol) | Vol regime: rising ratio = expanding regime |
| Volume | `volume_ratio` | Existing filter; consider raising threshold |
| Volume | `volume_zscore` | Volume spike detection for strong confirmation |
| Volume | `obv_divergence` | Secondary volume trend filter |
| Momentum | `rsi_14` | Overbought guard at entry |
| Momentum | `roc_12` | Rate of change — require positive (upward acceleration) |
| Momentum | `price_acceleration` | Second derivative: positive at flip confirms impulse |
| Microstructure | `pressure_imbalance` | Buying pressure dominance at flip bar |
| Microstructure | `close_location` | Bar closes in upper half at flip confirms bullish bar |
| Mean-reversion | `dist_from_high_20` | Avoid entries very near 20-bar high (stretched) |
| Regime | `_regime` | Existing gate (trend/breakout only) |

---

## 5. Parameters That Should Enter the Sweep/Genome

| Parameter | Type | Low | High | Notes |
|---|---|---|---|---|
| `multiplier` | float | 2.0 | 5.0 | Core sensitivity; current 3.9 near high end |
| `atr_period` | int | 7 | 21 | ATR smoothing; sweep around 14 |
| `adx_min` | float | 15.0 | 35.0 | Current 17.2 is near-trivial; needs real range |
| `volume_threshold` | float | 0.8 | 2.0 | Current 1.06 near-trivial; explore stricter |
| `stop_loss_atr_mult` | float | 1.5 | 4.0 | Tighten floor vs current 3.0 |
| `take_profit_atr_mult` | float | 3.0 | 8.0 | Must remain > SL mult |
| `time_stop_hours` | int | 24 | 120 | Explore longer trends |
| `rsi_overbought_cap` | float | 65.0 | 80.0 | NEW: block entries at RSI exhaustion |
| `bb_bw_pct_min` | float | 30.0 | 60.0 | NEW: require vol expansion at flip |
| `require_bmsb` | bool | 0 | 1 | NEW: BMSB structural bull filter |

---

## 6. Parameters That Should NOT Be Optimized

| Parameter | Reason |
|---|---|
| `allowed_regimes` | Letting the GA choose regime sets creates regime-label overfitting; fix to `["trend", "breakout"]` |
| `require_trend_regime` | Always True; removing this turns the strategy into an always-on flipper overfit to IS trends |
| `require_volume` | Always True for spot long-only; volume confirmation is a fundamental quality control |
| `atr_period` jointly with `multiplier` | These two are collinear (wider period ≈ larger effective multiplier); sweeping both creates a degenerate solution space |

---

## 7. Long-Only Risk Management

**Stop-loss:** Hard ATR stop at `entry_price - stop_loss_atr_mult * atr_14`. Checked against bar low each bar. Triggers full exit to cash.

**Take-profit:** Fixed ATR target at `entry_price + take_profit_atr_mult * atr_14`. Checked against bar high. Wide targets (5-8x ATR) preferred to let trends run.

**Trailing stop (optional):** After unrealized gain exceeds 1.5x ATR, trail the stop at 1.0x ATR below the rolling high-water mark. Implemented externally as a per-bar update since the current backtester does not natively support trailing stops.

**Volatility targeting:** Use `volatility_scaled` sizing method. Target ~15-20% annualized portfolio vol. `position_size = equity * (vol_target / gk_vol_annualized)`, capped at `max_position_pct = 0.25`. This prevents oversizing when ATR is compressed.

**Maximum position size:** 25% of equity per trade. Spot only, no leverage, no borrow.

**Cooldown:** 6-hour minimum cooldown after any stop-loss exit (via `min_time_between_trades` in RiskConfig). Extend to 24 hours after two consecutive stop-outs in the same direction.

**Risk-off gate:** No new entries when `_regime in ["bear", "range"]`, or when `adx_14 < 15` for the last 5 bars, or when `bmsb_bullish = False` (if `require_bmsb = True`). Circuit breaker halts all new entries if portfolio drawdown exceeds 15% from peak (existing RiskManager logic).

---

## 8. Relevant Metrics

| Metric | Target / Notes |
|---|---|
| CAGR (OOS 2024-2026) | > 15% annualized |
| Sharpe ratio | > 1.0 (OOS); skeptical if < 0.8 |
| Sortino ratio | > 1.3 (more relevant for long-only) |
| Maximum drawdown | < 20% OOS |
| Calmar ratio | > 0.5 |
| Hit rate | 35-50% acceptable for trend following with wide TP |
| Avg win / avg loss | > 1.8 to compensate for sub-50% hit rate |
| Turnover | 3-8 trades/month (wide multiplier reduces frequency) |
| Exposure % | 20-40% of time in market |
| Average trade duration | Validate bimodal: quick stops vs TP hits |
| Cost sensitivity | Rerun OOS with 2x commission + slippage; edge must survive |
| WF fold Sharpe CV | < 0.5 (consistency across folds) |
| Bootstrap Sharpe 95% CI | Lower bound > 0.4 with 1,000 resamplings |
| Per-year CAGR | Positive in >= 2 of 3 years (2024, 2025, H1 2026) |
| OOS degradation % | < 50% vs IS walk-forward Sharpe |
| Monte Carlo ruin probability | < 5% across 1,000 simulations |

---

## 9. Concrete Sweep Changes

1. **Convert to long-only:** Replace lines 132-146 in `supertrend.py`. Change `side = "long" if dir_now == 1 else "short"` to `if dir_now != 1: return None` followed by `side = "long"`. Remove the short-side SL/TP block entirely.

2. **Add `rsi_overbought_cap` filter:** After ADX/volume checks, add:
   ```python
   if current.get("rsi_14", 50) > p["rsi_overbought_cap"]:
       return None
   ```

3. **Add `bb_bw_pct_min` filter:**
   ```python
   if current.get("bb_bw_percentile", 50) < p["bb_bw_pct_min"]:
       return None
   ```

4. **Add `require_bmsb` filter:**
   ```python
   if p["require_bmsb"] and not current.get("bmsb_bullish", True):
       return None
   ```

5. **Update PARAMS defaults:** Add `rsi_overbought_cap: 75.0`, `bb_bw_pct_min: 35.0`, `require_bmsb: True`.

6. **Re-add to STRATEGY_REGISTRY** with updated `param_space` covering all 10 sweepable parameters listed in Section 5.

7. **Remove `require_volume` and `allowed_regimes` from param_space** — fix them as non-evolvable constants.

8. **Set minimum trade count to 40** for this strategy in fitness evaluation to ensure statistical meaningfulness.

9. **Do NOT re-add to active registry** until a 6-month pre-lockbox validation window confirms positive IS Sharpe > 0.8 with >= 40 trades in the long-only variant. The 0.3% positive-fitness prior is a strong empirical constraint.

10. **Enforce TP > SL + 1.5** (not just 0.5 as in existing crossover logic) to ensure adequate R:R for a low-hit-rate strategy.

---

## 10. Methodological Risks

1. **Extreme survivorship bias in evolved PARAMS:** The single surviving genome from 36,000 evaluations with only 69 trades is the most likely explanation for IS Sharpe 1.20 being a sampling artifact. The 95% bootstrap CI on 69 trades will be very wide.

2. **0.3% positive-fitness rate is a strong empirical prior:** Converting to long-only does not improve the fundamental signal quality — it only removes the short branch (which was ~50% of trades). The expected positive-fitness rate for the long-only variant is unlikely to exceed 1-2% without major additional filtering.

3. **Regime label leakage:** If `_regime` labels are computed on the full dataset before walk-forward splitting, regime-based filtering constitutes a form of lookahead leakage (the regime label at test bar N was computed using information from future bars in the same rolling window). Verify that regime labels are computed causally within each fold.

4. **SuperTrend band inconsistency (capped lookback):** The 200-bar lookback cap in `_compute_supertrend` causes band values to differ from a full-history computation, especially at fold boundaries where the rolling window restarts. This creates IS/OOS inconsistency in the signal itself.

5. **Near-trivial evolved filters:** ADX floor at 17.2 and volume_ratio at 1.06 are statistically indistinguishable from "no filter." The GA selected these values not because they identify high-quality entries but because they kept more trades in the count, potentially inflating the fitness signal.

6. **Lagging entry timing:** SuperTrend flips arrive after the impulse move. Combined with the 1-bar entry lag in the backtester, the strategy buys into a trend that is already established. In fast-moving crypto markets (BTC ±5% in hours), this means entering into a partially depleted move with wide ATR bands.

7. **Insufficient trade count for statistical inference:** 69 IS trades over ~5 years on hourly data means the strategy is in cash ~97% of the time. Even if the long-only variant achieves 40+ trades, bootstrap variance on Sharpe will remain very high.

8. **Cost absorption:** 69 trades × 2 legs × ~10bps commission/leg = ~1.4% of capital consumed in fees alone. At a 5.84x ATR TP (typically 2-4% per trade), fee drag represents ~5-10% of gross PnL per trade. Any reduction in trade count from stricter long-only filters worsens this ratio.

9. **Single-asset overfitting risk:** The evolved genome was optimized on a single primary asset. The cross-asset generalisation test (cross_asset_median_sharpe) was not reported for this pruned genome, so we do not know if even the IS Sharpe generalizes to BTC/ETH or is XRP/altcoin-specific.

10. **Time stop interaction with trend dynamics:** The 48-hour time stop may systematically cut winning trend trades that need 3-7 days to develop while allowing losing trades to hit their wide ATR stops (3.0x ATR). The interaction between time stop and ATR stop in a lagging-entry strategy should be explicitly stress-tested.
