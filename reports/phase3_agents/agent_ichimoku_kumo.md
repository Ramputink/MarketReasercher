# Ichimoku Kumo Strategy — Phase 3 Spot Long-Only Research Report

## 1. Diagnosis

**Strategy file:** `/strategies/ichimoku_kumo.py`
**Registry entry:** `auto_evolve.py` lines 148-162

The strategy implements a full Ichimoku cloud breakout system with crypto-adapted parameters (classic 9-26-52 replaced by 10-34-59 with displacement=30). All Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) are computed internally via a rolling Donchian midline helper (`_donchian_mid`), independent of the pre-built feature pipeline.

**Current logic summary:**
1. Compute Tenkan (10-bar Donchian mid), Kijun (34-bar), Senkou B (59-bar), and Senkou A = (Tenkan+Kijun)/2, all displaced by 30 bars backward to get "current cloud."
2. Detect a bullish breakout (close crosses above cloud top) or bearish breakout (close crosses below cloud bottom) on the transition from previous bar.
3. Optional Chikou confirmation (require_chikou_confirm=False in evolved PARAMS — disabled).
4. Optional TK cross (require_tk_cross=True — active): for longs, requires tenkan > kijun either as a fresh cross or current alignment; for shorts, inverse.
5. ADX gate: adx_14 > 12.1 (very weak filter).
6. Regime gate: only ["trend", "breakout"].
7. Signal strength scaled by cloud thickness and ADX.

**CRITICAL: The strategy emits SHORT signals.** Line 138 sets `side = "short"` for bearish breakouts. Lines 151-153 compute inverted stop/TP for shorts. This must be entirely removed for Phase 3.

**Evolved PARAMS** (gen 183, 1074 evals, WF Sharpe=2.28, 106 trades): These metrics are contaminated by the short branch — the 2.28 WF Sharpe cannot be attributed to the long-only edge alone.

**Core edge:** Cloud breakouts in crypto capture genuine trending momentum. The cloud is a thick dynamic S/R zone; a confirmed close above it after a period of price consolidation (cloud thickness) represents a structural regime change from range to markup. This is a well-studied edge in trending markets.

**Weakness:** Pure breakout logic with no mean-reversion awareness, high component latency (30-bar displacement), and a very weak ADX filter (12.1) that barely excludes choppy conditions.

---

## 2. Spot Long-Only Edge Hypothesis

In crypto spot markets, upward Kumo breakouts reliably precede sustained trending moves. The cloud condenses months of price action into a single dynamic support structure, so a confirmed close above the cloud top (especially when: the cloud is thick, ADX is rising, volume is expanding, and the macro trend is bullish via BMSB) identifies the transition from range/distribution to markup phase.

**Hypothesis:** Enter long when close breaks above cloud top with TK alignment and regime = trend/breakout; hold with an ATR-based stop (optionally trailing to cloud top once profit margin is established); exit at TP, time stop, or stop loss. Stay flat (cash) in all other conditions, including what would have been bearish breakout entries.

The bullish cloud scenario aligns with the crypto risk-on cycle that dominates spot returns. Historical bull phases (Q4 2020, 2021, 2023, 2024-2025) produce multiple valid cloud breakout entries that compound well in the long-only framework. Bear phases (2022) result in cash periods or stop-loss exits that protect capital.

---

## 3. Market Techniques

1. **Kumo breakout momentum** — enter long only on confirmed close above cloud top; treat bearish breakout as flat signal (exit to cash).
2. **Senkou Span B as dynamic trailing stop** — once unrealized PnL > 2x ATR, trail stop to cloud_top_current (the live cloud bottom is strong structural support after a valid breakout).
3. **TK cross as entry quality filter** — require tenkan > kijun at the entry bar; this confirms momentum alignment within the Ichimoku system.
4. **Cloud thickness as position-sizing signal** — thicker cloud = stronger structural break = larger size fraction (already partially implemented via strength scalar).
5. **Regime gating** — restrict entries to trend/breakout regimes; add explicit bear-market gate via BMSB.
6. **ADX trend-strength filter** — raise threshold from evolved 12.1 to at least 15-20 to require genuine trending conditions before entry.
7. **Volume spike confirmation** — volume_ratio > 1.2-1.5 at the breakout bar validates institutional participation.
8. **Chikou span confirmation** — current close above the close from 30 bars ago confirms broader trend alignment; worth sweeping as a quality gate.
9. **Time stop discipline** — 72h cap prevents trades from becoming open-ended drawdowns in ranging conditions after a false breakout.
10. **ATR-based initial stop below cloud bottom** — structurally motivated rather than arbitrary; cloud_bottom - buffer anchors the stop to the violated structure.

---

## 4. Useful Features

| Category | Feature | Role |
|---|---|---|
| Trend | `adx_14` | Primary trend-strength gate; raise min to 15-20 |
| Trend | `roc_12` | Confirms momentum direction at breakout bar |
| Trend | `bmsb_bullish` | Macro long-only gate; block entries when below BMSB |
| Volatility | `atr_14` | Stop placement and position sizing denominator |
| Volatility | `bb_bw_percentile` | Low percentile = pre-breakout compression = higher quality |
| Volatility | `gk_vol` | Garman-Klass vol for volatility-targeted sizing |
| Volume | `volume_ratio` | Breakout volume confirmation gate |
| Volume | `volume_zscore` | Spike detection; add to strength scalar |
| Volume | `obv_divergence` | OBV trend before price breakout = leading confirmation |
| Momentum | `rsi_14` | Avoid overbought entries (rsi > 75 after breakout = chasing) |
| Momentum | `ret_zscore_20` | High positive z at entry = extended; reduce size or skip |
| Momentum | `price_acceleration` | Positive second derivative at cloud break is bullish |
| Regime | `bmsb_bullish` (as regime gate) | Macro bear filter |
| Regime | `dist_from_high_20` | Close near 20-bar high at breakout confirms strength |
| Mean-reversion | `bb_pct_b` | Secondary awareness: price not at extreme upper BB before entry |
| Microstructure | `pressure_imbalance` | Buying pressure dominance at breakout bar |
| Microstructure | `close_location` | High close_location (close near bar high) on breakout = bullish |

**Informative-only (not PnL drivers):** No funding features are used. BMSB and regime are informative filters only.

---

## 5. Params for the Sweep/Genome

| Param | Type | Low | High | Notes |
|---|---|---|---|---|
| `tenkan_period` | int | 7 | 14 | Faster Tenkan captures crypto momentum; tight range around evolved 10 |
| `kijun_period` | int | 22 | 38 | Medium-term equilibrium; must remain > tenkan |
| `senkou_b_period` | int | 44 | 65 | Long cloud boundary; ~2x kijun conceptually |
| `displacement` | int | 22 | 34 | Cloud lag; ideally tracks kijun_period |
| `adx_min` | float | 15.0 | 30.0 | Raise floor from evolved 12.1; require genuine trend |
| `stop_loss_atr_mult` | float | 1.5 | 4.0 | Stop width; lower bound suits confirmed breakouts |
| `take_profit_atr_mult` | float | 3.0 | 8.0 | TP; wide suits trending regime |
| `require_chikou_confirm` | bool | 0 | 1 | Toggle Chikou confirmation |
| `require_tk_cross` | bool | 0 | 1 | Toggle concurrent TK cross |
| `require_volume` | bool | 0 | 1 | Volume spike confirmation toggle |
| `volume_threshold` | float | 0.8 | 2.0 | Only active when require_volume=True |
| `time_stop_hours` | int | 48 | 120 | Trade duration cap; not currently swept |

---

## 6. Params NOT to Optimize

- **`allowed_regimes`** — hardcoded to ["trend","breakout"] for long-only; changing the set has only 3-4 meaningful combinations and massively overfits to specific OOS periods.
- **`require_trend_regime`** — must be locked True for long-only spot; toggling it off removes the regime gate and invites whipsaws.
- **`displacement`** — tightly coupled to `kijun_period` by Ichimoku theory; if swept independently it will find pathological combinations (e.g., displacement=22 with kijun=38) that are structurally invalid curve-fit artifacts.
- **`senkou_b_period`** — should remain approximately 2x `kijun_period`; independent sweeping breaks the structural relationship and produces non-Ichimoku hybrids.

---

## 7. Long-Only Risk Management

- **Stop loss:** ATR hard stop at `cloud_bottom - 0.5 * atr_14` at entry (structurally motivated); fallback to `close - stop_loss_atr_mult * atr_14` if cloud_bottom is above entry. Minimum stop distance = 1.5 * atr_14.
- **Take profit:** Fixed `close + take_profit_atr_mult * atr_14`; evolved 6.82x ATR suits trending regimes; sweep 3.0-8.0x.
- **Trailing stop:** Optional — once unrealized gain > 2 * atr_14, raise stop to current cloud_top value; locks in the cloud structure as a trailing floor, conceptually grounded in Ichimoku methodology.
- **Volatility targeting:** Size via `gk_vol`: `size = equity * (0.20 / current_annual_vol)`; cap at max_position_pct. Naturally reduces exposure in volatile regimes.
- **Max position size:** 40% of equity per trade maximum; signal strength scalar (0.6-1.0) reduces this further based on cloud thickness and ADX.
- **Cooldown:** Minimum 12h after any stop_loss exit; prevents re-entry into the same failed breakout.
- **Risk-off regime gate:** Block all new longs when: (1) `bmsb_bullish == False` for 3 consecutive daily periods, (2) `regime` not in ["trend","breakout"] (already implemented), (3) `adx_14 < adx_min` (already implemented), (4) `dist_from_high_20 < -0.20` suggests deep recent drawdown.

---

## 8. Relevant Metrics

| Metric | Target | Notes |
|---|---|---|
| CAGR (annualized) | 25-60% | Primary return target for crypto trend strategy |
| Sharpe ratio | > 1.0 | WF-reported 2.28 was with shorts; long-only expect 1.0-1.8 |
| Sortino ratio | > 1.5 | More appropriate for asymmetric crypto returns |
| Maximum drawdown | < 35% | Should beat buy-and-hold max DD |
| Calmar ratio | > 0.5 | CAGR / maxDD composite |
| Hit rate | 35-50% | Wide TP + tight SL; profit factor compensates low hit rate |
| Profit factor | > 1.4 | Current 1.30 with shorts; long-only may improve |
| Trade count | > 40 long-only | Dropping short branch may halve trade count; minimum for significance |
| Average trade duration | < 60h median | Distinguish TP exits from time stops |
| Market exposure | 15-40% | Higher exposure = more beta-like; balance exposure vs selectivity |
| Cost sensitivity | Stable at 5x costs | Cloud breakouts are infrequent; costs manageable |
| Bootstrap Sharpe 5th pctile | > 0.8 | Verify at 1000 resamples |
| Walk-forward fold consistency | 6/8 folds positive Sharpe | On pre-2024 history only |
| Per-year return stability | No single year > 50% of total | 2021/2024 bull years should not dominate |

---

## 9. Concrete Changes to Implement in the Sweep

1. **Remove the short branch entirely:** Delete bearish_breakout detection (line 113), its inclusion in the compound condition (line 115), short-side Chikou logic (lines 126-127), short-side TK check (lines 135-136), and short stop/TP arithmetic (lines 151-153). Add `if not bullish_breakout: return None` after breakout detection.

2. **Fix side assignment:** Line 138 must become unconditionally `side = "long"`.

3. **Add BMSB macro gate:** Before bullish_breakout check, add: `if not current.get("bmsb_bullish", True): return None`.

4. **Add `time_stop_hours` to param_space** in auto_evolve.py with range (48, 120) as int — currently hardcoded and not swept.

5. **Add `volume_threshold` to param_space** as float (0.8, 2.0) — currently fixed at 0.8 when require_volume=True.

6. **Raise `adx_min` param_space floor** from (10.0, 25.0) to (15.0, 30.0) — the evolved 12.1 is too permissive for a trend filter.

7. **Remove `senkou_b_period` and `displacement` from independent sweep** — or add a constraint: `displacement == kijun_period` and `senkou_b_period == 2 * kijun_period` (rounded), then only sweep `kijun_period`.

8. **Add `volume_zscore` to strength scalar:** When `volume_zscore > 1.5`, add 0.1 to base strength of 0.6, so position sizing naturally reflects breakout quality.

9. **Lock `allowed_regimes` and `require_trend_regime`** — remove from param_space to prevent genome from disabling the regime gate.

10. **Re-optimize entirely on pre-2024 history:** The current evolved PARAMS were finalized on 2026-03-22, within the OOS lockbox window. A clean re-run on 2019-2024 history is required before any OOS evaluation is valid.

---

## 10. Methodological Risks

1. **Short-branch contamination of reported metrics:** WF Sharpe=2.28 and PF=1.30 include short trades. Long-only backtest will have materially different statistics. Do not use existing benchmarks for comparison.

2. **OOS lockbox violation:** Evolved PARAMS dated 2026-03-22 fall inside the 2-year OOS window (2024-06-07 to 2026-06-07). Any final OOS evaluation with current PARAMS is already compromised. Must re-optimize cleanly on pre-2024 data.

3. **Displacement lag ambiguity:** The code uses `senkou_a/b` at index `[last - disp]` to represent the current cloud. This is the correct Ichimoku interpretation (cloud was computed disp bars ago for the current bar), but it is non-obvious and should be validated against a reference implementation (TradingView). An error here would silently misalign the breakout signal.

4. **Per-bar Donchian recomputation:** `_donchian_mid` recomputes all rolling values from scratch on every strategy call. This is O(n*period) per bar and correct but very slow for genetic sweeps with many param configurations. Pre-computing in `build_all_features()` with fixed params is not viable (params change per genome). Consider caching or numba JIT for sweep runs.

5. **Non-standard ADX implementation:** `features.py` uses EMA-smoothed DM (not Wilder smoothing). This produces different numerical levels than standard ADX. The `adx_min` threshold calibrated on this implementation cannot be compared to textbook Ichimoku ADX thresholds.

6. **Cloud thickness threshold not evolved:** The 1% absolute threshold (`cloud_thickness > 1.0`) is hardcoded. This may be too tight for early 2019-2020 history (high ATR%) or too loose in mature markets. Should be expressed as a percentile of recent cloud thickness distribution.

7. **Low trade count after dropping shorts:** If 106 historical trades split roughly 50/50 between long and short signals, the long-only subset has ~50 trades. Bootstrap confidence intervals will be very wide, and per-year fold stability cannot be reliably assessed with fewer than 8-10 trades per year.

8. **BMSB feature calibration mismatch:** `features.py` BMSB uses 20/21 *bar* periods. For hourly data this equals ~20 hours, not 20 weeks as originally designed. The `bmsb_bullish` signal at hourly resolution is essentially a very short-term moving average crossover, not a macro trend indicator. Verify or recompute with correct weekly-equivalent periods (~3360/3528 bars for hourly).

9. **Parameter interdependence not enforced:** The genome can independently sample `tenkan_period=12, kijun_period=22, senkou_b_period=44, displacement=22`. This produces a degenerate Ichimoku where displacement equals kijun and senkou_b is only 2x tenkan. The optimizer will find such configurations accidentally profitable on specific training windows.

10. **Regime classifier lag:** The `regime` parameter is produced by an external classifier. If the regime flip (e.g., "trend" detection) lags the actual market regime change by 5-10 bars, the regime gate adds latency to entries but also to exits — potentially allowing the strategy to remain long in a deteriorating regime. Validate the regime classifier's lag empirically.

---

**Classification: CANDIDATE** — The long-only Kumo breakout edge is real and well-supported: cloud breakouts capture crypto trending momentum with structural logic. However, the strategy requires three non-trivial fixes before it can be evaluated as long-only (removing the short branch, re-optimizing on clean pre-2024 history, and resolving the BMSB calibration issue). The core mechanics are sound, the trade count after short removal may be borderline for statistical robustness, and the displacement/senkou_b interdependency must be constrained in the genome to prevent degenerate configurations. With these fixes applied, this is a legitimate Phase 3 candidate.
